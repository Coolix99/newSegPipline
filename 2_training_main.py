import numpy as np
import napari
import os
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import time
import git

from NucleiDataset import NucleiDataset
from config import *
from IO import *

from CPC.UNet3D import UNet3D

def plot_example(nuc_img, masks_img, flow):
    flow_vector_field = flow.transpose(1, 2, 3, 0)
    viewer = napari.Viewer()
    viewer.add_image(nuc_img, name='3D Nuc')
    viewer.add_labels(masks_img, name='3D Labels')
    z, y, x = np.nonzero(masks_img)
    origins = np.stack((z, y, x), axis=-1)
    vectors = flow_vector_field[z, y, x]
    vector_data = np.stack((origins, vectors), axis=1)
    viewer.add_image(np.linalg.norm(flow_vector_field, axis=3), name='norm 3D Flow Field')
    viewer.add_vectors(vector_data, name='3D Flow Field', edge_width=0.1, length=1, ndim=3)
    napari.run()

def save_model(elapsed_time, model,epoch, avg_train_loss, avg_val_loss, name):
    model_file_name = 'checkpoint_' + name + '.pth'
    model_subfolder_name = 'checkpoint_' +name
    model_subfolder_path=os.path.join(model_folder_path,model_subfolder_name)
    make_path(model_subfolder_path)
    torch.save(model.state_dict(), os.path.join(model_subfolder_path, model_file_name))
    print("Checkpoint saved as", model_file_name)
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    MetaData_model = {}
    repo = git.Repo(gitPath, search_parent_directories=True)
    sha = repo.head.object.hexsha
    MetaData_model['git hash'] = sha
    MetaData_model['git repo'] = 'newSegPipline'
    MetaData_model['Training version'] = Training_version
    MetaData_model['model file'] = model_file_name
    MetaData_model['elapsed_time'] = elapsed_time
    MetaData_model['avg_train_loss'] = avg_train_loss
    MetaData_model['avg_val_loss'] = avg_val_loss
    MetaData_model['epoch'] = epoch

    writeJSON(model_subfolder_path, 'Model_MetaData', MetaData_model)

def pre_train_main():
    # Collect Data
    nuc_img_list = []
    mask_img_list = []
    flow_list = []
    profil_list = []
    example_folder_list = os.listdir(pretrainData_path)
    for example_folder in example_folder_list:
        print(example_folder)
        example_folder_path = os.path.join(pretrainData_path, example_folder)
        MetaData = get_JSON(example_folder_path)['Example_MetaData']

        nuc_file_name = MetaData['nuc file']
        masks_file_name = MetaData['masks file']
        flow_file_name = MetaData['flow file']
        profil_file_name = MetaData['profile file']

        nuc_file_path = os.path.join(example_folder_path, nuc_file_name)
        masks_file_path = os.path.join(example_folder_path, masks_file_name)
        flow_file_path = os.path.join(example_folder_path, flow_file_name)
        profil_file_path = os.path.join(example_folder_path, profil_file_name)

        nuc_img = getImage(nuc_file_path)
        masks_img = getImage(masks_file_path)
        flow = np.load(flow_file_path)['arr_0']
        profil = np.load(profil_file_path)

        print(nuc_img.shape)
        print(masks_img.shape)
        print(flow.shape)
        print(profil)

        nuc_img_list.append(nuc_img)
        mask_img_list.append(masks_img > 0)
        flow_list.append(flow)
        profil_list.append(profil)

        # plot_example(nuc_img,masks_img,flow)

    dataset = NucleiDataset(nuc_img_list, mask_img_list, flow_list, profil_list)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,num_workers=n_cores)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,num_workers=n_cores)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Initialize model
    n_channels = 1  # Adjust this if your input has more channels
    context_size = 8
    model = UNet3D(n_channels, context_size).to(device)

    # Define loss functions
    class_weights = torch.tensor([1.0, 1.0], dtype=torch.float32).to(device)  # Adjust class weights as needed
    criterion_segmentation = nn.CrossEntropyLoss(weight=class_weights)

    def angle_loss(pred_flow, true_flow, mask):
        # Calculate cosine similarity
        cosine_similarity = torch.sum(pred_flow * true_flow, dim=1)
        # Angle loss
        angle_loss = 1 - cosine_similarity
        # Mask the loss to only include regions where mask is true
        angle_loss = angle_loss * mask
        # Average the loss over the masked regions
        return angle_loss.sum() / mask.sum()

    def masked_cross_entropy_loss(logits, target, mask):
        # Apply the mask
        mask = mask.float()
        loss = criterion_segmentation(logits, target.long())
        masked_loss = loss * mask
        # Average the loss over the masked regions
        return masked_loss.sum() / mask.sum()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training and validation loop
    num_epochs = 150
    best_val_loss = np.inf  # Initialize best validation loss for checkpointing
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, masks, flows, prof in train_loader:
            images, masks, flows, prof = images.to(device), masks.to(device), flows.to(device), prof.to(device)
            optimizer.zero_grad()
            seg_logits, pred_flows = model(images, prof)
            
            # Apply the mask for intensities > 0
            mask = images > 0

            loss_segmentation = masked_cross_entropy_loss(seg_logits, masks, mask)

            # Compute flow field loss
            loss_flow = angle_loss(pred_flows, flows, mask)
            
            # Total loss
            loss = loss_segmentation + loss_flow
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks, flows, prof in val_loader:
                images, masks, flows, prof = images.to(device), masks.to(device), flows.to(device), prof.to(device)
                seg_logits, pred_flows = model(images, prof)
                
                # Apply the mask for intensities > 0
                mask = images > 0

                # Compute segmentation loss
                loss_segmentation = masked_cross_entropy_loss(seg_logits, masks, mask)
                
                # Compute flow field loss
                loss_flow = angle_loss(pred_flows, flows, mask)
                
                # Total loss
                loss = loss_segmentation + loss_flow
                val_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}")

        # Checkpoint model if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            elapsed_time = time.time() - start_time
            save_model(elapsed_time, model, epoch, avg_train_loss, avg_val_loss, 'pretraining')

def train_main():
    # Collect Data
    nuc_img_list = []
    mask_img_list = []
    flow_list = []
    profil_list = []
    example_folder_list = os.listdir(trainData_path)
    for example_folder in example_folder_list:
        print(example_folder)
        example_folder_path = os.path.join(trainData_path, example_folder)
        MetaData = get_JSON(example_folder_path)['Example_MetaData']

        nuc_file_name = MetaData['nuc file']
        masks_file_name = MetaData['masks file']
        flow_file_name = MetaData['flow file']
        profil_file_name = MetaData['profile file']

        nuc_file_path = os.path.join(example_folder_path, nuc_file_name)
        masks_file_path = os.path.join(example_folder_path, masks_file_name)
        flow_file_path = os.path.join(example_folder_path, flow_file_name)
        profil_file_path = os.path.join(example_folder_path, profil_file_name)

        nuc_img = getImage(nuc_file_path)
        masks_img = getImage(masks_file_path)
        flow = np.load(flow_file_path)['arr_0']
        profil = np.load(profil_file_path)

        print(nuc_img.shape)
        print(masks_img.shape)
        print(flow.shape)
        print(profil)

        nuc_img_list.append(nuc_img)
        mask_img_list.append(masks_img > 0)
        flow_list.append(flow)
        profil_list.append(profil)

        # plot_example(nuc_img,masks_img,flow)

    dataset = NucleiDataset(nuc_img_list, mask_img_list, flow_list, profil_list)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,num_workers=n_cores)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,num_workers=n_cores)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Initialize model
    n_channels = 1  # Adjust this if your input has more channels
    context_size = 8
    model = UNet3D(n_channels, context_size).to(device)
    model_file_name = 'checkpoint_' + 'pretraining' + '.pth'
    model_subfolder_name = 'checkpoint_' +'pretraining'
    model_subfolder_path=os.path.join(model_folder_path,model_subfolder_name)
    model_path=os.path.join(model_subfolder_path, model_file_name)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Define loss functions
    class_weights = torch.tensor([1.0, 1.0], dtype=torch.float32).to(device)  # Adjust class weights as needed
    criterion_segmentation = nn.CrossEntropyLoss(weight=class_weights)

    def angle_loss(pred_flow, true_flow, mask):
        # Calculate cosine similarity
        cosine_similarity = torch.sum(pred_flow * true_flow, dim=1)
        # Angle loss
        angle_loss = 1 - cosine_similarity
        # Mask the loss to only include regions where mask is true
        angle_loss = angle_loss * mask
        # Average the loss over the masked regions
        return angle_loss.sum() / mask.sum()

    def masked_cross_entropy_loss(logits, target, mask):
        # Apply the mask
        mask = mask.float()
        loss = criterion_segmentation(logits, target.long())
        masked_loss = loss * mask
        # Average the loss over the masked regions
        return masked_loss.sum() / mask.sum()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    # Training and validation loop
    num_epochs = 150
    best_val_loss = np.inf  # Initialize best validation loss for checkpointing
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, masks, flows, prof in train_loader:
            images, masks, flows, prof = images.to(device), masks.to(device), flows.to(device), prof.to(device)
            optimizer.zero_grad()
            seg_logits, pred_flows = model(images, prof)
            
            # Apply the mask for intensities > 0
            mask = images > 0

            loss_segmentation = masked_cross_entropy_loss(seg_logits, masks, mask)

            # Compute flow field loss
            loss_flow = angle_loss(pred_flows, flows, mask)
            
            # Total loss
            loss = loss_segmentation + loss_flow
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks, flows, prof in val_loader:
                images, masks, flows, prof = images.to(device), masks.to(device), flows.to(device), prof.to(device)
                seg_logits, pred_flows = model(images, prof)
                
                # Apply the mask for intensities > 0
                mask = images > 0

                # Compute segmentation loss
                loss_segmentation = masked_cross_entropy_loss(seg_logits, masks, mask)
                
                # Compute flow field loss
                loss_flow = angle_loss(pred_flows, flows, mask)
                
                # Total loss
                loss = loss_segmentation + loss_flow
                val_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}")

        # Checkpoint model if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            elapsed_time = time.time() - start_time
            save_model(elapsed_time, model, epoch, avg_train_loss, avg_val_loss, 'training')

if __name__ == "__main__":
    #pre_train_main()
    train_main()
