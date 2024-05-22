import numpy as np
import torch
import os
from torch.utils.data import DataLoader, Dataset

from CPC.UNet3D import UNet3D
from CPC.std_op import prepareData
from CPC.CPC_config import patch_size
from config import model_folder_path, pretrainData_path,batch_size,nuclei_folders_path,applyresult_folder_path
from IO import getImage, get_JSON

class ExampleDataset(Dataset):
    def __init__(self, image, profile, patch_size, overlap):
        self.image = image
        self.profile = profile
        self.patch_size = patch_size
        self.overlap = overlap
        self.patches, self.positions = self._extract_patches()

    def _extract_patches(self):
        patches = []
        positions = []
        stride = tuple(s - o for s, o in zip(self.patch_size, self.overlap))
        for z in range(0, self.image.shape[0] - self.patch_size[0] + 1, stride[0]):
            for y in range(0, self.image.shape[1] - self.patch_size[1] + 1, stride[1]):
                for x in range(0, self.image.shape[2] - self.patch_size[2] + 1, stride[2]):
                    patch = self.image[z:z + self.patch_size[0], y:y + self.patch_size[1], x:x + self.patch_size[2]]
                    if np.any(patch):  # Only process patches with non-zero content
                        patches.append(patch)
                        positions.append(np.array((z, y, x)))
        return patches, positions

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        position = self.positions[idx]
        combined_patch = np.expand_dims(np.expand_dims(patch, axis=0), axis=0)  # Add channel and batch dimensions
        combined_patch = np.ascontiguousarray(combined_patch)
        profile = np.ascontiguousarray(self.profile)
        return torch.from_numpy(combined_patch).float(), torch.from_numpy(profile).float(), position

def load_model(name, device):
    model_file_name = 'checkpoint_' + name + '.pth'
    model_subfolder_name = 'checkpoint_' +name
    model_subfolder_path=os.path.join(model_folder_path,model_subfolder_name)
    model_path=os.path.join(model_subfolder_path, model_file_name)

    print('load_model with metadata')
    print(get_JSON(model_subfolder_path))


    n_channels = 1  # Adjust this if your input has more channels
    context_size = 8
    model = UNet3D(n_channels, context_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def apply_model_to_patch(model, patch, profile, device):
    with torch.no_grad():
        patch = patch.to(device)
        profile = profile.to(device)
        seg_logits, pred_flows = model(patch, profile)
        segmentation = torch.argmax(seg_logits, dim=1).cpu().numpy()[0]
        pred_flows = pred_flows.cpu().numpy()[0]
    return segmentation, pred_flows

def reconstruct_image_from_patches(image_shape, patch_size, overlap, patches, positions):
    reconstructed_segmentation = np.zeros(image_shape, dtype=np.float32)  
    reconstructed_flows = np.zeros((3,) + image_shape, dtype=np.float32)
    counts = np.zeros(image_shape, dtype=np.int64)

    for (seg_patch, flow_patch), pos in zip(patches, positions):
        z, y, x = pos  # Extract the z, y, x positions
        z_slice = slice(z, z + patch_size[0])
        y_slice = slice(y, y + patch_size[1])
        x_slice = slice(x, x + patch_size[2])
        
        reconstructed_segmentation[z_slice, y_slice, x_slice] += seg_patch
        reconstructed_flows[:, z_slice, y_slice, x_slice] += flow_patch
        counts[z_slice, y_slice, x_slice] += 1
    
    nonzero_mask = counts > 0
    reconstructed_segmentation[nonzero_mask] /= counts[nonzero_mask]
    reconstructed_flows[:, nonzero_mask] /= counts[nonzero_mask]

    flow_magnitudes = np.linalg.norm(reconstructed_flows, axis=0)
    nonzero_flows_mask = flow_magnitudes > 0
    reconstructed_flows[:, nonzero_flows_mask] /= flow_magnitudes[nonzero_flows_mask]

    reconstructed_segmentation = (reconstructed_segmentation>0.49).astype(bool) 
    
    return reconstructed_segmentation, reconstructed_flows

def plot_compare(nuc_img,masks_img,flow,segmentation,pred_flows):
    import napari

    viewer = napari.Viewer()
    viewer.add_image(nuc_img, name='3D Nuc')
    viewer.add_labels(masks_img, name='3D Labels')
    z, y, x = np.nonzero(masks_img)
    origins = np.stack((z, y, x), axis=-1)
    flow_vector_field = flow.transpose(1, 2, 3, 0)
    vectors = flow_vector_field[z, y, x]
    vector_data = np.stack((origins, vectors), axis=1)
    viewer.add_vectors(vector_data, name='3D Flow Field', edge_width=0.1, length=1, ndim=3,edge_color='red')

    viewer.add_labels(segmentation, name='seg')
    z, y, x = np.nonzero(segmentation)
    origins = np.stack((z, y, x), axis=-1)
    flow_vector_field = pred_flows.transpose(1, 2, 3, 0)
    vectors = flow_vector_field[z, y, x]
    vector_data = np.stack((origins, vectors), axis=1)
    viewer.add_vectors(vector_data, name='3D Flow Field pred', edge_width=0.1, length=1, ndim=3,edge_color='blue')

    napari.run()

def test_examples():
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

        overlap=(16,16,16)        

        # Create dataset and dataloader
        dataset = ExampleDataset(nuc_img, profil, patch_size, overlap)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model('pretraining', device)
        
        # Process patches
        processed_patches = []
        positions = []
        for patches, profiles, pos in dataloader:
            for i, patch in enumerate(patches):
                profile = profiles[i]
                position = pos[i]
                seg_patch, flow_patch = apply_model_to_patch(model, patch, profile, device)
                processed_patches.append((seg_patch, flow_patch))
                positions.append(position)
        # Reconstruct full image
        segmentation, pred_flows = reconstruct_image_from_patches(nuc_img.shape, patch_size, overlap, processed_patches, positions)

        # Apply mask to zero out irrelevant areas
        segmentation[nuc_img == 0] = 0
        pred_flows[:, nuc_img == 0] = 0
        print(segmentation.shape)
        print(pred_flows.shape)
        
        plot_compare(nuc_img,masks_img,flow,segmentation,pred_flows)

def apply_model_to_data():
    nuclei_folder_list = os.listdir(nuclei_folders_path)
    for nuclei_folder in nuclei_folder_list:
        print(nuclei_folder)
        nuclei_folder_path = os.path.join(nuclei_folders_path, nuclei_folder)
        res_folder_path = os.path.join(applyresult_folder_path, nuclei_folder)



        MetaData = get_JSON(nuclei_folder_path)
        print(MetaData)
        return

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

        overlap=(16,16,16)        

        # Create dataset and dataloader
        dataset = ExampleDataset(nuc_img, profil, patch_size, overlap)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model('pretraining', device)
        
        # Process patches
        processed_patches = []
        positions = []
        for patches, profiles, pos in dataloader:
            for i, patch in enumerate(patches):
                profile = profiles[i]
                position = pos[i]
                seg_patch, flow_patch = apply_model_to_patch(model, patch, profile, device)
                processed_patches.append((seg_patch, flow_patch))
                positions.append(position)
        # Reconstruct full image
        segmentation, pred_flows = reconstruct_image_from_patches(nuc_img.shape, patch_size, overlap, processed_patches, positions)

        # Apply mask to zero out irrelevant areas
        segmentation[nuc_img == 0] = 0
        pred_flows[:, nuc_img == 0] = 0
        print(segmentation.shape)
        print(pred_flows.shape)
        
        plot_compare(nuc_img,masks_img,flow,segmentation,pred_flows)

if __name__ == "__main__":
    #test_examples()
    apply_model_to_data()

