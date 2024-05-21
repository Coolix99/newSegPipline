import numpy as np
import torch
import os
from torch.utils.data import DataLoader, Dataset
from CPC.UNet3D import UNet3D
from config import model_folder_path
from IO import getImage, get_JSON

class PatchDataset(Dataset):
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
                        positions.append((z, y, x))
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

def load_model(model_path, device):
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
    reconstructed_segmentation = np.zeros(image_shape, dtype=np.int64)
    reconstructed_flows = np.zeros((3,) + image_shape, dtype=np.float32)
    counts = np.zeros(image_shape, dtype=np.int64)
    
    stride = tuple(s - o for s, o in zip(patch_size, overlap))
    for (seg_patch, flow_patch), (z, y, x) in zip(patches, positions):
        z_slice = slice(z, z + patch_size[0])
        y_slice = slice(y, y + patch_size[1])
        x_slice = slice(x, x + patch_size[2])
        
        reconstructed_segmentation[z_slice, y_slice, x_slice] += seg_patch
        reconstructed_flows[:, z_slice, y_slice, x_slice] += flow_patch
        counts[z_slice, y_slice, x_slice] += 1
    
    nonzero_mask = counts > 0
    reconstructed_segmentation[nonzero_mask] /= counts[nonzero_mask]
    reconstructed_flows[:, nonzero_mask] /= counts[nonzero_mask]
    
    return reconstructed_segmentation, reconstructed_flows

def main():
    # Load data
    example_folder_path = 'path_to_your_example_folder'  # Change this to your actual path
    MetaData = get_JSON(example_folder_path)['Example_MetaData']

    nuc_file_name = MetaData['nuc file']
    profil_file_name = MetaData['profile file']

    nuc_file_path = os.path.join(example_folder_path, nuc_file_name)
    profil_file_path = os.path.join(example_folder_path, profil_file_name)

    nuc_img = getImage(nuc_file_path)
    profil = np.load(profil_file_path)

    # Parameters
    patch_size = (64, 64, 64)
    overlap = (16, 16, 16)
    batch_size = 4

    # Create dataset and dataloader
    dataset = PatchDataset(nuc_img, profil, patch_size, overlap)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load model
    model_path = os.path.join(model_folder_path, 'checkpoint_pretraining.pth')  # Change the name if necessary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    # Process patches
    processed_patches = []
    positions = []
    for patches, profiles, pos in dataloader:
        for patch, profile in zip(patches, profiles):
            seg_patch, flow_patch = apply_model_to_patch(model, patch, profile, device)
            processed_patches.append((seg_patch, flow_patch))
        positions.extend(pos)

    # Reconstruct full image
    segmentation, pred_flows = reconstruct_image_from_patches(nuc_img.shape, patch_size, overlap, processed_patches, positions)

    # Apply mask to zero out irrelevant areas
    segmentation[nuc_img == 0] = 0
    pred_flows[:, nuc_img == 0] = 0

    # Save or display results
    np.save(os.path.join(example_folder_path, 'segmentation.npy'), segmentation)
    np.save(os.path.join(example_folder_path, 'pred_flows.npy'), pred_flows)
    print("Segmentation and predicted flows saved.")

if __name__ == "__main__":
    main()
