import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.ndimage import rotate

import time
seed = int(time.time())
np.random.seed(seed)

def random_rotation_and_mirror(combined_patch, mask_patch, flow_patch):
    k = np.random.randint(0, 4)
    combined_patch = rotate(combined_patch, angle=90*k, axes=(2, 3), reshape=False)
    mask_patch = rotate(mask_patch, angle=90*k, axes=(1, 2), reshape=False)
    flow_patch = rotate(flow_patch, angle=90*k, axes=(2, 3), reshape=False)
    # Calculate the 2x2 rotation matrix
    cos_angle = np.cos(np.deg2rad(90*k))
    sin_angle = np.sin(np.deg2rad(90*k))
    rotation_matrix = np.array([
        [cos_angle, -sin_angle],
        [sin_angle, cos_angle]
    ])

    # Apply the rotation matrix to the flow vectors
    flow_vectors = flow_patch[1:3].reshape(2, -1)
    rotated_vectors = np.dot(rotation_matrix, flow_vectors).reshape(2, *flow_patch.shape[1:])
    flow_patch[1:3] = rotated_vectors

    if np.random.rand() > 0.5:
        combined_patch = np.flip(combined_patch, axis=2)
        mask_patch = np.flip(mask_patch, axis=1)
        flow_patch = np.flip(flow_patch, axis=2)
        flow_patch[1,:,:,:] = -flow_patch[1,:,:,:]

    if np.random.rand() > 0.5:
        combined_patch = np.flip(combined_patch, axis=1)
        mask_patch = np.flip(mask_patch, axis=0)
        flow_patch = np.flip(flow_patch, axis=1)
        flow_patch[0,:,:,:] = -flow_patch[0,:,:,:]

    

    return combined_patch, mask_patch, flow_patch

class NucleiDataset(Dataset):
    def __init__(self, nuclei_images, mask_images, flow_fields, profiles, patch_size=(64, 64, 64), min_nonzero=0.05, transform=random_rotation_and_mirror):
        """
        Args:
            nuclei_images (list): List of 3D arrays of the nuclei images.
            mask_images (list): List of 3D arrays of mask annotations.
            flow_fields (list): List of 3D arrays of flow fields.
            profiles (list): List of 1D numpy arrays, each an intensity profile.
            patch_size (tuple): Size of the patches to extract.
            min_nonzero (float): Minimum fraction of nonzero pixels for a patch to be valid.
            transform (callable): Optional transform to apply to each patch.
        """
        self.nuclei_images = nuclei_images
        self.mask_images = mask_images
        self.flow_fields = flow_fields
        self.context_vectors = profiles
        self.patch_size = patch_size
        self.min_nonzero = min_nonzero
        self.transform = transform
        self.patches = self._extract_patches()

    def _extract_patches(self):
        patches = []
        for img, mask, flow, context in zip(self.nuclei_images, self.mask_images, self.flow_fields, self.context_vectors):
            pz, py, px = self.patch_size
            nz, ny, nx = img.shape
            for z in range(0, nz - pz + 1, pz):
                for y in range(0, ny - py + 1, py):
                    for x in range(0, nx - px + 1, px):
                        img_patch = img[z:z + pz, y:y + py, x:x + px]
                        if np.sum(img_patch > 0) < int(self.min_nonzero * np.prod(self.patch_size)):
                            continue
                        mask_patch = mask[z:z + pz, y:y + py, x:x + px]
                        flow_patch = flow[:, z:z + pz, y:y + py, x:x + px]
                        patches.append((img_patch, mask_patch, flow_patch, context))
        return patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img_patch, mask_patch, flow_patch, context = self.patches[idx]
        combined_patch = np.expand_dims(img_patch, axis=0)  # Add a channel dimension
        if self.transform:
            combined_patch, mask_patch, flow_patch = self.transform(combined_patch, mask_patch, flow_patch)
        combined_patch = np.ascontiguousarray(combined_patch)
        mask_patch = np.ascontiguousarray(mask_patch)
        flow_patch = np.ascontiguousarray(flow_patch)
        context = np.ascontiguousarray(context)
        return torch.from_numpy(combined_patch).float(), torch.from_numpy(mask_patch).long(), torch.from_numpy(flow_patch).float(), torch.from_numpy(context).float()
