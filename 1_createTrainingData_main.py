import numpy as np
import napari
import os
import git
import random
import uuid
import torch
import tifffile as tiff

from config import *
from IO import *

from CPC.CPC_config import patch_size
from CPC.std_op import prepareExample
from CPC.dynamics import masks_to_flows_gpu_3d

def crop_image_at_random(nuclei,masks, size):
    """
    Crop a patch from a 3D image at a random start coordinate.

    :param nuclei: 3D array from which to crop the patch.
    :param masks: 3D array from which to crop the patch.
    :param size: List of three integers specifying the size of the patch in each dimension.
    :return: Cropped patch as a 3D array.
    """
    # Calculate the maximum valid start coordinates to ensure the patch does not go out of bounds
    max_z = nuclei.shape[0] - size[0]
    max_y = nuclei.shape[1] - size[1]
    max_x = nuclei.shape[2] - size[2]

    while True:
        start_z = random.randint(0, max_z)
        start_y = random.randint(0, max_y)
        start_x = random.randint(0, max_x)
        
        nuclei_crop=nuclei[start_z:start_z+size[0], start_y:start_y+size[1], start_x:start_x+size[2]]
        if np.sum(nuclei_crop>1e-2)>size[0]*size[1]*size[2]*0.2:
            return nuclei_crop,masks[start_z:start_z+size[0], start_y:start_y+size[1], start_x:start_x+size[2]] 

def create_unique_subfolder(base_path):
    """
    Create a unique-named subfolder in the given base directory.

    :param base_path: Path to the directory where the subfolder will be created.
    :return: The path to the newly created subfolder.
    """
    while True:
        # Generate a random UUID as the folder name
        unique_folder_name = str(uuid.uuid4())
        new_folder_path = os.path.join(base_path, unique_folder_name)
        
        # Check if this directory already exists
        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)
            print(f"Created new subfolder: {new_folder_path}")
            return new_folder_path

def calculateFlow(masks):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    res=masks_to_flows_gpu_3d(masks,device)

    # flow_vector_field = res.transpose(1, 2, 3, 0)
    # viewer = napari.Viewer()
    # viewer.add_labels(masks, name='3D Labels')
    # z, y, x = np.nonzero(masks)
    # origins = np.stack((z, y, x), axis=-1)
    # vectors = flow_vector_field[z, y, x]
    # vector_data = np.stack((origins, vectors), axis=1)
    # viewer.add_image(np.linalg.norm(flow_vector_field, axis=3), name='norm 3D Flow Field')
    # viewer.add_vectors(vector_data, name='3D Flow Field', edge_width=0.2, length=5, ndim=3)
    # napari.run()

    return res
    

def crop_trainData(nuclei_crop,modified_masks,flow, size, d=10):
    max_z = nuclei_crop.shape[0] - size[0] - d
    max_y = nuclei_crop.shape[1] - size[1] - d
    max_x = nuclei_crop.shape[2] - size[2] - d

    while True:
        start_z = random.randint(d, max_z)
        start_y = random.randint(d, max_y)
        start_x = random.randint(d, max_x)
        
        nuclei_patch=nuclei_crop[start_z:start_z+size[0], start_y:start_y+size[1], start_x:start_x+size[2]]
        if np.sum(nuclei_patch>1e-2)>size[0]*size[1]*size[2]*0.05:
            masks_patch=modified_masks[start_z:start_z+size[0], start_y:start_y+size[1], start_x:start_x+size[2]]
            flow_patch=flow[:,start_z:start_z+size[0], start_y:start_y+size[1], start_x:start_x+size[2]]
            return nuclei_patch,masks_patch,flow_patch

def createTrainingData():
    nuclei_folder_list=os.listdir(struct_nuclei_images_path)
    random.shuffle(nuclei_folder_list)
    for nuclei_folder in nuclei_folder_list:
        print(nuclei_folder)

        masks_dir_path=os.path.join(struct_masks_path,nuclei_folder+'_filtered_fp_masks')
        masks_MetaData=get_JSON(masks_dir_path)
        if masks_MetaData=={}:
            continue
        
        nuclei_dir_path=os.path.join(struct_nuclei_images_path,nuclei_folder)
        nuclei_MetaData=get_JSON(nuclei_dir_path)

        masks_file_name=masks_MetaData['masks_MetaData']['masks file']
        masks=getImage(os.path.join(masks_dir_path,masks_file_name))
        nuclei_file_name=nuclei_MetaData['nuclei_image_MetaData']['nuclei image file name']
        nuclei=getImage(os.path.join(nuclei_dir_path,nuclei_file_name))
        print(masks.shape)
        print(nuclei.shape)
        scale=np.array(nuclei_MetaData['nuclei_image_MetaData']['XYZ size in mum']).copy()
        scale[0], scale[2] = scale[2], scale[0]
        print(scale)
        
        nuclei,masks,nuclei_profile=prepareExample(nuclei,masks,scale)
        print(nuclei.shape)

        nuclei_crop,masks_crop=crop_image_at_random(nuclei,masks, 5*np.array(patch_size))
        print(nuclei_crop.shape)
        print(np.max(masks))

        viewer = napari.Viewer(ndisplay=3)
        viewer.add_image(nuclei_crop,name='nuclei_crop')
        viewer.add_labels(masks_crop,name='masks_crop')
        modified_masks=None
        @viewer.bind_key('S')
        def save_changes(viewer):
            nonlocal modified_masks
            # This function can be triggered by pressing 'S' and will save the current state of the mask.
            modified_masks = viewer.layers['masks_crop'].data
            print('Modifications saved.')
        napari.run()
        print(modified_masks.shape)
        
        flow=calculateFlow(modified_masks)

        #create N examples from this
        for i in range(20):
            nuclei_patch,masks_patch,flow_patch=crop_trainData(nuclei_crop,modified_masks,flow, np.array(patch_size))

            example_folder_path=create_unique_subfolder(trainData_path)
            print(example_folder_path)

            tiff.imwrite(os.path.join(example_folder_path,'nuclei_patch.tif'), nuclei_patch, dtype=np.float32)
            tiff.imwrite(os.path.join(example_folder_path,'masks_patch.tif'), masks_patch, dtype=np.int32)
            np.savez_compressed(os.path.join(example_folder_path,'flow_patch.npz'), flow_patch)
            np.save(os.path.join(example_folder_path,'profile.npy'),nuclei_profile)

            MetaData_Example={}
            repo = git.Repo(gitPath,search_parent_directories=True)
            sha = repo.head.object.hexsha
            MetaData_Example['git hash']=sha
            MetaData_Example['git repo']='newSegPipline'
            MetaData_Example['Example version']=Example_version
            MetaData_Example['nuc file']='nuclei_patch.tif'
            MetaData_Example['masks file']='masks_patch.tif'
            MetaData_Example['flow file']='flow_patch.npz'
            MetaData_Example['profile file']='profile.npy'
            MetaData_Example['XYZ size in mum']=nuclei_MetaData['nuclei_image_MetaData']['XYZ size in mum']
            MetaData_Example['axes']=nuclei_MetaData['nuclei_image_MetaData']['axes']
            MetaData_Example['is control']=nuclei_MetaData['nuclei_image_MetaData']['is control']
            MetaData_Example['time in hpf']=nuclei_MetaData['nuclei_image_MetaData']['time in hpf']
            MetaData_Example['experimentalist']=nuclei_MetaData['nuclei_image_MetaData']['experimentalist']
            writeJSON(example_folder_path,'Example_MetaData',MetaData_Example)

        return

        


def generate_synthetic_3d_data(shape=(64, 32, 70), num_objects=5):
    data = np.zeros(shape, dtype=np.float32)
    labels = np.zeros(shape, dtype=np.int32)
    
    # Create grid of coordinates
    x = np.arange(shape[0])
    y = np.arange(shape[1])
    z = np.arange(shape[2])
    xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
    
    for i in range(1, num_objects + 1):
        # Randomly choose the center of the object
        center = np.random.randint(10, min(shape) - 10, size=3)
        # Randomly choose the radius/size of the object
        radius = np.random.randint(5, 10)
        
        # Create a spherical object using broadcasting
        mask = (xv - center[0])**2 + (yv - center[1])**2 + (zv - center[2])**2 < radius**2
        data[mask] = 1.0
        labels[mask] = i
    
    return data, labels

def test():
    data, labels=generate_synthetic_3d_data()
   
    calculateFlow(labels)

if __name__ == "__main__":
    #test()
    createTrainingData()
