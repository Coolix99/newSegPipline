import numpy as np
from typing import List
import napari
import os
#import git
import pandas as pd
from simple_file_checksum import get_checksum
import random
import uuid
import torch


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
    print(res.shape)

    viewer = napari.Viewer()
    viewer.add_labels(masks, name='3D Labels')
    # Napari expects the vector field in (z, y, x, 3) shape
    flow_vector_field = res.transpose(1, 2, 3, 0)
    viewer.add_vectors(flow_vector_field, name='3D Flow Field1', edge_width=0.2, length=1)
    flow_vector_field = res.transpose(3, 2, 1, 0)
    viewer.add_vectors(flow_vector_field, name='3D Flow Field2', edge_width=0.2, length=1)
    flow_vector_field = res.transpose(2, 3, 1, 0)
    viewer.add_vectors(flow_vector_field, name='3D Flow Field3', edge_width=0.2, length=1)

    # Start the Napari viewer
    napari.run()
    

def crop_trainData(nuclei_crop,modified_masks, size):
    pass

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
        scale=np.array(nuclei_MetaData['nuclei_image_MetaData']['XYZ size in mum'])
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
        
        calculateFlow(modified_masks)

        #create 10 examples from this
        for i in range(10):
            nuclei_crop,masks_crop=crop_trainData(nuclei_crop,modified_masks, np.array(patch_size))


            example_folder_path=create_unique_subfolder(trainData_path)
            print(example_folder_path)


        return
        PastMetaData=evalStatus_orient(BRE_dir_path,LMcoord_dir_path)
        if not isinstance(PastMetaData,dict):
            continue

        make_path(LMcoord_dir_path)
        MetaData_image=PastMetaData['BRE_image_MetaData']
        writeJSON(LMcoord_dir_path,'BRE_image_MetaData',MetaData_image)

        image_file=MetaData_image['BRE image file name']
        
        #actual calculation
        print('start interactive session')
        Orient_df=orient_session(os.path.join(BRE_dir_path,image_file))
        
        Orient_file_name=BRE_folder+'_Orient.h5'
        Orient_file=os.path.join(LMcoord_dir_path,Orient_file_name)
        Orient_df.to_hdf(Orient_file, key='data', mode='w')

        MetaData_Orient={}
        repo = git.Repo(gitPath,search_parent_directories=True)
        sha = repo.head.object.hexsha
        MetaData_Orient['git hash']=sha
        MetaData_Orient['git repo']='BRE_Trafo'
        MetaData_Orient['Orient version']=Orient_version
        MetaData_Orient['Orient file']=Orient_file_name
        MetaData_Orient['XYZ size in mum']=MetaData_image['XYZ size in mum']
        MetaData_Orient['axes']=MetaData_image['axes']
        MetaData_Orient['is control']=MetaData_image['is control']
        MetaData_Orient['time in hpf']=MetaData_image['time in hpf']
        MetaData_Orient['experimentalist']=MetaData_image['experimentalist']
        check_Orient=get_checksum(Orient_file, algorithm="SHA1")
        MetaData_Orient['output Orient checksum']=check_Orient
        writeJSON(LMcoord_dir_path,'Orient_MetaData',MetaData_Orient)


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
    test()
    #createTrainingData()
