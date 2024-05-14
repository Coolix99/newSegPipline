import numpy as np
from typing import List
import napari
import os
import git
import pandas as pd
from simple_file_checksum import get_checksum
import random

from config import *
from IO import *

from CPC.CPC_config import patch_size
from CPC.std_op import prepareExample

def orient_session(im_file):
    im=getImage(im_file)

    viewer = napari.Viewer(ndisplay=3)
    im_layer = viewer.add_image(im)
    last_pos=None
    last_viewer_direction=None
    points = []
    points_data=None
    line_layer=None

    @im_layer.mouse_drag_callbacks.append
    def on_click(layer, event):
        nonlocal last_pos,last_viewer_direction
        near_point, far_point = layer.get_ray_intersections(
            event.position,
            event.view_direction,
            event.dims_displayed
        )
        last_pos=event.position
        last_viewer_direction=event.view_direction
        print(event.position,
            event.view_direction,
            event.dims_displayed)
        
    @viewer.bind_key('a')
    def first(viewer):
        nonlocal points,last_pos
        points=[last_pos]

    @viewer.bind_key('b')
    def second(viewer):
        nonlocal points,last_pos,points_data,line_layer
        points.append(last_pos)
        line = np.array([points])
        print(line)
        print(viewer.camera)
        print(viewer)
        try:
            line_layer=viewer.add_shapes(line, shape_type='line', edge_color='red', edge_width=2)
        except:
            print('catched')

        viewer.layers.select_previous()
        points_data = [
        {'coordinate_px': np.array(points[0]), 'name': 'Proximal_pt'},
        {'coordinate_px': np.array(points[1]), 'name': 'Distal_pt'},
        {'coordinate_px': last_viewer_direction, 'name': 'viewer_direction_DV'}
        ]

    @viewer.bind_key('c')
    def first2(viewer):
        nonlocal points,last_pos
        points=[last_pos]

    @viewer.bind_key('d')
    def second2(viewer):
        nonlocal points,last_pos,points_data,line_layer
        points.append(last_pos)
        line = np.array([points])
        try:
            line_layer=viewer.add_shapes(line, shape_type='line', edge_color='green', edge_width=2)
        except:
            print('catched')
        viewer.layers.select_previous()
        viewer.layers.select_previous()
        points_data =points_data+ [
        {'coordinate_px': np.array(points[0]), 'name': 'Anterior_pt'},
        {'coordinate_px': np.array(points[1]), 'name': 'Posterior_pt'},
        {'coordinate_px': last_viewer_direction, 'name': 'viewer_direction_DP'}
        ]

    @viewer.bind_key('e')
    def first3(viewer):
        nonlocal points,last_pos
        points=[last_pos]

    @viewer.bind_key('f')
    def second3(viewer):
        nonlocal points,last_pos,points_data,line_layer
        points.append(last_pos)
        line = np.array([points])
        try:
            line_layer=viewer.add_shapes(line, shape_type='line', edge_color='blue', edge_width=2)
        except:
            print('catched')
        points_data =points_data+ [
        {'coordinate_px': np.array(points[0]), 'name': 'Proximal2_pt'},
        {'coordinate_px': np.array(points[1]), 'name': 'Distal2_pt'},
        {'coordinate_px': last_viewer_direction, 'name': 'viewer_direction_AP'}
        ]
    napari.run()
    df = pd.DataFrame(points_data)
    
    print(df)

    v1=extract_coordinate(df,'Proximal_pt')-extract_coordinate(df,'Distal_pt')
    v1 = v1 / np.linalg.norm(v1)
    v2=extract_coordinate(df,'Anterior_pt')-extract_coordinate(df,'Posterior_pt')
    #v2 = v2 - np.dot(v1, v2)
    v3=extract_coordinate(df,'Proximal2_pt')-extract_coordinate(df,'Distal2_pt')
    n=np.cross(v3,v2)
    n = n / np.linalg.norm(n)

    new_row = pd.DataFrame({'coordinate_px': [n], 'name': ['fin_plane']})
    df = pd.concat([df, new_row], ignore_index=True)

    return df

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

if __name__ == "__main__":
    createTrainingData()
