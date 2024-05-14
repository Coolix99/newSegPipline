import numpy as np
from typing import List
import napari
import os
import git
import pandas as pd
from simple_file_checksum import get_checksum

from config import *
from IO import *

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

def evalStatus_orient(image_dir_path,LMcoord_dir_path):
    """
    checks MetaData to desire wether to evaluate the file or not
    returns a dict of MetaData to be written if we should evaluate the file
    """
    AllMetaData_image=get_JSON(image_dir_path)
    if not 'BRE_image_MetaData' in AllMetaData_image:
        print('no BRE MetaData -> skip')
        return False

    if not isinstance(AllMetaData_image['BRE_image_MetaData'],dict):
        print('BRE MetaData not good')
        return False
    
    res={}
    res['BRE_image_MetaData']=AllMetaData_image['BRE_image_MetaData']

    AllMetaData_LM=get_JSON(LMcoord_dir_path)
    if not isinstance(AllMetaData_LM,dict):
        print('nothing done jet')
        return res  #do it
    try:
        if AllMetaData_LM['Orient_MetaData']['Orient version']>=Orient_version:
            print('already done')
            return False    #skip
    except:
        return res  #corrupted -> do again
    
def createTrainingData():
    BRE_folder_list=os.listdir(BRE_images_path)
    for BRE_folder in BRE_folder_list:
        print(BRE_folder)

        BRE_dir_path=os.path.join(BRE_images_path,BRE_folder)
        LMcoord_dir_path=os.path.join(LMcoord_path,BRE_folder[:-3] + "nuclei"+'_LMcoord')

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
