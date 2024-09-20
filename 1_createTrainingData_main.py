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

def crop_image_at_random(nuclei,masks, size:np.ndarray):
    """
    Crop a patch from a 3D image at a random start coordinate.

    :param nuclei: 3D array from which to crop the patch.
    :param masks: 3D array from which to crop the patch.
    :param size: List of three integers specifying the size of the patch in each dimension.
    :return: Cropped patch as a 3D array.
    """
    size=size.astype(int)

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

def relabel_image(image):
    # Get unique labels, excluding the background (0)
    unique_labels = np.unique(image)
    unique_labels = unique_labels[unique_labels != 0]

    # Create a mapping from old labels to new labels
    new_labels = np.arange(1, len(unique_labels) + 1)
    label_mapping = dict(zip(unique_labels, new_labels))

    # Create an output image to store the relabeled data
    relabeled_image = np.zeros_like(image)

    # Apply the label mapping
    for old_label, new_label in label_mapping.items():
        relabeled_image[image == old_label] = new_label

    return relabeled_image

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

        nuclei_crop,masks_crop=crop_image_at_random(nuclei,masks, 1.25*np.array(patch_size))
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
        for i in range(30):
            nuclei_patch,masks_patch,flow_patch=crop_trainData(nuclei_crop,modified_masks,flow, np.array(patch_size),d=3)
            masks_patch=relabel_image(masks_patch).astype(np.int32)

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

def createPreTrainingData():
    nuclei_folder_list=os.listdir(struct_nuclei_images_path)
    random.shuffle(nuclei_folder_list)
    for nuclei_folder in nuclei_folder_list:
        print(nuclei_folder)

        masks_dir_path=os.path.join(struct_masks_path,nuclei_folder)
        masks_MetaData=get_JSON(masks_dir_path)
        if masks_MetaData=={}:
            continue
        
        nuclei_dir_path=os.path.join(struct_nuclei_images_path,nuclei_folder)
        nuclei_MetaData=get_JSON(nuclei_dir_path)
        
        masks_file_name=masks_MetaData['seg_MetaData']['seg file']
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

        nuclei_crop,masks_crop=crop_image_at_random(nuclei,masks, 4*np.array(patch_size))
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
            masks_patch=relabel_image(masks_patch).astype(np.int32)

            example_folder_path=create_unique_subfolder(pretrainData_path)
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

def selectROI(image,labels):
    viewer = napari.Viewer()

    # Add the image and labels layers
    viewer.add_image(image, name='3D Image')
    viewer.add_labels(labels, name='Labels')

    # Add a points layer
    points_layer = viewer.add_points(
        ndim=3,
        name='Points',
        size=5,
        face_color='red',
        edge_color='white',
    )
    points=None
    # Define a callback to capture the points when the viewer is closed
    @viewer.bind_key('q')
    def on_close(event):
        nonlocal points
        points = points_layer.data
        viewer.close()

    # Start the Napari event loop
    napari.run()
    return points

def crop_around_center(nuclei,masks,center,target_shape):
    center = np.round(center).astype(int)
    target_shape = np.round(target_shape).astype(int)
    half_shape = (target_shape / 2).astype(int)
    
    # Calculate start and end indices for cropping
    start = np.maximum(center - half_shape, 0)
    end = np.minimum(center + half_shape, np.array(nuclei.shape))
    
    # Adjust start and end if the target_shape is larger than the array dimensions
    for i in range(3):
        if end[i] - start[i] < target_shape[i]:
            if start[i] == 0:
                end[i] = min(target_shape[i], nuclei.shape[i])
            if end[i] == nuclei.shape[i]:
                start[i] = max(nuclei.shape[i] - target_shape[i], 0)
    
    # Create slices for each dimension
    slices = tuple(slice(start[dim], end[dim]) for dim in range(3))
    
    # Return the cropped array
    return nuclei[slices],masks[slices]

def createTrainingDataFromSeg():
    seg_folder_list=os.listdir(segresult_folder_path)
    random.shuffle(seg_folder_list)
    for seg_folder in seg_folder_list:
        print(seg_folder)

        seg_dir_path=os.path.join(segresult_folder_path,seg_folder)
        seg_MetaData=get_JSON(seg_dir_path)
        if seg_MetaData=={}:
            continue
        
        nuclei_dir_path = os.path.join(nuclei_folders_path, seg_folder)
        nuclei_MetaData=get_JSON(nuclei_dir_path)

        seg_file_name=seg_MetaData['seg_MetaData']['seg file']
        seg=getImage(os.path.join(seg_dir_path,seg_file_name))
        nuclei_file_name=nuclei_MetaData['nuclei_image_MetaData']['nuclei image file name']
        nuclei=getImage(os.path.join(nuclei_dir_path,nuclei_file_name))
        print(seg.shape)
        print(nuclei.shape)
        scale=np.array(nuclei_MetaData['nuclei_image_MetaData']['XYZ size in mum']).copy()
        scale[0], scale[2] = scale[2], scale[0]
        print(scale)
        
        nuclei,masks,nuclei_profile=prepareExample(nuclei,seg,scale)
        print(nuclei.shape)

        res=selectROI(nuclei,masks)
        print(res.shape)
        for i in range(res.shape[0]):
            center=res[i,:]
            target_shape=1.25*np.array(patch_size)
    

            nuclei_crop,masks_crop=crop_around_center(nuclei,masks,center,target_shape)
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
            for i in range(30):
                nuclei_patch,masks_patch,flow_patch=crop_trainData(nuclei_crop,modified_masks,flow, np.array(patch_size),d=3)
                masks_patch=relabel_image(masks_patch).astype(np.int32)

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

def createTrainingDataCrop():
    seg_folder_list=os.listdir(segresult_folder_path)
    seg_folder_list=[
    '20240229_Smoc1_Smoc2-tilling_BRE-laux_GFP_H2A-mCh_60hpf_4_nuclei',
    '20240309_Smoc1_Smoc2-tilling_BRE-laux_GFP_H2A-mCh_96hpf_2_nuclei']
    random.shuffle(seg_folder_list)
    for seg_folder in seg_folder_list:
        print(seg_folder)

        seg_dir_path=os.path.join(segresult_folder_path,seg_folder)
        seg_MetaData=get_JSON(seg_dir_path)
        if seg_MetaData=={}:
            continue
        
        nuclei_dir_path = os.path.join(nuclei_folders_path, seg_folder)
        nuclei_MetaData=get_JSON(nuclei_dir_path)

        seg_file_name=seg_MetaData['seg_MetaData']['seg file']
        seg=getImage(os.path.join(seg_dir_path,seg_file_name))
        nuclei_file_name=nuclei_MetaData['nuclei_image_MetaData']['nuclei image file name']
        nuclei=getImage(os.path.join(nuclei_dir_path,nuclei_file_name))
        print(seg.shape)
        print(nuclei.shape)
        scale=np.array(nuclei_MetaData['nuclei_image_MetaData']['XYZ size in mum']).copy()
        scale[0], scale[2] = scale[2], scale[0]
        print(scale)
        
        nuclei,masks,nuclei_profile=prepareExample(nuclei,seg,scale)
        print(nuclei.shape)

        res=selectROI(nuclei,masks)
        if res is None:
            continue
        for i in range(res.shape[0]):
            center=res[i,:]
            target_shape=1.25*np.array(patch_size)
    

            nuclei_crop,masks_crop=crop_around_center(nuclei,masks,center,target_shape)
            masks_crop=relabel_image(masks_crop).astype(np.int32)
            print(nuclei_crop.shape)
            print(np.max(masks_crop))
            print(masks_crop.shape)
            
            viewer = napari.Viewer(ndisplay=3)
            viewer.add_image(nuclei_crop,name='nuclei_crop')
            viewer.add_labels(masks_crop,name='masks_crop')
            napari.run()
            
           
            

            example_folder_path=create_unique_subfolder(crop_trainData_path)
            print(example_folder_path)

            tiff.imwrite(os.path.join(example_folder_path,'nuclei_crop.tif'), nuclei_crop, dtype=np.float32)
            tiff.imwrite(os.path.join(example_folder_path,'masks_crop.tif'), masks_crop, dtype=np.int32)
            np.save(os.path.join(example_folder_path,'profile.npy'),nuclei_profile)

            MetaData_Example={}
            repo = git.Repo(gitPath,search_parent_directories=True)
            sha = repo.head.object.hexsha
            MetaData_Example['highest mask number']=str(np.max(masks_crop))
            MetaData_Example['git hash']=sha
            MetaData_Example['git repo']='newSegPipline'
            MetaData_Example['Example version']=Example_version
            MetaData_Example['nuc file']='nuclei_crop.tif'
            MetaData_Example['masks file']='masks_crop.tif'
            MetaData_Example['profile file']='profile.npy'
            MetaData_Example['XYZ size in mum']=nuclei_MetaData['nuclei_image_MetaData']['XYZ size in mum']
            MetaData_Example['axes']=nuclei_MetaData['nuclei_image_MetaData']['axes']
            MetaData_Example['is control']=nuclei_MetaData['nuclei_image_MetaData']['is control']
            MetaData_Example['time in hpf']=nuclei_MetaData['nuclei_image_MetaData']['time in hpf']
            MetaData_Example['experimentalist']=nuclei_MetaData['nuclei_image_MetaData']['experimentalist']
            MetaData_Example['genotype']=nuclei_MetaData['nuclei_image_MetaData']['genotype']
            writeJSON(example_folder_path,'Example_MetaData',MetaData_Example)
    
def createTrainingDatafromCrop():
    folder_list=os.listdir(crop_trainData_path)
    
    for folder in folder_list:
        print(folder)

        folder_path=os.path.join(crop_trainData_path,folder)
        Crop_MetaData=get_JSON(folder_path)
        nuclei_crop=getImage(os.path.join(folder_path,'nuclei_crop.tif'))
        masks_crop=getImage(os.path.join(folder_path,'final_mask.tif'))
        nuclei_profile=np.load(os.path.join(folder_path,'profile.npy'))

        flow=calculateFlow(masks_crop)
        
        #create N examples from this
        for i in range(30):
            nuclei_patch,masks_patch,flow_patch=crop_trainData(nuclei_crop,masks_crop,flow, np.array(patch_size),d=3)
            masks_patch=relabel_image(masks_patch).astype(np.int32)

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
            MetaData_Example['XYZ size in mum']=Crop_MetaData['Example_MetaData']['XYZ size in mum']
            MetaData_Example['axes']=Crop_MetaData['Example_MetaData']['axes']
            MetaData_Example['is control']=Crop_MetaData['Example_MetaData']['is control']
            MetaData_Example['time in hpf']=Crop_MetaData['Example_MetaData']['time in hpf']
            MetaData_Example['experimentalist']=Crop_MetaData['Example_MetaData']['experimentalist']
            writeJSON(example_folder_path,'Example_MetaData',MetaData_Example)

if __name__ == "__main__":
    #createPreTrainingData()
    #createTrainingData()
    #createTrainingDataFromSeg()

    #createTrainingDataCrop() #save crop, you can work after that on it
    createTrainingDatafromCrop() #load the worked image