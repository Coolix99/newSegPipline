import git
from simple_file_checksum import get_checksum
from scipy.ndimage import label
from sklearn.cluster import DBSCAN
import napari
from numba import njit
import tifffile

from IO import *
from CPC.std_op import std_reverse_scaling

def plot_clusters(labels, p):
    """
    Plots the given points and their corresponding labels using napari.
    
    Args:
    - labels (numpy.ndarray): The labels array with cluster ids.
    - p (numpy.ndarray): The points array (3, N) where N is the number of points.
    """
    # Ensure p is in the correct shape
    assert p.shape[0] == 3, "Points array should have shape (3, N)"

    # Create a napari viewer
    viewer = napari.Viewer(ndisplay=3)
    
    # Add points layer
    viewer.add_points(p.T, size=1, face_color='red', name='points')

    # Add labels layer
    viewer.add_labels(labels, name='labels')
    
    # Start the napari event loop
    napari.run()

def plot_compare(nuc_img,seg_result):
    viewer = napari.Viewer(ndisplay=3)
    
    # Add points layer
    viewer.add_image(nuc_img,name='nuclei')

    # Add labels layer
    viewer.add_labels(seg_result, name='labels')
    
    # Start the napari event loop
    napari.run()

@njit("(int64[:], int32, int32[:,:,:],int64[:,:])", nogil=True)
def update_labels(cluster_labels, overall_cluster_id, labels, filtered_p0):
    unique_cluster_labels = np.unique(cluster_labels)
    for cid in unique_cluster_labels:
        if cid >= 0:  # Ignore noise points
            mask = (cluster_labels == cid)
            cluster_points = filtered_p0[:, mask]
            for i in range(cluster_points.shape[1]):
                x, y, z = cluster_points[:, i]
                labels[int(x), int(y), int(z)] = overall_cluster_id
            overall_cluster_id += 1
    return overall_cluster_id

def segmentImage(p0_file, p_file):
    p0 = loadArr(p0_file)
    p = loadArr(p_file)

    big_result_shape = np.max(p0, axis=1) + 1
    start_mask = np.zeros(big_result_shape, dtype=bool)
    start_mask[p0[0, :], p0[1, :], p0[2, :]] = 1

    labels, num_features = label(start_mask)
    print(f"Label shape: {labels.shape}")

    # Create a mapping of p0 indices to their label ids
    p0_indices = np.ravel_multi_index(p0, big_result_shape)
    label_map = labels.ravel()[p0_indices]
    
    # Initialize a counter for unique cluster labels
    overall_cluster_id = 1

    # Process each label
    unique_labels = np.unique(label_map[label_map > 0])
    for label_id in unique_labels:
        print(f"Processing label: {label_id}")

        # Create a boolean mask for points in p0 that have the current label_id
        mask = (label_map == label_id)
        filtered_p0 = p0[:, mask]
        selected_p = p[:, mask]

        # Apply DBSCAN to the selected points
        db = DBSCAN(eps=3, min_samples=27).fit(selected_p.T)
        cluster_labels = db.labels_
        
        # Update labels array with refined clusters using the numba-optimized function
        overall_cluster_id = update_labels(cluster_labels, overall_cluster_id, labels, filtered_p0)


    #plot_clusters(labels,p)

    return labels

def evalStatus(prop_dir_path,seg_dir_path):
    MetaData_prop=get_JSON(prop_dir_path)
    print(MetaData_prop)
    if not 'prop_MetaData' in MetaData_prop:
        print('no prop_MetaData')
        return False

    MetaData_seg=get_JSON(seg_dir_path)

    if not 'seg_MetaData' in MetaData_seg:
        print('no seg_MetaData -> do it')
        return MetaData_prop

    if not MetaData_seg['seg_MetaData']['seg version']==Seg_version:
        print('not current version')
        return MetaData_prop  

    if not MetaData_seg['seg_MetaData']['input start checksum']==MetaData_prop['prop_MetaData']['output start checksum']:
        print('differnt start')
        return MetaData_prop
    
    if not MetaData_seg['seg_MetaData']['input end checksum']==MetaData_prop['prop_MetaData']['output end checksum']:
        print('differnt end')
        return MetaData_prop

    return False

def do_masks():
    prop_folder_list=os.listdir(propresult_folder_path)
    for prop_folder in prop_folder_list:
        print(prop_folder)
         
        prop_dir_path=os.path.join(propresult_folder_path,prop_folder)
        seg_dir_path=os.path.join(segresult_folder_path,prop_folder)
        nuclei_dir_path = os.path.join(nuclei_folders_path, prop_folder)

        PastMetaData=evalStatus(prop_dir_path,seg_dir_path)
        if not isinstance(PastMetaData,dict):
            continue

        if not make_path(seg_dir_path):
            continue

        MetaData_prop=PastMetaData["prop_MetaData"]
        p0_file=os.path.join(prop_dir_path,MetaData_prop['start file'])
        p_file=os.path.join(prop_dir_path,MetaData_prop['end file'])

        #actual calculation
        print('start calculate')
        seg=segmentImage(p0_file,p_file)
        print(seg.shape)
        

        MetaData_nuc=get_JSON(nuclei_dir_path)
        writeJSON(seg_dir_path,'raw_image_MetaData',MetaData_nuc['raw_image_MetaData'])
        writeJSON(seg_dir_path,'nuclei_image_MetaData',MetaData_nuc['nuclei_image_MetaData'])

        scale=np.array(MetaData_nuc['nuclei_image_MetaData']['XYZ size in mum']).copy()
        scale[0], scale[2] = scale[2], scale[0]
        print(scale)
        seg=std_reverse_scaling(seg,scale)
        print(seg.shape)
        nuc_img=getImage(os.path.join(nuclei_dir_path,MetaData_nuc['nuclei_image_MetaData']['nuclei image file name']))
        print(nuc_img.shape)

        seg_result=np.zeros_like(nuc_img,dtype=np.int32)
        seg_result[0:seg.shape[0],0:seg.shape[1],0:seg.shape[2]]=seg

        #plot_compare(nuc_img,seg_result)

        res_file=os.path.join(seg_dir_path,prop_folder+'_seg.tiff')
        tifffile.imwrite(res_file, seg_result)
       
        MetaData_seg={}
        repo = git.Repo(gitPath,search_parent_directories=True)
        sha = repo.head.object.hexsha
        MetaData_seg['git hash']=sha
        MetaData_seg['git repo']='newSegPipline'
        MetaData_seg['seg version']=Seg_version
        MetaData_seg['seg file']=prop_folder+'_seg.tiff'
        MetaData_seg['XYZ size in mum']=MetaData_nuc['nuclei_image_MetaData']['XYZ size in mum']
        MetaData_seg['axes']=MetaData_nuc['nuclei_image_MetaData']['axes']
        MetaData_seg['is control']=MetaData_nuc['nuclei_image_MetaData']['is control']
        MetaData_seg['time in hpf']=MetaData_nuc['nuclei_image_MetaData']['time in hpf']
        MetaData_seg['experimentalist']=MetaData_nuc['nuclei_image_MetaData']['experimentalist']
        MetaData_seg['input start checksum']=MetaData_prop['output start checksum']
        MetaData_seg['input end checksum']=MetaData_prop['output end checksum']
        MetaData_seg['output seg checksum']=get_checksum(res_file, algorithm="SHA1")
        writeJSON(seg_dir_path,'seg_MetaData',MetaData_seg)

if __name__ == '__main__':
    do_masks()