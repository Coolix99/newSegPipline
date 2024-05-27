#from sklearn.cluster import DBSCAN
import git
from typing import List
import skimage as ski
from simple_file_checksum import get_checksum
from scipy.ndimage import label
from sklearn.cluster import DBSCAN
import napari

from IO import *



# class Region():
#     def __init__(self) -> None:
#         self.range_z=None
#         self.range_y=None
#         self.range_x=None



# def cpu_std_down_scaling(im,):
#     return im[1::3]

# def mask_of_region(p,cp_mask):
#     """
#     Creates labels for a given region 
#     """

#     """DBSCAN clustering to get cells"""
#     p=p[:,cp_mask].swapaxes(0, 1)
#     #print(p.shape)
#     if(p.shape[0]==0):
#         return np.zeros_like(cp_mask,dtype=int)
#     p_scaled=p*np.array((4,1,1)) #should maybe removed idk

#     print('start DBSCAN')
#     point_labels = DBSCAN(eps=2.5, min_samples=70).fit(p_scaled).labels_ #20
#     print('end DBSCAN')

#     labels=np.zeros_like(cp_mask,dtype=int)
#     labels[cp_mask]=point_labels+1
    
#     return labels

# def lim(size,range):
#     if range[0]<0:
#         range[0]=0
#     if range[1]>size:
#         range[1]=size
#     return range

# def makeRegions(im_shape,nz:int,ny:int,nx:int,dmax:int,extra:int=3) ->List[Region]:
#     dmax=int(dmax)
#     extra=int(extra)

#     RegionList=[]
#     if nz<1:
#         nz=1
#     if ny<1:
#         ny=1
#     if nx<1:
#         nx=1
#     print('divide in parts: ',nz,ny,nx)
#     dz=im_shape[0]//nz
#     dy=im_shape[1]//ny
#     dx=im_shape[2]//nx
#     for i_z in range(nz):
#         range_z=[i_z*dz-dmax//2-extra,(i_z+1)*dz+dmax//2+extra]
#         range_z=lim(im_shape[0],range_z)
#         for i_y in range(ny):
#             range_y=[i_y*dy-dmax//2-extra,(i_y+1)*dy+dmax//2+extra]
#             range_y=lim(im_shape[1],range_y)
#             for i_x in range(nx):
#                 range_x=[i_x*dx-dmax//2-extra,(i_x+1)*dx+dmax//2+extra]
#                 range_x=lim(im_shape[2],range_x)
#                 RegionList.append(Region())
#                 RegionList[-1].range_z=range_z     
#                 RegionList[-1].range_y=range_y  
#                 RegionList[-1].range_x=range_x             
                
#     return RegionList

# def calculate_masks(flow_file,prop_file):
#     """INPUT"""
#     flow=loadArr(flow_file).astype(np.float32)
#     print(flow.shape)
#     prop=loadArr(prop_file).astype(np.float32)
#     print(prop.shape)
#     norm=np.sqrt(flow[0,:,:,:]*flow[0,:,:,:]+flow[1,:,:,:]*flow[1,:,:,:]+flow[2,:,:,:]*flow[2,:,:,:])
#     flow[0,:,:,:]=np.divide(flow[0,:,:,:],norm,out=np.zeros_like(flow[0,:,:,:]),where=norm!=0)
#     flow[1,:,:,:]=np.divide(flow[1,:,:,:],norm,out=np.zeros_like(flow[1,:,:,:]),where=norm!=0)
#     flow[2,:,:,:]=np.divide(flow[2,:,:,:],norm,out=np.zeros_like(flow[2,:,:,:]),where=norm!=0)
#     del norm
    
#     """Follow Flow"""
#     cp_mask=prop>0.0
#     #logger, log_file = logger_setup()
#     print('start follow flow')
#     p, inds = dynamics.follow_flows(flow * cp_mask / 5., niter=50, use_gpu=useGPU_followFlow) #30 35
#     print('end follow flow')
    
#     RegionList = makeRegions(cp_mask.shape,cp_mask.shape[0]//100,cp_mask.shape[1]//100,cp_mask.shape[2]//100,80)
#     N_regions=len(RegionList)
#     labels=np.zeros_like(cp_mask,dtype=int)
#     for i,r in enumerate(RegionList):
#         print(i)
#         crop_p=p[:,r.range_z[0]:r.range_z[1],r.range_y[0]:r.range_y[1],r.range_x[0]:r.range_x[1]]
#         crop_cp_mask=cp_mask[r.range_z[0]:r.range_z[1],r.range_y[0]:r.range_y[1],r.range_x[0]:r.range_x[1]]
#         lab_crop=mask_of_region(crop_p,crop_cp_mask)
#         res=ski.measure.regionprops_table(lab_crop,properties=('bbox','image'))
#         N_cells=len(res['image'])
#         for k in range(N_cells):
#             if np.max((res['bbox-3'][k]-res['bbox-0'][k],res['bbox-4'][k]-res['bbox-1'][k],res['bbox-5'][k]-res['bbox-2'][k]))<80:
#                 #print(res['image'][k].shape)
#                 if res['bbox-0'][k]<1:
#                     continue
#                 if res['bbox-1'][k]<1:
#                     continue
#                 if res['bbox-2'][k]<1:
#                     continue
#                 if res['bbox-3'][k]>lab_crop.shape[0]-2:
#                     continue
#                 if res['bbox-4'][k]>lab_crop.shape[1]-2:
#                     continue
#                 if res['bbox-5'][k]>lab_crop.shape[2]-2:
#                     continue
#             nonzero=np.where(res['image'][k])
#             labels[nonzero[0]+res['bbox-0'][k]+r.range_z[0],nonzero[1]+res['bbox-1'][k]+r.range_y[0],nonzero[2]+res['bbox-2'][k]+r.range_x[0]]=N_regions*k+i+1

#     """Finish"""
#     labels=utils.fill_holes_and_remove_small_masks(labels, min_size=20)
#     #showPoints(p,db.labels_)
#     #showImage(im,labels,p,point_labels)
   
#     labels_down=cpu_std_down_scaling(labels)
#     print('down ',labels_down.shape)
#     print(np.max(labels_down))
#     #print(len(ski.measure.regionprops_table(labels_down,properties=('label','image'))['label']))
#     #labels_down=ski.measure.label(labels_down,connectivity=2)
#     #print('down ',labels_down.shape)
#     #print(np.max(labels_down))
#     #showImage(im_original,labels_down)
#     return labels_down


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

def segmentImage(p0_file, p_file):
    p0 = loadArr(p0_file)
    p = loadArr(p_file)

    # Select a subset based on (zmax, ymax, xmax)
    subset_mask = (p0[0, :] < 500) & (p0[1, :] < 500) & (p0[2, :] < 500)
    p0 = p0[:, subset_mask]
    p = p[:, subset_mask]
    print(p)
    print(p0)
    return
    print(p0.shape)

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

        # print(f"Filtered P0 shape: {filtered_p0.shape}")
        # print(f"Selected P shape: {selected_p.shape}")

        # Apply DBSCAN to the selected points
        db = DBSCAN(eps=3, min_samples=10).fit(selected_p.T)
        cluster_labels = db.labels_

        # Update labels array with refined clusters
        cluster_ids = np.unique(cluster_labels[cluster_labels >= 0])
        for cid in cluster_ids:
            cluster_mask = (cluster_labels == cid)
            cluster_points = filtered_p0[:, cluster_mask]
           
            labels[cluster_points[0,:],cluster_points[1,:],cluster_points[2,:]] = overall_cluster_id
            overall_cluster_id += 1

    plot_clusters(labels,p)

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

        PastMetaData=evalStatus(prop_dir_path,seg_dir_path)
        if not isinstance(PastMetaData,dict):
            continue

        # if not make_path(seg_dir_path):
        #     continue
        
        #writeJSONDict(prop_dir_path,PastMetaData) #overwrites everything

        MetaData_prop=PastMetaData["prop_MetaData"]
        p0_file=os.path.join(prop_dir_path,MetaData_prop['start file'])
        p_file=os.path.join(prop_dir_path,MetaData_prop['end file'])

        #actual calculation
        print('start calculate')
        seg_result=segmentImage(p0_file,p_file)
        return
        # nuclei_folder_path = os.path.join(nuclei_folders_path, nuclei_folder)
        
        # p0_file=os.path.join(prop_dir_path,'start_pos')
        # saveArr(p0,p0_file)
        # p_file=os.path.join(prop_dir_path,'end_pos')
        # saveArr(p,p_file)
        
        # MetaData_prop={}
        # repo = git.Repo(gitPath,search_parent_directories=True)
        # sha = repo.head.object.hexsha
        # MetaData_prop['git hash']=sha
        # MetaData_prop['git repo']='newSegPipline'
        # MetaData_prop['apply version']=Apply_version
        # MetaData_prop['start file']='start_pos'
        # MetaData_prop['end file']='end_pos'
        # MetaData_prop['XYZ size in mum']=MetaData_apply['XYZ size in mum']
        # MetaData_prop['axes']=MetaData_apply['axes']
        # MetaData_prop['is control']=MetaData_apply['is control']
        # MetaData_prop['time in hpf']=MetaData_apply['time in hpf']
        # MetaData_prop['experimentalist']=MetaData_apply['experimentalist']
        # MetaData_prop['input seg checksum']=MetaData_apply['output seg checksum']
        # MetaData_prop['input flow checksum']=MetaData_apply['output flow checksum']
        # MetaData_prop['output start checksum']=get_checksum(p0_file+'.npy', algorithm="SHA1")
        # MetaData_prop['output end checksum']=get_checksum(p_file+'.npy', algorithm="SHA1")
        # writeJSON(prop_dir_path,'prop_MetaData',MetaData_prop)

if __name__ == '__main__':
    do_masks()