from sklearn.cluster import DBSCAN
import git
from typing import List
import skimage as ski
from simple_file_checksum import get_checksum

from IO import *

from cellpose import dynamics, utils
from cellpose.io import logger_setup

class Region():
    def __init__(self) -> None:
        self.range_z=None
        self.range_y=None
        self.range_x=None



def cpu_std_down_scaling(im,):
    return im[1::3]

def mask_of_region(p,cp_mask):
    """
    Creates labels for a given region 
    """

    """DBSCAN clustering to get cells"""
    p=p[:,cp_mask].swapaxes(0, 1)
    #print(p.shape)
    if(p.shape[0]==0):
        return np.zeros_like(cp_mask,dtype=int)
    p_scaled=p*np.array((4,1,1)) #should maybe removed idk

    print('start DBSCAN')
    point_labels = DBSCAN(eps=2.5, min_samples=70).fit(p_scaled).labels_ #20
    print('end DBSCAN')

    labels=np.zeros_like(cp_mask,dtype=int)
    labels[cp_mask]=point_labels+1
    
    return labels

def lim(size,range):
    if range[0]<0:
        range[0]=0
    if range[1]>size:
        range[1]=size
    return range

def makeRegions(im_shape,nz:int,ny:int,nx:int,dmax:int,extra:int=3) ->List[Region]:
    dmax=int(dmax)
    extra=int(extra)

    RegionList=[]
    if nz<1:
        nz=1
    if ny<1:
        ny=1
    if nx<1:
        nx=1
    print('divide in parts: ',nz,ny,nx)
    dz=im_shape[0]//nz
    dy=im_shape[1]//ny
    dx=im_shape[2]//nx
    for i_z in range(nz):
        range_z=[i_z*dz-dmax//2-extra,(i_z+1)*dz+dmax//2+extra]
        range_z=lim(im_shape[0],range_z)
        for i_y in range(ny):
            range_y=[i_y*dy-dmax//2-extra,(i_y+1)*dy+dmax//2+extra]
            range_y=lim(im_shape[1],range_y)
            for i_x in range(nx):
                range_x=[i_x*dx-dmax//2-extra,(i_x+1)*dx+dmax//2+extra]
                range_x=lim(im_shape[2],range_x)
                RegionList.append(Region())
                RegionList[-1].range_z=range_z     
                RegionList[-1].range_y=range_y  
                RegionList[-1].range_x=range_x             
                
    return RegionList

def calculate_masks(flow_file,prop_file):
    """INPUT"""
    flow=loadArr(flow_file).astype(np.float32)
    print(flow.shape)
    prop=loadArr(prop_file).astype(np.float32)
    print(prop.shape)
    norm=np.sqrt(flow[0,:,:,:]*flow[0,:,:,:]+flow[1,:,:,:]*flow[1,:,:,:]+flow[2,:,:,:]*flow[2,:,:,:])
    flow[0,:,:,:]=np.divide(flow[0,:,:,:],norm,out=np.zeros_like(flow[0,:,:,:]),where=norm!=0)
    flow[1,:,:,:]=np.divide(flow[1,:,:,:],norm,out=np.zeros_like(flow[1,:,:,:]),where=norm!=0)
    flow[2,:,:,:]=np.divide(flow[2,:,:,:],norm,out=np.zeros_like(flow[2,:,:,:]),where=norm!=0)
    del norm
    
    """Follow Flow"""
    cp_mask=prop>0.0
    #logger, log_file = logger_setup()
    print('start follow flow')
    p, inds = dynamics.follow_flows(flow * cp_mask / 5., niter=50, use_gpu=useGPU_followFlow) #30 35
    print('end follow flow')
    
    RegionList = makeRegions(cp_mask.shape,cp_mask.shape[0]//100,cp_mask.shape[1]//100,cp_mask.shape[2]//100,80)
    N_regions=len(RegionList)
    labels=np.zeros_like(cp_mask,dtype=int)
    for i,r in enumerate(RegionList):
        print(i)
        crop_p=p[:,r.range_z[0]:r.range_z[1],r.range_y[0]:r.range_y[1],r.range_x[0]:r.range_x[1]]
        crop_cp_mask=cp_mask[r.range_z[0]:r.range_z[1],r.range_y[0]:r.range_y[1],r.range_x[0]:r.range_x[1]]
        lab_crop=mask_of_region(crop_p,crop_cp_mask)
        res=ski.measure.regionprops_table(lab_crop,properties=('bbox','image'))
        N_cells=len(res['image'])
        for k in range(N_cells):
            if np.max((res['bbox-3'][k]-res['bbox-0'][k],res['bbox-4'][k]-res['bbox-1'][k],res['bbox-5'][k]-res['bbox-2'][k]))<80:
                #print(res['image'][k].shape)
                if res['bbox-0'][k]<1:
                    continue
                if res['bbox-1'][k]<1:
                    continue
                if res['bbox-2'][k]<1:
                    continue
                if res['bbox-3'][k]>lab_crop.shape[0]-2:
                    continue
                if res['bbox-4'][k]>lab_crop.shape[1]-2:
                    continue
                if res['bbox-5'][k]>lab_crop.shape[2]-2:
                    continue
            nonzero=np.where(res['image'][k])
            labels[nonzero[0]+res['bbox-0'][k]+r.range_z[0],nonzero[1]+res['bbox-1'][k]+r.range_y[0],nonzero[2]+res['bbox-2'][k]+r.range_x[0]]=N_regions*k+i+1

    """Finish"""
    labels=utils.fill_holes_and_remove_small_masks(labels, min_size=20)
    #showPoints(p,db.labels_)
    #showImage(im,labels,p,point_labels)
   
    labels_down=cpu_std_down_scaling(labels)
    print('down ',labels_down.shape)
    print(np.max(labels_down))
    #print(len(ski.measure.regionprops_table(labels_down,properties=('label','image'))['label']))
    #labels_down=ski.measure.label(labels_down,connectivity=2)
    #print('down ',labels_down.shape)
    #print(np.max(labels_down))
    #showImage(im_original,labels_down)
    return labels_down

def evalStatus(masks_dir_path,flowprop_dir_path):
    """
    checks MetaData to desire wether to evaluate the file or not
    returns a dict of MetaData to be written if we should evaluate the file
    """
    
    AllMetaData_flowprop=get_JSON(flowprop_dir_path)
    if not 'nuclei_image_MetaData' in AllMetaData_flowprop:
        print('no nuclei MetaData -> skip')
        return False
    if not 'filtered_MetaData' in AllMetaData_flowprop:
        print('no filtered MetaData -> skip')
        return False
    if not 'flow_prop_MetaData' in AllMetaData_flowprop:
        print('no flow_prop_MetaData -> skip')
        return False
    if not isinstance(AllMetaData_flowprop['flow_prop_MetaData'],dict):
        print('not ready with flow prop')
        return False
    
    AllMetaData_masks=get_JSON(masks_dir_path)
    if 'masks_MetaData' in AllMetaData_masks:
        #already sometime started
        if AllMetaData_masks['masks_MetaData']==masks_version:
            print('may be work in progress')
            return False    #just started by other programm
        #check wether already finished
        try:
            if AllMetaData_masks['masks_MetaData']['masks version']==masks_version:
                if AllMetaData_masks['masks_MetaData']['input flow checksum']==AllMetaData_flowprop['flow_prop_MetaData']['output flow checksum']:
                    if AllMetaData_masks['masks_MetaData']['input prop checksum']==AllMetaData_flowprop['flow_prop_MetaData']['output prop checksum']:
                        print('already calculated')
                        return False #already done with same input
        except:
            pass 

    res={}
    res['nuclei_image_MetaData']=AllMetaData_flowprop['nuclei_image_MetaData']
    res['filtered_MetaData']=AllMetaData_flowprop['filtered_MetaData']
    res['flow_prop_MetaData']=AllMetaData_flowprop['flow_prop_MetaData']
    res['masks_MetaData']=masks_version
    return res

def do_masks():
    flowprop_folder_list=os.listdir(flow_prop_path)
    #flowprop_folder_list=['20220914_mAG-zGem_H2a-mcherry_48hpf_LM_D1_analyzed_nuclei_filtered_fp']
    for flowprop_folder in flowprop_folder_list:
        print(flowprop_folder)
         
        masks_dir_path=os.path.join(masks_images_path,flowprop_folder+'_masks')
        flowprop_dir_path=os.path.join(flow_prop_path,flowprop_folder)

        PastMetaData=evalStatus(masks_dir_path,flowprop_dir_path)
        if not isinstance(PastMetaData,dict):
            continue

        if not make_path(masks_dir_path):
            continue
        writeJSONDict(masks_dir_path,PastMetaData) #overwrites everything

        MetaData_flowprop=PastMetaData["flow_prop_MetaData"]
        flow_file=os.path.join(flowprop_dir_path,MetaData_flowprop['flow file'])
        prop_file=os.path.join(flowprop_dir_path,MetaData_flowprop['prop file'])

        #actual calculation
        print('start calculate')
        masks=calculate_masks(flow_file,prop_file)
        
        masks_file=os.path.join(masks_dir_path,flowprop_folder+'_mask.tif')
        tifffile.imwrite(masks_file,masks.astype(np.uint16))
        
        MetaData_masks={}
        check_masks=get_checksum(masks_file, algorithm="SHA1")
        repo = git.Repo(gitPath,search_parent_directories=True)
        sha = repo.head.object.hexsha
        MetaData_masks['git hash']=sha
        MetaData_masks['git repo']='cp_mask_images'
        MetaData_masks['masks version']=masks_version
        MetaData_masks['masks file']=flowprop_folder+'_mask.tif'

        MetaData_masks['input flow checksum']=PastMetaData['flow_prop_MetaData']['output flow checksum']
        MetaData_masks['input prop checksum']=PastMetaData['flow_prop_MetaData']['output prop checksum']
        MetaData_masks['output masks checksum']=check_masks

        writeJSON(masks_dir_path,'masks_MetaData',MetaData_masks)

if __name__ == '__main__':
    print('start masks')
    do_masks()