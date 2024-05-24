import git
from simple_file_checksum import get_checksum

from IO import *
from CPC import dynamics



def calculate_prop_points(flow_file,mask_file):
    """INPUT"""
    flow=load_compressed_array(flow_file)
    mask=load_compressed_array(mask_file)
    
    """Follow Flow"""
    p0=np.array(np.where(mask>0))
    print(p0)
    print(p0.shape)

    print('start follow flow')
    p=dynamics.steps3D(p0.copy(),flow * mask / 5.,niter=50)
    print('end follow flow')
    print(p.shape)
   
    return p,p0

def evalStatus(prop_dir_path,apply_dir_path):
    MetaData_apply=get_JSON(apply_dir_path)
    print(MetaData_apply)
    if not 'apply_MetaData' in MetaData_apply:
        print('no MetaData_apply')
        return False

    MetaData_prop=get_JSON(prop_dir_path)

    if not 'prop_MetaData' in MetaData_prop:
        print('no MetaData_prop -> do it')
        return MetaData_apply

    if not MetaData_prop['prop_MetaData']['prop version']==Prop_version:
        print('not current version')
        return MetaData_apply  

    if not MetaData_prop['prop_MetaData']['input seg checksum']==MetaData_apply['apply_MetaData']['output seg checksum']:
        print('differnt prop')
        return MetaData_apply
    
    if not MetaData_prop['prop_MetaData']['input flow checksum']==MetaData_apply['apply_MetaData']['output flow checksum']:
        print('differnt flow')
        return MetaData_apply

    return False

def propagatePoints():
    apply_folder_list=os.listdir(applyresult_folder_path)
    
    for apply_folder in apply_folder_list:
        print(apply_folder)
         
        prop_dir_path=os.path.join(propresult_folder_path,apply_folder)
        apply_dir_path=os.path.join(applyresult_folder_path,apply_folder)

        PastMetaData=evalStatus(prop_dir_path,apply_dir_path)
        if not isinstance(PastMetaData,dict):
            continue

        # if not make_path(prop_dir_path):
        #     continue
        
        #writeJSONDict(prop_dir_path,PastMetaData) #overwrites everything

        MetaData_apply=PastMetaData["apply_MetaData"]
        flow_file=os.path.join(apply_dir_path,MetaData_apply['pred_flows file'])
        mask_file=os.path.join(apply_dir_path,MetaData_apply['segmentation file'])

        #actual calculation
        print('start calculate')
        prop_points=calculate_prop_points(flow_file,mask_file)
        
        points_file=os.path.join(prop_dir_path,apply_folder)
        saveArr(points_file)
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
    propagatePoints()