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
    print(p0.shape)
    pstart=p0.copy().astype(np.float32)
    #flow[:,~mask]=0
    print('start follow flow')
    p=dynamics.steps3D(pstart,flow,niter=50,rate=0.2)
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

        if not make_path(prop_dir_path):
            continue
        
        #writeJSONDict(prop_dir_path,PastMetaData) #overwrites everything

        MetaData_apply=PastMetaData["apply_MetaData"]
        flow_file=os.path.join(apply_dir_path,MetaData_apply['pred_flows file'])
        mask_file=os.path.join(apply_dir_path,MetaData_apply['segmentation file'])

        #actual calculation
        print('start calculate')
        p,p0=calculate_prop_points(flow_file,mask_file)
        

        
        p0_file=os.path.join(prop_dir_path,'start_pos')
        saveArr(p0,p0_file)
        p_file=os.path.join(prop_dir_path,'end_pos')
        saveArr(p,p_file)
        
        MetaData_prop={}
        repo = git.Repo(gitPath,search_parent_directories=True)
        sha = repo.head.object.hexsha
        MetaData_prop['git hash']=sha
        MetaData_prop['git repo']='newSegPipline'
        MetaData_prop['prop version']=Prop_version
        MetaData_prop['start file']='start_pos'
        MetaData_prop['end file']='end_pos'
        MetaData_prop['XYZ size in mum']=MetaData_apply['XYZ size in mum']
        MetaData_prop['axes']=MetaData_apply['axes']
        MetaData_prop['is control']=MetaData_apply['is control']
        MetaData_prop['time in hpf']=MetaData_apply['time in hpf']
        MetaData_prop['experimentalist']=MetaData_apply['experimentalist']
        MetaData_prop['input seg checksum']=MetaData_apply['output seg checksum']
        MetaData_prop['input flow checksum']=MetaData_apply['output flow checksum']
        MetaData_prop['output start checksum']=get_checksum(p0_file+'.npy', algorithm="SHA1")
        MetaData_prop['output end checksum']=get_checksum(p_file+'.npy', algorithm="SHA1")
        writeJSON(prop_dir_path,'prop_MetaData',MetaData_prop)

if __name__ == '__main__':
    propagatePoints()