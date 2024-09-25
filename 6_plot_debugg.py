import os
from config import nuclei_folders_path,applyresult_folder_path,propresult_folder_path,segresult_folder_path
from IO import get_JSON,getImage,load_compressed_array,loadArr
from CPC.std_op import prepareData
import numpy as np

def plot_result(nuc_img,mask,pred_flows):
    import napari

    viewer = napari.Viewer()
    viewer.add_image(nuc_img, name='3D Nuc')
    viewer.add_labels(mask, name='mask')
    #viewer.add_points(proppoints.T,name='proppoints')



    # z, y, x = np.nonzero(segmentation) 
    # origins = np.stack((z, y, x), axis=-1)
    # flow_vector_field = pred_flows.transpose(1, 2, 3, 0)
    # vectors = flow_vector_field[z, y, x]
    # vector_data = np.stack((origins, vectors), axis=1)
    # viewer.add_vectors(vector_data, name='3D Flow Field pred', edge_width=0.1, length=1, ndim=3,edge_color='blue')

    napari.run()


def load_and_plot():
    
    nuclei_folder_list = os.listdir(nuclei_folders_path)
    for nuclei_folder in nuclei_folder_list:
        print(nuclei_folder)
        nuclei_folder_path = os.path.join(nuclei_folders_path, nuclei_folder)
        MetaData=get_JSON(nuclei_folder_path)
        if not 'nuclei_image_MetaData' in MetaData:
            continue
        
        res_folder_path = os.path.join(applyresult_folder_path, nuclei_folder)
        MetaData_apply=get_JSON(res_folder_path)
        if not 'apply_MetaData' in MetaData_apply:
            continue
        

        print(MetaData)
        
        nuc_file_name = MetaData['nuclei_image_MetaData']['nuclei image file name']
        nuc_file_path = os.path.join(nuclei_folder_path, nuc_file_name)
        nuc_img = getImage(nuc_file_path)
        scale=np.array(MetaData['nuclei_image_MetaData']['XYZ size in mum']).copy()
        scale[0], scale[2] = scale[2], scale[0]
        
        print(nuc_img.shape)
        print(scale)
        
       
        nuclei,profile=prepareData(nuc_img,scale)
       
        mask=load_compressed_array(os.path.join(res_folder_path,'segmentation.h5py'))
        pred_flows=load_compressed_array(os.path.join(res_folder_path,'pred_flows.h5py'))
        print(mask.shape)
        print(pred_flows.shape)

        # res_folder_path = os.path.join(propresult_folder_path, nuclei_folder)
        # MetaData_prop=get_JSON(res_folder_path)
        # if not 'prop_MetaData' in MetaData_prop:
        #     continue
        # propPoints=loadArr(os.path.join(res_folder_path,'end_pos'))
        # total_points = propPoints.shape[1]
        # N_max = 1000
        # if N_max > total_points:
        #     raise ValueError(f"N_max ({N_max}) cannot be greater than total number of points ({total_points})")
        # random_indices = np.random.choice(total_points, size=N_max, replace=False)
        # subset_propPoints = propPoints[:, random_indices]
        # print(propPoints.shape)
        # print(subset_propPoints.shape)
        # res_folder_path = os.path.join(segresult_folder_path, nuclei_folder)
        # MetaData_seg=get_JSON(res_folder_path)
        # if not 'seg_MetaData' in MetaData_seg:
        #     continue
        # seg=getImage(os.path.join(res_folder_path, MetaData_seg['seg_MetaData']['seg file']))
        # print(seg.shape)
        
        plot_result(nuclei,mask,pred_flows)

if __name__ == '__main__':
    load_and_plot()