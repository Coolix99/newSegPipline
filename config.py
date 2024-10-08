import os

from config_machine import *

if(machine=='Home'):
    model_folder_path=None
    structured_data_path=(r'\\vs-grp07.zih.tu-dresden.de\max_kotz\structured_data\{}').format("")
    structured_data_path=(r'E:\02_Data\structured_data\{}').format("")
    trainData_path=(r'E:\02_Data\train_data\{}').format("")
    pretrainData_path=(r'E:\02_Data\pre_train_data\{}').format("")
    applyresult_folder_path=(r'\\vs-grp07.zih.tu-dresden.de\max_kotz\seg_on_cluster\applyresult\{}').format("")
    applyresult_folder_path=(r'E:\02_Data\applyresult\{}').format("")
    propresult_folder_path=(r'E:\02_Data\propresult\{}').format("")
    segresult_folder_path=(r'E:\02_Data\segmentationresult\{}').format("")
    nuclei_folders_path=(r'E:\02_Data\structured_data\images\RAW_images_and_splitted\raw_images_nuclei\{}').format("")
    batch_size=4
    n_cores=3
    
if(machine=='Laptop'):
    pass


if(machine=='BA'):
    structured_data_path=(r'/media/max_kotz/structured_data/{}').format("")

    nuclei_folders_path=(r'/media/max_kotz/structured_data/images/RAW_images_and_splitted/raw_images_nuclei')
    applyresult_folder_path=(r'/media/max_kotz/random_data/applyresult/')
    propresult_folder_path=(r'/media/max_kotz/random_data/propresult/{}').format("")
    model_folder_path=(r'/media/max_kotz/random_data/models/{}').format("")
    pretrainData_path=(r'/media/max_kotz/seg_on_cluster/pre_train_data/{}').format("")
    trainData_path=(r'/media/max_kotz/seg_on_cluster/train_data/{}').format("")
    crop_trainData_path=(r'/media/max_kotz/share_summer/annotation_nuclei/{}').format("")
    segresult_folder_path=(r'/media/max_kotz/random_data/segmentationresult')
    batch_size=4
    n_cores=6

if(machine=='Alpha'):
    structured_data_path=(r'/home/max/Documents/02_Data/structured_data/{}').format("")
    pretrainData_path=(r'/data/horse/ws/s0095413-nuclei_segmentation/data/pre_train_data/{}').format("")
    trainData_path=(r'/data/horse/ws/s0095413-nuclei_segmentation/data/train_data/{}').format("")
    model_folder_path=(r'/data/horse/ws/s0095413-nuclei_segmentation/models/{}').format("")
    

    applyresult_folder_path=(r'/data/horse/ws/s0095413-nuclei_segmentation/data/applyresult')
    propresult_folder_path=(r'/data/horse/ws/s0095413-nuclei_segmentation/data/propresult')
    segresult_folder_path=(r'/data/horse/ws/s0095413-nuclei_segmentation/data/segmentationresult')
    nuclei_folders_path=(r'/data/horse/ws/s0095413-nuclei_segmentation/data/new_nuclei_for_segmentation')

    batch_size=20
    n_cores=4

"""path structure"""
#local
script_dir = os.path.dirname(os.path.abspath(__file__))
gitPath=script_dir


#mounted
struct_nuclei_images_path=os.path.join(structured_data_path,'images','RAW_images_and_splitted','raw_images_nuclei')
struct_masks_path=os.path.join(structured_data_path,'images','newNucleiSegmentation','segmentationresult')



"""versions"""
Example_version=1
Training_version=1
Apply_version=1
Prop_version=1
Seg_version=1
