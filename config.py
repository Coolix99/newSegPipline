import os

from config_machine import *

if(machine=='Home'):
    structured_data_path=(r'\\vs-grp07.zih.tu-dresden.de\max_kotz\structured_data\{}').format("")
    structured_data_path=(r'E:\02_Data\structured_data\{}').format("")
    trainData_path=(r'E:\02_Data\train_data\{}').format("")
    
if(machine=='Laptop'):
    pass


if(machine=='BA'):
    structured_data_path=(r'/media/max_kotz/structured_data/{}').format("")
    structured_data_path=(r'/home/max/Documents/02_Data/structured_data/{}').format("")
    trainData_path=(r'/media/max_kotz/train_data/{}').format("")

"""path structure"""
#local
script_dir = os.path.dirname(os.path.abspath(__file__))
gitPath=script_dir


#mounted
struct_nuclei_images_path=os.path.join(structured_data_path,'images','RAW_images_and_splitted','raw_images_nuclei')
struct_masks_path=os.path.join(structured_data_path,'images','segmentation_pipline','masks_images')


"""versions"""
Example_version=0

