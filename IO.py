import tifffile
import numpy as np

#from sklearn.cluster import DBSCAN
import os
import json
#import git

from config import *

import h5py

def save_compressed_array(filename, array, mask):
    # Ensure the mask is boolean
    mask = mask.astype(bool)
    
    # Extract non-zero elements using the mask
    non_zero_elements = array[mask]
    
    # Save the mask, non-zero elements, and original shape to a file
    with h5py.File(filename, 'w') as f:
        f.create_dataset('mask', data=mask, compression='gzip')
        f.create_dataset('non_zero_elements', data=non_zero_elements, compression='gzip')
        f.attrs['shape'] = array.shape

def load_compressed_array(filename):
    # Load the mask, non-zero elements, and original shape from the file
    with h5py.File(filename, 'r') as f:
        mask = f['mask'][:]
        non_zero_elements = f['non_zero_elements'][:]
        shape = f.attrs['shape']

    # Create an empty array of the original shape
    array = np.zeros(shape, dtype=non_zero_elements.dtype)

    # Reconstruct the original array using the mask and the non-zero elements
    array[mask] = non_zero_elements

    return array

def getImage(file):
    with tifffile.TiffFile(file) as tif:
            try:
                image=tif.asarray()
            except:
                return None
            
            return image

def saveArr(arr,path):
    np.save(path, arr)

def loadArr(path):
    return np.load(path+".npy")

def get_JSON(dir,name=None):
    if name is None:
        name='MetaData.json'
    else:
        name = 'MetaData_'+name+'.json'
    json_file_path=os.path.join(dir, name)
    try:
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        print("MetaData doesn't exist", dir, name)
        data = {}  # Create an empty dictionary if the file doesn't exist
    return data

def writeJSON(directory,key,value,name=None):
    if name is None:
        name='MetaData.json'
    else:
        name = 'MetaData_'+name+'.json'
    json_file_path=os.path.join(directory, name)
    try:
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        data = {}  # Create an empty dictionary if the file doesn't exist

    # Edit the values
    data[key] = value
    with open(json_file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    # Close the file
    json_file.close()

def writeJSONlist(directory,keys,values,name=None):
    if name is None:
        name='MetaData.json'
    else:
        name = 'MetaData_'+name+'.json'
    json_file_path=os.path.join(directory, name)
    try:
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        data = {}  # Create an empty dictionary if the file doesn't exist

    # Edit the values
    for i,key in enumerate(keys):
        data[key] = values[i]
    
    with open(json_file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    # Close the file
    json_file.close()

def make_path(newpath):
    if not os.path.exists(newpath):
        os.makedirs(newpath)
        return True
    return False

def exists_path(path):
    return os.path.exists(path)