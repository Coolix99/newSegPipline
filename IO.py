import tifffile
import numpy as np

#from sklearn.cluster import DBSCAN
import os
import json
#import git

from config import *



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

def exists_path(path):
    return os.path.exists(path)