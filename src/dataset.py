import h5py
import numpy as np

import os
from typing import Union, Tuple
import warnings

from data.paths import *


# For Linux and Mac OS
H5PY_DIR_PATH = "/media/isir/PHD/code/data_processing/Xml2Py/data"


# For Windows
# H5PY_DIR_PATH = "\media\isir\PHD\code\data_processing\Xml2Py\data"
# DATA_PATH = "\media\isir\storage\PHD\Data_CareToys\\"



class Dataset():


    def __init__(self, path : str) -> None:

        """
            Parent class for dataset-like classes
            Implements indexing, dataset structure visualization and merging across trials

        """


        self.h5dataset = h5py.File(path,'r')
        self.id = list(self.h5dataset.keys())
        self.annotations = {i:{} for i in self.id}


    def __len__(self) -> int :

        return len(self.id)


    def __getitem__(self, idx : Union[int,Tuple[str,str]]) -> str:

        """
            Parameters: 
            idx: int (trial position in the dataset) or tuple(str,str) (infant id + session id)

            Return:
            idx [str]: index from trial id list

        """

        if isinstance(idx,tuple):
            idx = f'{idx[0]}_{idx[1]}'
            assert idx in self.id
            

        if isinstance(idx,int):
            assert idx in range(len(self.id))
            idx = self.id[idx]


        return idx
    
    def get_intervals(self, idx : int, absolute : bool):

        if absolute:
            return np.array(self.h5dataset[idx]['intervals'])
        else:
            t = np.array(self.h5dataset[idx]['intervals'])/1000
            return t - t[0]
    

    def show_struct(self) -> None:

        """
            Prints hdf5 dataset's structure
        """

        print(self.name + " dataset structure :")


        def printname(name):
            if name.count('/') > 0:
                name = (name.count('/')*'  ' + '|') + name
            print('|' + name)

        idx = self.id[np.random.randint(0,len(self.id))]
        self.h5dataset[idx].visit(printname) 

        print("\n")

    def merge(self) -> dict:

        """
            Merges data accross trials (i.e., accross infant and session)

            Return:
            merged_dict [dict]: dictionnary of signals accross every trials

        """


        merged_dict = self.__getitem__(0)
        merged_dict = {k:[v] for k,v in merged_dict.items()}
        merged_dict['id'] = [self.id[0]]
        
        for i in range(1,len(self.id)):
            for k,v in self.__getitem__(i).items():
                merged_dict[k].append(v)
            merged_dict['id'].append(self.id[i])
                

        return merged_dict
    

    
        

            

class RingDataset(Dataset):

    def __init__(self) -> None:

        """
            Dataset for ring data
        """

        super().__init__(os.path.join(H5PY_DIR_PATH, 'ringDataset'))
        self.name = "Ring"
    
    def __getitem__(self, idx: Union[int,Tuple[str, str]], absolute_time : bool = False) -> dict:
        
        idx = super().__getitem__(idx)

        data = {}
        

        data['intervals'] = self.get_intervals(idx,absolute_time)

        data['pressure'] = np.array(self.h5dataset[idx]['pressure']['value'])
        data['raw_pressure'] = np.array(self.h5dataset[idx]['pressure']['raw_value'])
        data['baseline'] = self.h5dataset[idx].attrs['baseline']



        for s in ['acc','gyro','mag']:
            for x in ['x','y','z']:
                data[f'{s}_{x}'] = np.array(self.h5dataset[idx]['imu'][f'{s}_{x}'])

        data['speaker'] = np.array(self.h5dataset[idx]['actuators']['speaker'])
        data['light'] = np.array(self.h5dataset[idx]['actuators']['light'])


        return data

        
        
    

class ImuDataset(Dataset):

    def __init__(self) -> None:

        """
            Dataset for imu data
        """

        super().__init__(os.path.join(H5PY_DIR_PATH, 'imuDataset'))
        self.name = "IMU"


    def __getitem__(self, idx: Union[int,Tuple[str, str]], absolute_time : bool = False) -> dict:
        
        idx = super().__getitem__(idx)

        data = {}

        data['intervals'] = self.get_intervals(idx,absolute_time)

        for part in ['lh','rh','trunk']:
            
            for sensor in ['acc','gyro','mag']:
                for axe in ['x','y','z']:
                    data[f'{part}_{sensor}_{axe}'] = np.array(self.h5dataset[idx][part][f'{sensor}_{axe}'])


            # Measures are derived from raw IMU data
            # For hand IMU: azimut and elevation
            if part in ['lh','rh']:
                for measure in ['az','az_base','elev','elev_base']: 
                    data[f'{part}_{measure}'] = np.array(self.h5dataset[idx][part][f'{part}_{measure}'])
            # For trunk IMU: rotational measurements
            else:
                for measure in ['alpha','pitch','roll','yaw']: 
                    data[f'{part}_{measure}'] = np.array(self.h5dataset[idx][part][f'{part}_{measure}'])



        return data

class MatDataset(Dataset):

    def __init__(self) -> None:

        """
            Dataset for mat data
        """
    
        super().__init__(os.path.join(H5PY_DIR_PATH, 'matDataset'))
        self.name = "Mat"


    def __getitem__(self, idx: Union[int,Tuple[str, str]],  absolute_time : bool = False) -> dict:
        
        idx = super().__getitem__(idx)

        data = {}

        data['intervals'] = self.get_intervals(idx,absolute_time)


        # Offset removal from the raw data
        ref_mat = np.array(self.h5dataset[idx]['ref_mat'])
        mat = self.h5dataset[idx]['data']
        mat = np.maximum(mat - ref_mat[np.newaxis,:], 0).reshape((mat.shape[0],64,32))
        data['mat'] = mat

        return data
    

class PosDataset(Dataset):

    def __init__(self) -> None:

        """
            Dataset for position and posture data
        """
    
        super().__init__(os.path.join(H5PY_DIR_PATH, 'posDataset'))
        self.name = "Pos"


    def __getitem__(self, idx: Union[int,Tuple[str, str]],  absolute_time : bool = False) -> dict:
        
        idx = super().__getitem__(idx)

        data = {}

        data['intervals'] = self.get_intervals(idx,absolute_time)

        err = np.array(self.h5dataset[idx]['error'])
        err[err!=0] = 1
        data['error'] = err


        location = list(self.h5dataset[idx].keys())
        location.remove('intervals')
        location.remove('error')

        for loc in location:

            data[loc] = {p:None for p in self.h5dataset[idx][loc].keys()}

            for part in self.h5dataset[idx][loc].keys():
                 data[loc][part] = np.array(self.h5dataset[idx][loc][part])
          

        return data
    







