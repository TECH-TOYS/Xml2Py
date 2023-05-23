import h5py

import numpy as np
import os


from typing import Union, Tuple

H5PY_DIR_PATH = "/media/isir/PHD/code/data_processing/Xml2Py/data"
DATA_PATH = "/media/isir/storage/PHD/Data_CareToys/"



class Dataset():


    def __init__(self, path : str) -> None:

        """
            Parent class for dataset-like classes
            Implements indexing, dataset structure visualization and merging across trials

        """


        self.h5dataset = h5py.File(path,'r')
        self.id = list(self.h5dataset.keys())



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
        
        for i in range(1,len(self.id)):
            for k,v in self.__getitem__(i).items():
                merged_dict[k].append(v)
                

        return merged_dict

class RingDataset(Dataset):

    def __init__(self, data : h5py.Dataset) -> None:

        """
            Dataset for ring data
        """

        super().__init__(data)
        self.name = "Ring"
    
    def __getitem__(self, idx: Union[int,Tuple[str, str]]) -> dict:
        
        idx = super().__getitem__(idx)

        data = {}

        t = self.h5dataset[idx]['intervals']
        t = t - t[0]

        data['intervals'] = t/1000 # converting to seconds

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

    def __init__(self, data : h5py.Dataset) -> None:

        """
            Dataset for imu data
        """

        super().__init__(data)
        self.name = "IMU"


    def __getitem__(self, idx: Union[int,Tuple[str, str]]) -> dict:
        
        idx = super().__getitem__(idx)

        data = {}

        t = self.h5dataset[idx]['intervals']
        t = t - t[0]

        data['intervals'] = t/1000

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

    def __init__(self, data : h5py.Dataset) -> None:

        """
            Dataset for mat data
        """
    
        super().__init__(data)
        self.name = "Mat"


    def __getitem__(self, idx: Union[int,Tuple[str, str]]) -> dict:
        
        idx = super().__getitem__(idx)

        data = {}

        t = self.h5dataset[idx]['intervals']
        t = t - t[0]

        data['intervals'] = t/1000


        # Offset removal from the raw data
        ref_mat = np.array(self.h5dataset[idx]['ref_mat'])
        mat = self.h5dataset[idx]['data']
        mat = np.maximum(mat - ref_mat[np.newaxis,:], 0).reshape((mat.shape[0],64,32))
        data['mat'] = mat

        return data
    







