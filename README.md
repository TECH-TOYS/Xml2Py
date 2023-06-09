# Documentation

## Requirements

```
h5py==3.8.0
lxml==4.9.2
numpy==1.24.3
scipy==1.10.1
```



## XML to hdf5 conversion
In order to work with dictionnary-like structure, data is extracted from the `sensors.xml` by running the `extract_data.py` file. Before to run it, you have to:

- Install the required python libraries (using the `requirements.txt` file)
- In `dataset.py`, replace `DATA_PATH` by the path to the directory which contains the data. This directory should be organized as follow:
```
DATA_PATH
    |
    |---- Infant 1
              |
              |----- Session 1
                          |
                          | ---- sensors.xml
                          | ---- outMatrix.mat
                          | ...
               
              |----- Session 2
                    ...
    |---- Infant 2
        ...                     

```
- Change, if needed, the output directory `H5PY_DIR_PATH` in `dataset.py`. A folder named `data` will be automatically created in the current directory by default.
- Finally, run the python script with `python3 extract_data.py`.



## HDF5 dataset
HDF5 is a convenient format to store large hierachical data. Data are split into three disctinct hdf5 dataset: `imuDataset`, `matDataset`, and `ringDataset`. Each of them contains modality-specific features for each infants and each sessions.

The python class `Dataset` and its child classes `RingDataset`, `ImuDataset`, and `MatDataset` are used to access the dataset. 

A full demo of the `Dataset` classes can be found in [example.py](example.py)

## Todo
- [ ] Video processing
- [ ] Filtering and Normalization functions
- [ ] Visualization 