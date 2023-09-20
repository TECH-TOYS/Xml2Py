import lxml
from lxml import etree

import h5py
import numpy as np

import os

#import cv2
import scipy.io

from src.dataset import *



imu_keys = ['acc_x','acc_y','acc_z','mag_x','mag_y','mag_z','gyro_x','gyro_y','gyro_z']

imu_measure_keys = ["rh_az","rh_elev","rh_az_base","rh_elev_base","lh_az","lh_elev","lh_az_base","lh_elev_base",
                    "trunk_roll","trunk_pitch","trunk_yaw","trunk_alpha"]

imu_sensor_keys = ['lh','rh','trunk']

ring_keys = {'imu':imu_keys,'pressure':['raw_value','value']}

# Key 'error' correspond to error log, i.e. if the measurement was succesful (=0) or not (!=0)
pos_keys = ['cop','head','upper_body','yaw','shoulder','hand','side','palm']


def check_format(fname: str):

    """
    Checks if 'sensors.xml' exists and if so, if it is parsable
    
    Return:  available sensor
             root of the .xml file
    """

    
    if not os.path.isfile(fname):
        print("No sensors.xml file for patient/session",fname)
        return None, None

    try :
        root = etree.parse(fname).getroot()
    except:
        return None, None

    return get_session_sensors(root), root



def get_session_sensors(root : etree.Element):

    """
    Return:  Available sensors in the .xml file
    """

    initial_frame = root.getchildren()[0]
    return np.unique([block.attrib['name'] for block in initial_frame.getchildren()])





"""

Modality-specific feature extraction functions for one session
Each modality as it own hierarchy but have the following organization:

    id [group]
        timestamp [dataset]
        error [dataset]
        sensor/location#1 [group]
            component#1 of sensor#1 [dataset]
            component#2 of sensor#1 [dataset]
            ....
        sensor/location#2 [group]
            component#1 of sensor#2 [dataset]
            component#2 of sensor#2 [dataset]
            ....
        

"""

def extract_pos_features(root : etree.Element, grp : h5py.Group):


    pos_intervals = root.xpath("frame/block[@name='mat']/@timestamp")
    grp.create_dataset("intervals",data=np.asarray(list(map(float,pos_intervals))))

    pos_error = root.xpath("frame/block[@name='mat']/sensors/sensor/@errorlog")
    grp.create_dataset("error",data=np.asarray(list(map(int,pos_error))))

    position_posture = root.xpath("frame/block[@name='mat']/sensors/sensor")
    keys = list(position_posture[0].attrib.keys())

    for location in pos_keys:
        pos_grp = grp.create_group(location)
        for key in [k for k in keys if location in k]:
            pos = [f.attrib[key] for f in position_posture]
            pos_grp.create_dataset(key,data=np.asarray(list(map(float,pos))))


            


def extract_imu_feature(root : etree.Element, grp : h5py.Group):

    imu_intervals = root.xpath("frame/block[@name='body_imu']/@timestamp")
    grp.create_dataset("intervals",data=np.asarray(list(map(float,imu_intervals))))

    measures = root.xpath("frame/block[@name='body_imu']/sensors/sensor[@type='measured_angles']")
    imu = root.xpath("frame/block[@name='body_imu']/sensors/sensor[@type='imu']") 
    

    for i,sensor_name in enumerate(imu_sensor_keys):
        sensor_grp = grp.create_group(sensor_name)
        
  
        for k in imu_keys:
            d_imu = [f.attrib[k] for f in imu[i:][::4]]
            sensor_grp.create_dataset(k,data=np.asarray(list(map(float,d_imu))))


        for m in [meas for meas in imu_measure_keys if meas.split("_")[0] == sensor_name]:
            meas = [f.attrib[m] for f in measures]
            sensor_grp.create_dataset(m,data=np.asarray(list(map(float,meas))))




def extract_ring_feature(root : etree.Element, grp : h5py.Group):

    ring_baseline = float(root.xpath("frame/block[@name='ring']/sensors/sensor[@type='pressure']/@baseline")[0])
    ring_intervals = root.xpath("frame/block[@name='ring']/@timestamp")
    grp.create_dataset("intervals",data=np.asarray(list(map(float,ring_intervals))))
    grp.attrs["baseline"] = ring_baseline


    ring_imu_grp = grp.create_group('imu')
    ring_imu = root.xpath(f"frame/block[@name='ring']/sensors/sensor[@type='imu']")

    for elem in ring_keys['imu']:
        ring = [f.attrib[elem] for f in ring_imu]
        ring_imu_grp.create_dataset(elem,data=np.asarray(list(map(float,ring))))


    ring_pressure_grp = grp.create_group('pressure')
    ring_pressure = root.xpath(f"frame/block[@name='ring']/sensors/sensor[@type='pressure']")

    for elem in ring_keys['pressure']:
        ring = [f.attrib[elem] for f in ring_pressure]
        ring_pressure_grp.create_dataset(elem,data=np.asarray(list(map(float,ring))))


    ring_actuator_grp = grp.create_group("actuators")
    ring_speaker = root.xpath("frame/block[@name='ring']/actuators/actuator[@type='speaker']/@active")
    ring_light = root.xpath("frame/block[@name='ring']/actuators/actuator[@type='light']/@active")

    strbool2int = lambda x : np.asarray([0 if b =='false' else 1 for b in x ])

    ring_actuator_grp.create_dataset("speaker",data=strbool2int(ring_speaker))
    ring_actuator_grp.create_dataset("light",data=strbool2int(ring_light))


def extract_mat_feature(root : etree.Element, grp : h5py.Group, path : str):

    mat = [strmat2array(m) for m in root.xpath("frame/block/sensors/sensor[@type='mat_raw']/@raw_data")]
    mat_intervals = root.xpath("frame/block[@name='mat_daq']/@timestamp")
    grp.create_dataset("data",data=np.asarray(mat))
    grp.create_dataset("intervals",data=np.asarray(list(map(float,mat_intervals))))


    path = path.replace('sensors.xml','')


    ref_mat = scipy.io.loadmat(os.path.join(path,'outMatrix.mat'))['RepairMatrix']
    ref_mat = np.fliplr(ref_mat).flatten()
    grp.create_dataset("ref_mat",data=ref_mat)
    


# def extract_video(path,grp):

#     video = cv2.VideoCapture(os.path.join(path,'video.mp4'))
#     ts = open(os.path.join(path,'video.ts'))

#     timestamp_video = [int(t.strip()) for t in ts.readlines()]
    

#     frames = []

#     success, img = video.read()
#     while success:
#         frames.append(img)
#         success, img = video.read()

#     if len(frames) != len(timestamp_video):
#         print("Number of frames and timestamps not matching!")
#         return 


#     grp.create_dataset("intervals",data=np.array(timestamp_video))
#     grp.create_dataset("frames",data=np.array(frames))


import matplotlib.pyplot as plt
import pandas as pd



"""
Data extraction is performed by running 

    $ python3 extract_data.py

once in command line.

--> Running this script takes a little bit of time (~15 to 20 minutes)

"""


if __name__ == "__main__":




    ringDataset = h5py.File(os.path.join(H5PY_DIR_PATH,"ringDataset"),"w")
    imuDataset = h5py.File(os.path.join(H5PY_DIR_PATH,"imuDataset"),"w")
    matDataset = h5py.File(os.path.join(H5PY_DIR_PATH,"matDataset"),"w")
    posDataset = h5py.File(os.path.join(H5PY_DIR_PATH,"posDataset"),"w")

    # annotation = pd.read_csv(DATA_PATH+'/labels.csv').iloc[:,[2,4]]
    # atypical_annotation = list(annotation[annotation.iloc[:,1]=='A'].iloc[:,0])
    


    count = 0
    count_ = 0


    strmat2array = lambda mat : list(map(int,mat.split(' ')[:-1]))


    ids = [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH,d))]


    for id_ in ids:
        

        for session in os.listdir(os.path.join(DATA_PATH,id_)):

            print("Extracting data from id = ",id_," / session = ",session,end=" | ")


            path = os.path.join(DATA_PATH,id_,session,'sensors.xml')

            if os.path.isfile(path):

                id_trial = id_.replace('_','-') + "_" + session
                used_sensors, root = check_format(path)

                print("Used sensors:", used_sensors)

                if root is NotImplemented:
                    continue

                
                if used_sensors is None:
                    continue

                
                if 'mat_daq' in used_sensors and os.path.isfile(os.path.join('/'.join(path.split('/')), 'outMatrix')):

                    mat_grp = matDataset.create_group(id_trial)
                    extract_mat_feature(root,mat_grp,path)

                if 'mat' in used_sensors:

                    pos_grp = posDataset.create_group(id_trial)
                    extract_pos_features(root,pos_grp)


                if 'ring' in used_sensors:

                    ring_grp = ringDataset.create_group(id_trial)
                    extract_ring_feature(root,ring_grp)
                    

                if 'body_imu' in used_sensors:

                    imu_grp = imuDataset.create_group(id_trial)
                    extract_imu_feature(root,imu_grp)
     

    ringDataset.close()
    imuDataset.close()
    matDataset.close()
    posDataset.close()

        