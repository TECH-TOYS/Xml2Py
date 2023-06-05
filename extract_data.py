from lxml import etree, objectify

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



def check_format(fname):

    
    if not os.path.isfile(fname):
        print("No sensors.xml file for patient/session",fname)
        return None, None

    root = etree.parse(fname).getroot()

    return get_session_sensors(root), root



def get_session_sensors(root):

    initial_frame = root.getchildren()[0]
    return np.unique([block.attrib['name'] for block in initial_frame.getchildren()])




def extract_imu_feature(root,grp):

    imu_intervals = root.xpath("frame/block[@name='body_imu']/@timestamp")
    grp.create_dataset("intervals",data=np.asarray(list(map(float,imu_intervals))))

    for i,sensor_name in enumerate(imu_sensor_keys):
        sensor_grp = grp.create_group(sensor_name)
        
        for k in imu_keys:
            imu = root.xpath("frame/block[@name='body_imu']/sensors/sensor[@id="+str(i+1)+"]/@"+k)
            sensor_grp.create_dataset(k,data=np.asarray(list(map(float,imu))))

        for m in [meas for meas in imu_measure_keys if meas.split("_")[0] == sensor_name]:
            measure = root.xpath("frame/block[@name='body_imu']/sensors/sensor[@type='measured_angles']/@"+m)
            sensor_grp.create_dataset(m,data=np.asarray(list(map(float,measure))))


def extract_ring_feature(root,grp):

    ring_baseline = float(root.xpath("frame/block[@name='ring']/sensors/sensor[@type='pressure']/@baseline")[0])
    ring_intervals = root.xpath("frame/block[@name='ring']/@timestamp")
    grp.create_dataset("intervals",data=np.asarray(list(map(float,ring_intervals))))
    grp.attrs["baseline"] = ring_baseline

    for k in ring_keys.keys():
        ring_sensor_grp = grp.create_group(k)
        for elem in ring_keys[k]:
            ring = root.xpath(f"frame/block[@name='ring']/sensors/sensor[@type='{k}']/@"+elem)
            ring_sensor_grp.create_dataset(elem,data=np.asarray(list(map(float,ring))))

    ring_actuator_grp = grp.create_group("actuators")
    ring_speaker = root.xpath("frame/block[@name='ring']/actuators/actuator[@type='speaker']/@active")
    ring_light = root.xpath("frame/block[@name='ring']/actuators/actuator[@type='light']/@active")

    strbool2int = lambda x : np.asarray([0 if b =='false' else 1 for b in x ])

    ring_actuator_grp.create_dataset("speaker",data=strbool2int(ring_speaker))
    ring_actuator_grp.create_dataset("light",data=strbool2int(ring_light))


def extract_mat_feature(root,grp,path):

    mat = [strmat2array(m) for m in root.xpath("frame/block/sensors/sensor[@type='mat_raw']/@raw_data")]
    mat_intervals = root.xpath("frame/block[@name='mat_daq']/@timestamp")
    grp.create_dataset("data",data=np.asarray(mat))
    grp.create_dataset("intervals",data=np.asarray(list(map(float,mat_intervals))))

    path = os.path.join('/',*(path.split('/')[1:-1]))


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







if __name__ == "__main__":




    ringDataset = h5py.File(os.path.join(H5PY_DIR_PATH,"ringDataset"),"w")
    imuDataset = h5py.File(os.path.join(H5PY_DIR_PATH,"imuDataset"),"w")
    matDataset = h5py.File(os.path.join(H5PY_DIR_PATH,"matDataset"),"w")





    strmat2array = lambda mat : list(map(int,mat.split(' ')[:-1]))


    out = False
    

    ids = [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH,d))]

    for id_ in ids:
        
        if out:
            break

        for session in os.listdir(os.path.join(DATA_PATH,id_)):

            print("Extracting data from id = ",id_," / session = ",session)

            path = os.path.join(DATA_PATH,id_,session,'sensors.xml')


            id_trial = id_.replace('_','-') + "_" + session


            used_sensors, root = check_format(path)
                

            if root is NotImplemented:
                continue


            

            if 'mat' in used_sensors:

                mat_grp = matDataset.create_group(id_trial)
                extract_mat_feature(root,mat_grp,path)


            if 'ring' in used_sensors:

                ring_grp = ringDataset.create_group(id_trial)
                extract_ring_feature(root,ring_grp)
                

            if 'body_imu' in used_sensors:

                imu_grp = imuDataset.create_group(id_trial)
                extract_imu_feature(root,imu_grp)


            
                        
    ringDataset.close()
    imuDataset.close()
    matDataset.close()

        