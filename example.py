from src.dataset import *
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    # Load teh ring dataset
    ring_ds = RingDataset(H5PY_DIR_PATH+'/ringDataset')

    # Visualize the dataset structure and trial list
    ring_ds.show_struct()
    print("Some trial id:",ring_ds.id[:4],"...")
    print("Total number of trials:",len(ring_ds.id))

    # Get data using trial id ...
    sample = ring_ds[0]

    # .. or using infant and session ids
    sample = ring_ds[('223','20140630-1648')]

    # Let's plot accelerometer data as well as pressure value
    
    intervals = sample['intervals']

    # As there are only one imu and one pressure sensor, there is no need
    # to sensor type by doing sample[sensor][signal], sample[signal] works fine.

    # You can check the different keys by looking at sample.keys()
    acc = np.array([sample['acc_x'], sample['acc_y'], sample['acc_z']])
    acc_energy = np.sum(acc**2,axis=0)
    pressure = sample['pressure']


    normalize = lambda x: (x - np.mean(x))/np.std(x)


    plt.plot(intervals,normalize(acc_energy),label='acc energy')
    plt.plot(intervals,normalize(pressure),label='pressure')
    plt.legend()
    plt.show()
    

    # You can also accept data from a sensor, across every trial, into one array
    ring_array = ring_ds.merge()

    # This format is convenient if you want to do some machine learning
    # (which requires, of course, extra steps like filtering and windowing)

