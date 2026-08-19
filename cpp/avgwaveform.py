import h5py
import numpy as np
import matplotlib.pyplot as plt

with h5py.File('run2.hdf5', 'r') as f:
    wvfms = np.array(f['data'])
    avg_wvfm = np.sum(wvfms, axis=0)/len(wvfms)
plt.plot(avg_wvfm[0:1900])
plt.show()
