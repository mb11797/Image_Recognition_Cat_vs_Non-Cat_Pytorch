import numpy as np
import h5py

def load_dataset():
    train_dataset = h5py.file('datasets/train_catvnoncat.h5', 'r')
    train_set_x_orig = np.array()






