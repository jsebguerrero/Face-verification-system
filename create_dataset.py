import os # Provides functions for interacting with the operating system
from utilities import load_dataset,load_dataset_person # Load the preset datasets
import numpy as np # Fundamental package
from numpy import savez_compressed # Used to save inputs in compressed arrays

# --------------------------------------------------------------------------
# --  Save the datasets in a single compressed NumPy array file  -----------
# --------------------------------------------------------------------------

base_path = './faces/'

# Take the photos contained in the path "faces" and create a file compressed dataset,
# the dataset is composed of trained and tested faces 
photos = os.listdir(base_path)
train_dir = base_path + '/train/'
val_dir = base_path + '/val/'
trainX, trainy = load_dataset(train_dir)
valX, valy = load_dataset(val_dir)
# Compress the dataset, it contains information about faces and its labels
savez_compressed('./dataset.npz', trainX, trainy,valX, valy)


