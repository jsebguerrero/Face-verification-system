import os
from utilities import load_dataset,load_dataset_person
import numpy as np
from numpy import savez_compressed

base_path = './faces/'

photos = os.listdir(base_path)
train_dir = base_path + '/train/'
val_dir = base_path + '/val/'
trainX, trainy = load_dataset(train_dir)
valX, valy = load_dataset(val_dir)
savez_compressed('./dataset.npz', trainX, trainy,valX, valy)



