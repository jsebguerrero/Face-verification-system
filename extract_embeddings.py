import numpy as np
from keras.models import load_model
from utilities import get_embedding
from numpy import savez_compressed

# load dataset

data = np.load('./dataset.npz', allow_pickle=True)
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)

# load the facenet model

model = load_model('./model/facenet_keras.h5')
print('Loaded Model')

# convert each face in the train set to an embedding

newTrainX = list()
for face_pixels in trainX:
    embedding = get_embedding(model, face_pixels)
    newTrainX.append(embedding)
newTrainX = np.asarray(newTrainX)
print(newTrainX.shape)

# convert each face in the test set to an embedding

newTestX = list()
for face_pixels in testX:
    embedding = get_embedding(model, face_pixels)
    newTestX.append(embedding)
newTestX = np.asarray(newTestX)
print(newTestX.shape)

# save arrays to one file in compressed format

savez_compressed('dataset-embeddings.npz', newTrainX, trainy, newTestX, testy)
