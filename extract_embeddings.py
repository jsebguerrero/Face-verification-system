import numpy as np # Fundamental package
from keras.models import load_model # Keras model with the neural network model
from utilities import get_embedding # Get the face embedding for one face
from numpy import savez_compressed #Used to save inputs in compressed arrays

# --------------------------------------------------------------------------
# -- Generate and Save the face embeddings vectors in a single compressed --
# -- NumPy array file  -----------------------------------------------------
# --------------------------------------------------------------------------

# Load dataset
data = np.load('./dataset.npz', allow_pickle=True)
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)

# Load the facenet model
model = load_model('./model/facenet_keras.h5')
print('Loaded Model')

# Convert each face in the train set to an embedding
newTrainX = list()
# Generate an append for each embedded component obtained from the trained faces
for face_pixels in trainX:
    embedding = get_embedding(model, face_pixels)
    newTrainX.append(embedding)
# When all embedding are generated, convert these values to an array    
newTrainX = np.asarray(newTrainX)
print(newTrainX.shape)

# Convert each face in the test set to an embedding
newTestX = list()
# Generate an append for each embedded component obtained from the tested faces
for face_pixels in testX:
    embedding = get_embedding(model, face_pixels)
    newTestX.append(embedding)
# When all embedding are generated, convert these values to an array 
newTestX = np.asarray(newTestX)
print(newTestX.shape)

# Save arrays to one file in compressed format
savez_compressed('dataset-embeddings.npz', newTrainX, trainy, newTestX, testy)

