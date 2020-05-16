
# --------------------------------------------------------------------------
# ------- FaceCAM - Entrega final  ----------------------------------
# ------- Procesamiento digital de imagenes --------------------------------
# ------- Por: Juan S. Guerrero    jsebastian.guerrero@udea.edu.co  --------
# -------      Johan Alexis Berrío        johan.berrio@udea.edu.co  --------
# -------      Estudiantes  ------------------------------------------------
# ------- Mayo de 2020  ---------------------------------------------------
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# --1. Initialize the system, import dependencies and models ---------------
# --------------------------------------------------------------------------

import cv2 # OpenCV library
from utilities import * #Contains some functions for the MTCNN
import pickle
from keras.models import load_model # Keras model with the neural network model
from sklearn.preprocessing import LabelEncoder # Used to convert strings in integers
from sklearn.preprocessing import Normalizer # Used to normalize the face embedding vectors
from sklearn.svm import SVC # Linear Support Vector Machine 


# --------------------------------------------------------------------------
# --2. Load pre-trained models  --------------------------------------------
# --------------------------------------------------------------------------

facenet = load_model('./model/facenet_keras.h5') # FaceNET model
print('FaceNet loaded')
svm = pickle.load(open('./model/svm_linear.sav', 'rb')) #SVM model
out_encoder = pickle.load(open('./model/label_encoder.sav','rb')) #Encoder model
print('SVM loaded')
in_encoder = Normalizer(norm='l2') # Normalizer  with l2 normalization
print('Normalizer loaded')

# --------------------------------------------------------------------------
# --3. Main function -------------------------------------------------------
# --------------------------------------------------------------------------

def main():
    # Capture video from camera
    cap = cv2.VideoCapture(0) 
    #Simple font selected- Normal size sans-serif font
    font = cv2.FONT_HERSHEY_SIMPLEX 
    
    # -------- Program execution loop -----------
    
    while cap.isOpened():
        # Frame-by-frame capture
        ret, frame = cap.read()
        # When no video is returned from de videocamera, the execution ends
        if not ret:
            break
        # When "q" is pressed, the execution ends
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        try:
            # Call function to detect and extract the face image from a frame
            # and it returns the rectangle that contains the face too.
            img, x1, y1, width, height = extract_face_image(frame)
            # Call fuctions to get the vector that represents the features extracted from the face
            embedding = get_embedding(facenet, img)
            # Normalize the face embedding vector
            embedding = in_encoder.transform([embedding])
            # Use the the SVM to predict the similarity between trained and tested faces
            yhat_class = svm.predict(embedding)
            yhat_prob = svm.predict_proba(embedding)
            # Convert model accuracy to probability
            class_index = yhat_class[0]
            class_probability = yhat_prob[0, class_index] * 100
            # Take the name of the detected person and the accuracy and shows them both on screen 
            name = out_encoder.inverse_transform(yhat_class)
            frame = cv2.putText(frame,str(name) + ' ' + str(int(class_probability)) + ' %',(x1 + width, y1 + height), font, 0.82, (255, 110, 100),2,cv2.LINE_AA)
            cv2.imshow('FaceCAM', frame)

        # Print the next message when a face is not detected
        except IndexError:
            print('face out of range')
            
    # Close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    
#--------------------------------------------------------------------------
#---------------------------  End of the program  -------------------------
#--------------------------------------------------------------------------