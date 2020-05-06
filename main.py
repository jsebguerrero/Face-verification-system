import cv2
from utilities import *
import pickle
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
facenet = load_model('./model/facenet_keras.h5')
print('FaceNet loaded')
svm = pickle.load(open('./model/svm_linear.sav', 'rb'))
out_encoder = pickle.load(open('./model/label_encoder.sav','rb'))
print('SVM loaded')
in_encoder = Normalizer(norm='l2')
print('Normalizer loaded')

def main():
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        try:
            img, x1, y1, width, height = extract_face_image(frame)
            embedding = get_embedding(facenet, img)
            embedding = in_encoder.transform([embedding])
            yhat_class = svm.predict(embedding)
            yhat_prob = svm.predict_proba(embedding)
            class_index = yhat_class[0]
            class_probability = yhat_prob[0, class_index] * 100
            name = out_encoder.inverse_transform(yhat_class)
            frame = cv2.putText(frame,str(name) + ' ' + str(int(class_probability)) + ' %',(x1 + width, y1 + height), font, 0.82, (255, 110, 100),2,cv2.LINE_AA)
            cv2.imshow('FaceCAM', frame)


        except IndexError:
            print('face out of range')

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
