# develop a classifier for the 5 Celebrity Faces Dataset
from random import choice
from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pickle

# --------------------------------------------------------------------------
# --  Train the SVM for the Faces Dataset ----------------------------------
# --------------------------------------------------------------------------

# load faces
data = load('dataset.npz',allow_pickle=True)
testX_faces = data['arr_2']
# load face embeddings
data = load('dataset-embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
# normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)
# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
n_classes = len(out_encoder.transform(out_encoder.classes_))
classes = out_encoder.transform(out_encoder.classes_)
# binarize labels for AUC
trainy = label_binarize(out_encoder.transform(trainy), classes=classes)
testy = label_binarize(out_encoder.transform(testy), classes=classes)
#trainy = out_encoder.transform(trainy)
#testy = out_encoder.transform(testy)
# fit model
model = OneVsRestClassifier(SVC(kernel='linear', probability=True))
#model.fit(trainX, trainy)
y_score = model.fit(trainX, trainy).decision_function(testX)
y_pred = model.predict(testX)

# --------------------------------------------------------------------------
# --  Compute ROC curve and ROC area for each class ------------------------
# --------------------------------------------------------------------------

# Compute a confusion matrix for each class
confussion_matrix = multilabel_confusion_matrix(testy, y_pred)
j = 0
# Print the number of samples that belong to each class
for cmatrix in confussion_matrix:
    cmatrix = cmatrix.ravel()
    print('class : ',j)
    print('verdaderos negativos', cmatrix[0])
    print('falsos positivos', cmatrix[1])
    print('falsos negativos', cmatrix[2])
    print('verdaderos negativos', cmatrix[3])
    j += 1
# Dictionary object 
fpr = dict()
tpr = dict()
roc_auc = dict()

# Obtain the roc curve and its AUC
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(testy[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot of a ROC curve for a specific class
for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])    
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

"""
filename = './model/svm_linear.sav'
pickle.dump(model, open(filename, 'wb'))
filename = './model/label_encoder.sav'
pickle.dump(out_encoder, open(filename, 'wb'))
print('done !')
"""

"""# test model on a random example from the test dataset
selection = choice([i for i in range(testX.shape[0])])
random_face_pixels = testX_faces[selection]
random_face_emb = testX[selection]
random_face_class = testy[selection]
random_face_name = out_encoder.inverse_transform([random_face_class])
# prediction for the face
samples = expand_dims(random_face_emb, axis=0)
yhat_class = model.predict(samples)
yhat_prob = model.predict_proba(samples)
# get name
class_index = yhat_class[0]
class_probability = yhat_prob[0,class_index] * 100
predict_names = out_encoder.inverse_transform(yhat_class)
print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
print('Expected: %s' % random_face_name[0])
# plot for fun
pyplot.imshow(random_face_pixels)
title = '%s (%.3f)' % (predict_names[0], class_probability)
pyplot.title(title)
pyplot.show()"""