import numpy as np
import pickle
import keras
from scipy import stats
import xgboost as xgb
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
from keras import regularizers
from keras.callbacks import LearningRateScheduler
from keras.models import model_from_json
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf


(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

mean = np.mean(X_train,axis=(0,1,2,3))
std = np.std(X_train,axis=(0,1,2,3))
X_train = (X_train-mean)/(std+1e-7)
X_test = (X_test-mean)/(std+1e-7)

y_test = np_utils.to_categorical(y_test, 10)

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
 
# evaluate loaded model on test data
opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)
model.compile(
	loss='categorical_crossentropy',
	optimizer=opt_rms,
	metrics=['accuracy'])

scores = model.evaluate(X_test, y_test,
			batch_size=128, verbose=0)
print('Accuracy of the Baseline CNN model:')
print("%.2f%%" % (scores[1]*100))

total_layers = len(model.layers)

fl_index = total_layers-2

feature_layer_model = keras.Model(
	inputs=model.input,
	outputs=model.get_layer(index=fl_index).output)

X_test_cnn = feature_layer_model.predict(X_test)
X_test_cnn = xgb.DMatrix(X_test_cnn)

model_svm = pickle.load(open("../../models/cnn_SVM.pickle.dat", "rb"))

y_pred = model_svm.predict(X_test_cnn)
y_pred = np.argmax(y_pred,axis=1)
y_true = np.argmax(y_test,axis=1)

accuracy = accuracy_score(y_true, y_pred)
print('Accuracy of the CNN-SVM model:')
print("%.2f%%" % (accuracy * 100.0))

model_kNN =  pickle.load(open("../../models/cnn_kNN.pickle.dat", "rb"))

y_pred = model_kNN.predict(X_test_cnn)
y_pred = np.argmax(y_pred,axis=1)
y_true = np.argmax(y_test,axis=1)

accuracy = accuracy_score(y_true, y_pred)
print('Accuracy of the CNN-kNN model:')
print("%.2f%%" % (accuracy * 100.0))

model_xgb =  pickle.load(open("cnn_xgboost_final.pickle.dat", "rb"))

y_pred = model_xgb.predict(X_test_cnn)
y_pred = np.argmax(y_pred,axis=1)
y_true = np.argmax(y_test,axis=1)

accuracy = accuracy_score(y_true, y_pred)
print('Accuracy of the Baseline CNN-XGBoost model:')
print("%.2f%%" % (accuracy * 100.0))



