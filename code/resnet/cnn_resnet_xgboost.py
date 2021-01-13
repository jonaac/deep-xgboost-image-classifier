import numpy as np
import pickle
import keras
from scipy import stats
import xgboost as xgb
from datetime import datetime
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers
from keras.datasets import cifar10
from keras import regularizers
from keras.callbacks import LearningRateScheduler

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import tensorflow.keras as keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model, model_from_json

import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

def load_cnn_model(X_test, y_test):
	
	json_file = open('model_resnet.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)
	# load weights into new model
	model.load_weights('model_weights.h5')

	model.compile(
		optimizer=optimizers.RMSprop(lr=2e-5),
		loss='binary_crossentropy',
		metrics=['acc'])
	'''
	y_test_ = np_utils.to_categorical(y_test, 10)
	model.evaluate(X_test, y_test_)
	'''

	return model

def get_feature_layer(model, data):
	
	total_layers = len(model.layers)

	fl_index = total_layers-7

	feature_layer_model = keras.Model(
		inputs=model.input,
		outputs=model.get_layer(index=fl_index).output)
	
	feature_layer_output = feature_layer_model.predict(data)
	
	return feature_layer_output

def xgb_model(X_train, y_train, X_test, y_test):

	dtrain = xgb.DMatrix(
		X_train,
		label=y_train
	)

	dtest = xgb.DMatrix(
		y_train,
		label=y_test
	)

	params = {
		'max_depth':12,
		'eta':0.05,
		'objective':'multi:softprob',
		'num_class':10,
		'early_stopping_rounds':5,
		'eval_metric':'merror'
	}

	watchlist = [(dtrain, 'train'),(dtest, 'eval')]
	n_round = 175

	model = xgb.train(
		params,
		dtrain,
		n_round,
		watchlist)

	pickle.dump(model, open(path + "cnn_xgboost_resnet_final.pickle.dat", "wb"))

	return model

def main():

	(X_train, y_train), (X_test, y_test) = cifar10.load_data()
	
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')

	X_train = X_train / 255.0
	X_test = X_test / 255.0

	cnn_model = load_cnn_model(X_test, y_test)
	print("Loaded CNN model from disk")

	X_train_cnn =  get_feature_layer(cnn_model,X_train)
	print("Features extracted of training data")
	X_test_cnn = get_feature_layer(cnn_model,X_test)
	print("Features extracted of test data\n")

	print("Build and save of XGBoost Model")
	model = xgb_model(X_train_cnn, y_train, X_test_cnn, y_test)

	
if __name__ == '__main__':
	main()