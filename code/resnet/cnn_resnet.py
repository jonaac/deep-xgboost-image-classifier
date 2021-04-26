from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import tensorflow.keras as keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import tensorflow as tf
from keras.utils import np_utils
from keras.models import load_model
from keras.datasets import cifar10
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf


conv_base = ResNet50(weights='imagenet', include_top=False, input_shape=(200, 200, 3))

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

print('CIFAR-10 dataset loaded')

model = models.Sequential()
model.add(layers.UpSampling2D((2,2),input_shape=(32, 32, 3)))
model.add(layers.UpSampling2D((2,2)))
model.add(layers.UpSampling2D((2,2)))
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.BatchNormalization())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.BatchNormalization())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.BatchNormalization())
model.add(layers.Dense(10, activation='softmax'))

model.compile(
	optimizer=optimizers.RMSprop(lr=2e-5),
	loss='binary_crossentropy',
	metrics=['acc'])

print('Build and save ResNet-50 model')

history = model.fit(x_train, y_train, 
	epochs=5,
	batch_size=20,
	validation_data=(x_test, y_test))

model_json = model.to_json()
with open('model_resnet.json', 'w') as json_file:
	json_file.write(model_json)
model.save_weights('model_resnet.h5')
