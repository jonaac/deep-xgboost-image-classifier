from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
import numpy as np
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

class cifar10vgg:
	def __init__(self,train=True):
		self.num_classes = 10
		self.weight_decay = 0.0005
		self.x_shape = [32,32,3]

		self.model = self.build_model()
		self.model = self.train(self.model)

	def build_model(self):

		model = Sequential()	
		weight_decay = self.weight_decay

		model.add(Conv2D(64, (3, 3), padding='same',
						 input_shape=self.x_shape,kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(Activation('relu'))
		model.add(BatchNormalization())
		model.add(Dropout(0.3))

		model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(Activation('relu'))
		model.add(BatchNormalization())

		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(Activation('relu'))
		model.add(BatchNormalization())
		model.add(Dropout(0.4))

		model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(Activation('relu'))
		model.add(BatchNormalization())

		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(Activation('relu'))
		model.add(BatchNormalization())
		model.add(Dropout(0.4))

		model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(Activation('relu'))
		model.add(BatchNormalization())
		model.add(Dropout(0.4))

		model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(Activation('relu'))
		model.add(BatchNormalization())

		model.add(MaxPooling2D(pool_size=(2, 2)))


		model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(Activation('relu'))
		model.add(BatchNormalization())
		model.add(Dropout(0.4))

		model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(Activation('relu'))
		model.add(BatchNormalization())
		model.add(Dropout(0.4))

		model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(Activation('relu'))
		model.add(BatchNormalization())

		model.add(MaxPooling2D(pool_size=(2, 2)))


		model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(Activation('relu'))
		model.add(BatchNormalization())
		model.add(Dropout(0.4))

		model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(Activation('relu'))
		model.add(BatchNormalization())
		model.add(Dropout(0.4))

		model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(Activation('relu'))
		model.add(BatchNormalization())

		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.5))

		model.add(Flatten())
		model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(Activation('relu'))
		model.add(BatchNormalization())

		model.add(Dropout(0.5))
		model.add(Dense(self.num_classes))
		model.add(Activation('softmax'))

		model_json = model.to_json()
		with open('model_vgg.json', 'w') as json_file:
			json_file.write(model_json)

		return model


	def normalize(self,X_train,X_test):
		mean = np.mean(X_train,axis=(0,1,2,3))
		std = np.std(X_train, axis=(0, 1, 2, 3))
		X_train = (X_train-mean)/(std+1e-7)
		X_test = (X_test-mean)/(std+1e-7)
		return X_train, X_test

	def normalize_production(self,x):
		mean = 120.707
		std = 64.15
		return (x-mean)/(std+1e-7)

	def predict(self,x,normalize=True,batch_size=50):
		if normalize:
			x = self.normalize_production(x)
		return self.model.predict(x,batch_size)

	def train(self,model):

		batch_size = 128
		maxepoches = 250
		learning_rate = 0.1
		lr_decay = 1e-6
		lr_drop = 20

		(x_train, y_train), (x_test, y_test) = cifar10.load_data()
		x_train = x_train.astype('float32')
		x_test = x_test.astype('float32')
		x_train, x_test = self.normalize(x_train, x_test)

		print('CIFAR-10 dataset loaded')

		y_train = keras.utils.to_categorical(y_train, self.num_classes)
		y_test = keras.utils.to_categorical(y_test, self.num_classes)

		def lr_scheduler(epoch):
			return learning_rate * (0.5 ** (epoch // lr_drop))
		reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

		datagen = ImageDataGenerator(
			featurewise_center=False,
			samplewise_center=False,
			featurewise_std_normalization=False,
			samplewise_std_normalization=False,
			zca_whitening=False,
			rotation_range=15, 
			width_shift_range=0.1,
			height_shift_range=0.1,
			horizontal_flip=True,
			vertical_flip=False)

		datagen.fit(x_train)

		sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
		model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])

		print('Build and save VGG16 model')

		historytemp = model.fit_generator(datagen.flow(x_train, y_train,
										 batch_size=batch_size),
							steps_per_epoch=x_train.shape[0] // batch_size,
							epochs=maxepoches,
							validation_data=(x_test, y_test),callbacks=[reduce_lr],verbose=2)
		model.save_weights('cifar10vgg.h5')
		return model

def main():

	model = cifar10vgg()


if __name__ == '__main__':
	main()




