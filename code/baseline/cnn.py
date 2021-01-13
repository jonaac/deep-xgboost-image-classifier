import numpy as np
import pickle
import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
from keras import regularizers
from keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt

def cnn(x_train, y_train, x_test, y_test):

	def lr_schedule(epoch):
	    lrate = 0.001
	    if epoch > 75:
	        lrate = 0.0005
	    if epoch > 100:
	        lrate = 0.0003
	    return lrate
 
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	 
	#z-score
	mean = np.mean(x_train,axis=(0,1,2,3))
	std = np.std(x_train,axis=(0,1,2,3))
	x_train = (x_train-mean)/(std+1e-7)
	x_test = (x_test-mean)/(std+1e-7)
	 
	num_classes = 10
	 
	weight_decay = 1e-4
	model = Sequential()
	model.add(Conv2D(
	    32,(3,3),
	    padding='same',
	    kernel_regularizer=regularizers.l2(weight_decay),
	    input_shape=x_train.shape[1:]))
	model.add(Activation('elu'))
	model.add(BatchNormalization())
	model.add(Conv2D(
	    32, (3,3),
	    padding='same',
	    kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('elu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.2))
	 
	model.add(Conv2D(
	    64, (3,3),
	    padding='same',
	    kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('elu'))
	model.add(BatchNormalization())
	model.add(Conv2D(
	    64, (3,3),
	    padding='same', 
	    kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('elu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.3))
	 
	model.add(Conv2D(
	    128, (3,3), 
	    padding='same',
	    kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('elu'))
	model.add(BatchNormalization())
	model.add(Conv2D(
	    128, (3,3),
	    padding='same',
	    kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('elu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.4))
	 
	model.add(Flatten())
	model.add(Dense(num_classes, activation='softmax'))
	 
	model.summary()
	 
	#data augmentation
	datagen = ImageDataGenerator(
	    rotation_range=15,
	    width_shift_range=0.1,
	    height_shift_range=0.1,
	    horizontal_flip=True,
	    )
	datagen.fit(x_train)
	 
	#training
	batch_size = 64
	 
	opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)
	model.compile(
		loss='categorical_crossentropy',
		optimizer=opt_rms,
		metrics=['accuracy'])
	model.fit_generator(datagen.flow(
	                        x_train,
	                        y_train,
	                        batch_size=batch_size),
	                    steps_per_epoch=x_train.shape[0] // batch_size,
	                    epochs=125,
	                    verbose=1,
	                    validation_data=(x_test,y_test),
	                    callbacks=[LearningRateScheduler(lr_schedule)])
	#save to disk
	model_json = model.to_json()
	with open('model.json', 'w') as json_file:
	    json_file.write(model_json)
	model.save_weights('model.h5')
	 
	#testing
	scores = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
	print('\nTest result: %.3f loss: %.3f' % (scores[1]*100,scores[0]))


def one_hot_encode(x):

	encoded = np.zeros((len(x), 10))
	
	for idx, val in enumerate(x):
		encoded[idx][val] = 1
	
	return encoded

def main():

	(X_train, y_train), (X_test, y_test) = cifar10.load_data()

	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')

	mean = np.mean(X_train,axis=(0,1,2,3))
	std = np.std(X_train,axis=(0,1,2,3))
	X_train = (X_train-mean)/(std+1e-7)
	X_test = (X_test-mean)/(std+1e-7)

	y_train = one_hot_encode(y_train)
	y_test = one_hot_encode(y_test)

	print('CIFAR-10 dataset loaded')
	print('Build and save Baseline CNN model')
	cnn(X_train,y_train,X_test,y_test)
	

if __name__ == '__main__':
	main()