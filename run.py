import os
import keras
from keras import backend
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
#from keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

import cv2
import imageio as io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse

ap = argparse.ArgumentParser()
ap.add_argument(
	'epochs',
	type=int,
	help='number of epochs to train the model on'
)
args = ap.parse_args()


IMAGE_RESIZE = 256
RESNET50_POOLING_AVERAGE = 'avg'
DENSE_LAYER_ACTIVATION = 'softmax'



# Common accuracy metric for all outputs, but can use different metrics for different output
LOSS_METRICS = ['accuracy']

# EARLY_STOP_PATIENCE must be < NUM_EPOCHS
NUM_EPOCHS =args.epochs
EARLY_STOP_PATIENCE = 3

# These steps value should be proper FACTOR of no.-of-images in train & valid folders respectively
# Training images processed in each step would be no.-of-train-images / STEPS_PER_EPOCH_TRAINING
STEPS_PER_EPOCH_TRAINING = 10
STEPS_PER_EPOCH_VALIDATION = 10

# These steps value should be proper FACTOR of no.-of-images in train & valid folders respectively
# NOTE that these BATCH* are for Keras ImageDataGenerator batching to fill epoch step input
BATCH_SIZE_TRAINING = 100
BATCH_SIZE_VALIDATION = 100

# Using 1 to easily manage mapping between test_generator & prediction for submission preparation
BATCH_SIZE_TESTING = 1
batch_size = 20

NUM_CLASSES = 1
IMAGE_DIR = 'img'
if NUM_CLASSES > 1 :
	OBJECTIVE_FUNCTION = 'categorical_crossentropy'
	CLASS_MODE = 'categorical'
else:
	OBJECTIVE_FUNCTION = 'binary_crossentropy'
	CLASS_MODE = 'binary'
#class_list = ["Pizza", "Burger", "Taco"]
FC_LAYERS = [8, 1024]
dropout = 0.4
cb_early_stopper = EarlyStopping(monitor = 'val_loss', patience = EARLY_STOP_PATIENCE)
cb_checkpointer = ModelCheckpoint(filepath = 'best.hdf5', monitor = 'val_loss', save_best_only = True, mode = 'auto')

fit_history = 0

#random.seed(0)

m=Sequential()



def makeDataFrame(dir):
	labels = []
	imgs = []
	classes = [name for name in os.listdir(dir)
				if os.path.isdir(os.path.join('.', dir, name))]
	print(classes)
	NUM_CLASSES = len(classes)
	for imgclass in classes:
		classDir = os.path.join('.', dir, imgclass)
		data = [os.path.join(classDir, name) for name in os.listdir(classDir)
				if os.path.isfile(os.path.join(classDir, name))]
		for path in data:
			#im = io.imread(d)
			#print(im.shape)
			#imgs.append(im)
			imgs.append(path)
			labels.append(imgclass)
	data = { 'img': imgs, 'label': labels}
	return pd.DataFrame(data)

def load_data_from_dataframe(img_dir):
	df = makeDataFrame(img_dir)
	datagen = ImageDataGenerator(
			rescale=1. / 255,
			width_shift_range=0.2,
			height_shift_range=0.2,
			zoom_range=0.2,
			horizontal_flip=True,
			validation_split=0.2)
			
	train_generator = datagen.flow_from_dataframe(
			df, x_col="img", 
			y_col="label",
			target_size=(IMAGE_RESIZE,IMAGE_RESIZE),
			batch_size=BATCH_SIZE_TRAINING,
			shuffle=True)
	validation_generator = datagen.flow_from_dataframe(
			df, x_col="img", 
			y_col="label",
			target_size=(IMAGE_RESIZE,IMAGE_RESIZE),
			batch_size=BATCH_SIZE_VALIDATION,
			shuffle=True)
	return train_generator,validation_generator
	
def load_data_from_dir(img_dir):
	#train_images_paths = [os.path.join(img_dir,'train/',img_path) for img_path in os.listdir(os.path.join(img_dir,'train/'))]
	#test_images_paths = [os.path.join(img_dir,'test/',img_path) for img_path in os.listdir(os.path.join(img_dir,'test/'))]

	train_datagen = ImageDataGenerator(
			rescale=1. / 255,
			width_shift_range=0.2,
			height_shift_range=0.2,
			zoom_range=0.2,
			horizontal_flip=True,
			validation_split=0.2)

	test_datagen = ImageDataGenerator(rescale=1. / 255,
			width_shift_range=0.2,
			height_shift_range=0.2,
			zoom_range=0.2,
			horizontal_flip=True,
			validation_split=0.2)

	train_generator = train_datagen.flow_from_directory(
			os.path.join(img_dir,'train/'),
			target_size=(IMAGE_RESIZE,IMAGE_RESIZE),
			batch_size=BATCH_SIZE_TRAINING,
			class_mode=CLASS_MODE)

	validation_generator = test_datagen.flow_from_directory(
			os.path.join(img_dir,'test/'),
			target_size=(IMAGE_RESIZE,IMAGE_RESIZE),
			batch_size=BATCH_SIZE_VALIDATION,
			class_mode=CLASS_MODE)
	return train_generator,validation_generator

def transfer_learn_model( dropout, fc_layers):
	m.add(ResNet50(include_top = False, pooling = RESNET50_POOLING_AVERAGE, weights = 'imagenet'))#, input_shape=(IMAGE_RESIZE, IMAGE_RESIZE, 3))
	for fc in fc_layers:
		m.add(Dense(NUM_CLASSES, activation = DENSE_LAYER_ACTIVATION))
	
	m.layers[0].trainable = False
	m.compile(
		optimizer=Adam(),
		metrics=LOSS_METRICS,
		loss=OBJECTIVE_FUNCTION)


def build_model():
	m.add(Conv2D(64,(3,3), input_shape=(IMAGE_RESIZE,IMAGE_RESIZE,3), activation='relu'))
	m.add(BatchNormalization())
	m.add(Conv2D(64,(3,3), input_shape=(IMAGE_RESIZE,IMAGE_RESIZE,3), activation='relu'))
	m.add(BatchNormalization())
	m.add(Conv2D(32,(3,3), input_shape=(IMAGE_RESIZE,IMAGE_RESIZE,3), activation='sigmoid'))
	m.add(BatchNormalization())
	m.add(Conv2D(32,(3,3), input_shape=(IMAGE_RESIZE,IMAGE_RESIZE,3), activation='sigmoid'))
	m.add(BatchNormalization())
	m.add(Flatten())
	m.add(Dense(64,activation='sigmoid'))
	m.add(Dropout(0.4))
	m.add(Dense(32,activation='sigmoid'))
	m.add(Dense(1,activation='sigmoid'))
	m.compile(
		optimizer=Adam(),
		metrics=['accuracy'],
		loss=OBJECTIVE_FUNCTION)



def train_model(train_data,test_data):
	#m.load_weights("model.h5")
	fit_history = m.fit_generator(
		train_data,
		steps_per_epoch=32,
		epochs=args.epochs,
		validation_data=test_data,
		validation_steps=20,
		callbacks=[cb_checkpointer, cb_early_stopper],
		use_multiprocessing=False,
		workers=4)
	
	model_json = m.to_json()
	with open(IMAGE_DIR + "-model.json", "w") as json_file:
		json_file.write(model_json)
	# serialize weights to HDF5
	
	m.save_weights(IMAGE_DIR + "-model.h5")
	
def invert_mapping(d):
	inverted=dict()

	for key , value in d.items():
		inverted[value]=key

	return inverted

def predict(img_dir):
	img_paths = [os.path.join(img_dir,'predict/',img_path) for img_path in os.listdir(os.path.join(img_dir,'predict/'))]
	images=[cv2.imread(img) for img in img_paths]
	images=[cv2.resize(img,(IMAGE_RESIZE,IMAGE_RESIZE)) for img in images]
	images=[np.reshape(img,[1,IMAGE_RESIZE,IMAGE_RESIZE,3]) for img in images]
	return [(m.predict_classes(img)[0][0], img_paths[i]) for i, img in enumerate(images)]

def predict_gen(img_dir):
	data_generator = ImageDataGenerator(rescale=1. / 255,
			width_shift_range=0.2,
			height_shift_range=0.2,
			zoom_range=0.2,
			horizontal_flip=True,
			validation_split=0.2)
			
	test_generator = data_generator.flow_from_directory(
			directory =os.path.join(img_dir,'test/'),
			target_size = (IMAGE_RESIZE, IMAGE_RESIZE),
			batch_size = BATCH_SIZE_TESTING,
			class_mode = None,
			shuffle = False,
			seed = 123
		)
	test_generator.reset()

	pred = m.predict_generator(test_generator, steps = len(test_generator), verbose = 1)

#	predicted_class_indices = np.argmax(pred, axis = 1)
#	
#	f, ax = plt.subplots(5, 5, figsize = (15, 15))
#
#	for i in range(0,25):
#		imgBGR = cv2.imread(os.path.join('.', img_dir, test_generator.filenames[i]))
#		if imgBGR == None: 
#			raise Exception("could not load image !")
#		imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
#		
#		# a if condition else b
#		predicted_class = "Dog" if predicted_class_indices[i] else "Cat"
#
#		ax[i//5, i%5].imshow(imgRGB)
#		ax[i//5, i%5].axis('off')
#		ax[i//5, i%5].set_title("Predicted:{}".format(predicted_class))    
#
#	plt.show()
def printTraining():
	print(fit_history.history.keys())

	plt.figure(1, figsize = (15,8)) 
	plt.subplot(221)  
	plt.plot(fit_history.history['acc'])  
	plt.plot(fit_history.history['val_acc'])  
	plt.title('model accuracy')  
	plt.ylabel('accuracy')  
	plt.xlabel('epoch')  
	plt.legend(['train', 'valid']) 
		
	plt.subplot(222)  
	plt.plot(fit_history.history['loss'])  
	plt.plot(fit_history.history['val_loss'])  
	plt.title('model loss')  
	plt.ylabel('loss')  
	plt.xlabel('epoch')  
	plt.legend(['train', 'valid']) 

	plt.show()

try:
	if __name__=='__main__':
		
		train_data,test_data=load_data_from_dir(IMAGE_DIR)
		#build_model()
		transfer_learn_model(dropout=dropout, fc_layers=FC_LAYERS)
		
		train_model(train_data,test_data)
	
	#predictions=predict('dog-imgs')
	predict_gen(IMAGE_DIR)
	mapping=invert_mapping(train_data.class_indices)
	for val, im_name in predictions:
		print(f'We think that {im_name} is {mapping[val]}')
	print(predictions)
	printTraining()
except KeyboardInterrupt:
	print('\nUser aborted!')

