import os
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense,Activation,Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import Adam
import argparse
ap = argparse.ArgumentParser()
from keras import backend
from keras.models import Sequential
from tensorflow.keras.models import Sequential
from tensorflow.keras import Sequential
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend
import cv2
import numpy as np 

from keras.preprocessing.image import ImageDataGenerator
im_width,im_height=700,700
m=Sequential()

def load_data(img_dir):
	#train_images_paths = [os.path.join(img_dir,'train/',img_path) for img_path in os.listdir(os.path.join(img_dir,'train/'))]
	#test_images_paths = [os.path.join(img_dir,'test/',img_path) for img_path in os.listdir(os.path.join(img_dir,'test/'))]	
	
	train_datagen = ImageDataGenerator(
            rescale=2./255,
            horizontal_flip=True,
            vertical_flip=True
        )

	test_datagen = ImageDataGenerator(rescale=1./255)

	train_generator = train_datagen.flow_from_directory(
			os.path.join(img_dir,'train/'),
			target_size=(im_width,im_height),
			batch_size=20,
			class_mode='binary')

	validation_generator = test_datagen.flow_from_directory(
			os.path.join(img_dir,'test/'),
			target_size=(im_width,im_height),
			batch_size=20,
            class_mode='binary')
	return train_generator,validation_generator

    

def build_model():
	m.add(Conv2D(64,(3,3), input_shape=(im_width,im_height,3), activation='sigmoid'))
	m.add(MaxPooling2D(pool_size=(5,5)))
	m.add(Flatten())
	m.add(Dense(32,activation='sigmoid'))
	m.add(Dropout(0.4))
	m.add(Dense(32,activation='sigmoid'))
	m.add(Dense(1,activation='sigmoid'))
	m.compile(
        optimizer='adam',
        metrics=['accuracy'],
        loss='binary_crossentropy')
	    


def train_model(train_data,test_data):
	m.fit_generator(
		train_data,
        steps_per_epoch=40,
        epochs=5,
        validation_data=test_data,
        validation_steps=40,
        use_multiprocessing=False,
        workers=4
        
        )
def invert_mapping(d):
	inverted=dict()
    
	for key , value in d.items():
		inverted[value]=key
        
	return inverted

def predict(img_dir):
	img_paths = [os.path.join(img_dir,'predict/',img_path) for img_path in os.listdir(os.path.join(img_dir,'predict/'))]
	images=[cv2.imread(img) for img in img_paths]
	images=[cv2.resize(img,(im_height,im_width)) for img in images]  
	images=[np.reshape(img,[1,im_height,im_width,3]) for img in images]    
	return [(m.predict_classes(img)[0][0], img_paths[i]) for i, img in enumerate(images)]   
    
    
try:
	if __name__=='__main__':
		train_data,test_data=load_data('img')
		build_model()
		train_model(train_data,test_data)
	predictions=predict('img') 
	mapping=invert_mapping(train_data.class_indices)
	for val, im_name in predictions:
		print(f'We think that {im_name} is {mapping[val]}')
	print(predictions)
except KeyboardInterrupt:
	print('\nUser aborted!')
 