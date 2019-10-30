import keras
from keras.applications import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications import VGG19
from keras.applications.resnet50 import preprocess_input
from keras.applications.vgg16 import preprocess_input
from keras.models import Sequential
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.callbacks import TensorBoard
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from keras.models import Model
import cv2
import sys
import csv

# check command line arguments
if (len(sys.argv) < 3): 
	sys.stderr.write("Usage: python3 genEmbedings.py \"inputfile\" > \"outputfile\"\n");
	sys.exit(1)

# define imageSize for VGG
imageSize = 224

# define the learning model using the Sequential function from Keras
learning_model = Sequential()

# add VGG layer and turn trainable off
learning_model.add(VGG16(weights= 'imagenet' ,include_top= False))
learning_model.layers[0].trainable = False

# compile the learning model
learning_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

missed_imgs = []
rows = []
    
with open(imgs_path) as csv_file:
    paths = csv.reader (csv_file, delimiter='\n')
    img_names = []
    for path in paths:
        row = []
        correct_path = path[0]
        correct_path.replace(' ', '\ ')
        correct_path.replace('(', '\(')
        correct_path.replace(')', '\)')
        try:
            '''
            #######gray########
            img_object = cv2.imread(correct_path, cv2.IMREAD_GRAYSCALE)
            img_object = np.stack((img_object,)*3, axis=-1)
            '''
            img_object = cv2.imread(correct_path)
            img_object = cv2.resize(img_object, (imageSize, imageSize))
            img_object = np.array(img_object, dtype = np.float64)
            img_object = preprocess_input(np.expand_dims(img_object.copy(), axis = 0))

            resnet_feature = learning_model.predict(img_object)
            resnet_feature = np.array(resnet_feature)
            '''import bpython
            bpython.embed(locals())
            exit()'''
            row.append(correct_path)
            row.extend(list(resnet_feature.flatten()))
            print(row)
        except: 
            missed_imgs.append(path)
