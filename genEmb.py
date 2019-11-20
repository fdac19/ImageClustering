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
import pymongo
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from keras.models import Model
import cv2
import sys
import csv

# connect to mongodb
client = pymongo.MongoClient()
icdb = client["ICDB"]
emb_col = icdb["Embedings2"] 

# check for command line arguments
if(len(sys.argv) < 2):
	print("Usage: python3 genEmb.py inputfile > outputfile")
	sys.exit()

# get image paths from input file
f = open(sys.argv[1], "r")
paths = f.readlines()

# define imageSize for VGG
imageSize = 224

# define the learning model using the Sequential function from Keras
learning_model = Sequential()

# add VGG layer and turn trainable off
#learning_model.add(VGG16(weights= 'imagenet' ,include_top= False))
#learning_model.layers[0].trainable = False

# use ResNet instead of VGG and imagenet 
learning_model.add(ResNet50(include_top = False, pooling='ave', weights = 'resnet50_weights.h5'))
learning_model.layers[0].trainable = False

# compile the learning model
learning_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# loop over images paths and generate feature vector
# then create document and add it to the mongodb
for img in paths:
	img = img.replace("\n","")
	img_object = cv2.imread(img)
	img_object = cv2.resize(img_object, (224, 224))
	img_object = np.array(img_object, dtype = np.float64)
	img_object = preprocess_input(np.expand_dims(img_object.copy(), axis = 0))
	resnet_feature = learning_model.predict(img_object)
	resnet_feature = np.array(resnet_feature)	
	emb = {"img": img, "emb": list(resnet_feature.flatten().astype(float))}
	emb_col.insert_one(emb);
