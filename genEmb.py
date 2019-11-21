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
vgg_col = icdb["VGG"] 
resnet_col = icdb["ResNet"] 

# check for command line arguments
if(len(sys.argv) < 2):
	print("Usage: python3 genEmb.py image_paths_file")
	sys.exit()

# get image paths from input file
f = open(sys.argv[1], "r")
paths = f.readlines()

# define imageSize for VGG
imageSize = 224

# define the learning model using the Sequential function from Keras
vgg_model = Sequential()
resnet_model = Sequential()

# build vgg model using imagenet weights
vgg_model.add(VGG16(weights= 'imagenet' ,include_top= False))
vgg_model.layers[0].trainable = False

# build resnet model using imagenet weights 
resnet_model.add(ResNet50(include_top = False, pooling='ave', weights = 'ResNet/resnet50_weights.h5'))
resnet_model.layers[0].trainable = False

# compile the learning model
vgg_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
resnet_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# loop over images paths and generate feature vector
# then create document and add it to the mongodb
for img in paths:
	# grab image path and 
	# create raw image object using cv2
	print(img)
	img = img.replace("\n","")
	img_object = cv2.imread(img)
	img_object = cv2.resize(img_object, (224, 224))
	img_object = np.array(img_object, dtype = np.float64)
	img_object = preprocess_input(np.expand_dims(img_object.copy(), axis = 0))
	
  # create resnet feature vectors and add to collection
	resnet_feature = resnet_model.predict(img_object)
	resnet_feature = np.array(resnet_feature)	
	resnet_emb = {"img": img, "emb": list(resnet_feature.flatten().astype(float))}
	resnet_col.insert_one(resnet_emb)

	# create vgg feature vectors and add to collection
  # first resize image for vgg
	vgg_feature = vgg_model.predict(img_object)
	vgg_feature = np.array(vgg_feature)	
	vgg_emb = {"img": img, "emb": list(vgg_feature.flatten().astype(float))}
	vgg_col.insert_one(vgg_emb)
