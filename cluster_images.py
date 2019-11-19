import pymongo
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
emb_col = icdb["Embedings"]

emb_dict = emb_col.find({})
emb_vectors = []
img_vector = []

for entry in emb_dict:
	print(entry)
	img_vector.append(entry['img'])
	emb_vectors.append(entry['emb'])
	
emb_vectors = np.array(emb_vectors)
emb_vectors = emb_vectors / emb_vectors.max(axis=0)
kmeans = KMeans(n_clusters = 15)
kmeans.fit(emb_vectors)
labels = kmeans.predict(emb_vectors)

for index, label in enumerate(labels):
	print(img_vector[index] + " " + str(label))
