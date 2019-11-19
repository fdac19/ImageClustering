import keras
import numpy as np
import pymongo
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import cv2
import sys
import csv

client = pymongo.MongoClient()
icdb = client["ICDB"]
emb_col = icdb["Embedings"]

emb_dict = emb_col.find({})
emb_vectors = []
img_vector = []

for entry in emb_dict:
	#print(entry)
	img_vector.append(entry['img'])
	emb_vectors.append(entry['emb'])

for i in range(len(img_vector)):
	print(img_vector[i], end="")
	print(emb_vectors[i])

