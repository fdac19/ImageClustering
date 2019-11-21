import numpy as np
import pymongo
from sklearn.cluster import KMeans
import cv2
import sys
import csv

client = pymongo.MongoClient()
icdb = client["ICDB"]
emb_col = icdb["ResNet"]

emb_dict = emb_col.find({})
emb_vectors = []
img_vector = []

for entry in emb_dict:
	#print(entry)
	img_vector.append(entry['img'])
	emb_vectors.append(entry['emb'])

for i in range(len(img_vector)):
	print(img_vector[i], end=" ")
	for j in range(len(emb_vectors[i])):
		print(emb_vectors[i][j], end=" ")
	print("")

