import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pymongo
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import cv2
import sys
import csv

# connect to mongodb
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
	
emb_vectors = np.array(emb_vectors)
#print(emb_vectors)
pca = PCA(n_components=2)
emb_vectors = pca.fit_transform(emb_vectors)
#print(emb_vectors)
kmeans = KMeans(n_clusters = 11)
kmeans.fit(emb_vectors)
labels = kmeans.predict(emb_vectors)


plt.scatter(emb_vectors[:,0], emb_vectors[:,1], c=labels, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.savefig('resnetclus.png')
plt.show()
#for i in range(len(emb_vectors)):
	#print(img_vector[i], end=" "),
	#for j in range(len(emb_vectors)):
		#print(emb_vectors[i][j], end=" ")
	#print("")

for index, label in enumerate(labels):
	print(img_vector[index] + " " + str(label))
