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
learning_model.add(VGG16(weights= 'imagenet' ,include_top= False))
learning_model.layers[0].trainable = False

# compile the learning model
learning_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

emb_vectors = []
img_vector = []

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
	#print(img + ";" + str(list(resnet_feature.flatten())))
	#row.append(img)
	#row.extend(list(resnet_feature.flatten()))
	#print(row)


#model = PCA(n_components = 256) 
#results = model.fit_transform(emb_vectors)
#print(results)

#emb_vectors = np.array(emb_vectors)
#emb_vectors = emb_vectors / emb_vectors.max(axis=0)
#kmeans = KMeans(n_clusters = 15)
#kmeans.fit(emb_vectors)
#labels = kmeans.predict(emb_vectors)



'''
def cluster(donor2img2embeding, donor2day2img):
    for donor in donor2img2embeding:
        img_names = []
        vectors = []
        for img in donor2img2embeding[donor]:
            img_names.append(img.replace('JPG','icon.JPG').replace(' ',' '))
            vectors.append(donor2img2embeding[donor][img])
        vectors = np.array(vectors)
        vectors = vectors / vectors.max(axis=0)
        ## kmeans:
        kmeans = KMeans(n_clusters = num_clusters)
        kmeans.fit(vectors)
        labels = kmeans.predict(vectors)
        ######### Agglomerative ######
        #agglomerative = AgglomerativeClustering(n_clusters = num_clusters, linkage='single')
        #agglomerative.fit(list(vectors))
        #labels = agglomerative.labels_#predict(vectors)
        for index, label in enumerate(labels):
            print(img_names[index] , ":" , donor, "_",  label)
'''
