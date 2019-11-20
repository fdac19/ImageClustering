import numpy as np
import pymongo
import cv2
import sys
import csv


if(len(sys.argv) < 2):
	print("Usage: python3 genEmb.py inputfile > outputfile")
	sys.exit()

# get image name and label from file
f = open(sys.argv[1], "r")
lines = f.readlines()

f = open("ids_num.txt", "r")
nums = f.readlines()
numMap = {}
for num in nums:
	print(num)
	tmpid = num.split(' ')[0]
	tmpnum = num.split(' ')[1].replace("\n", "")
	numMap[tmpid] = tmpnum

print(numMap)
labels = {}

for line in lines:
	print(line)
	idNum = line.split('_')[0]
	#print(idNum, end=" ")
	label = line.split(' ')[-1]
	label = label.replace('\n', '')
	#print(label)
	if label in labels:
		labels[label].append(idNum)
	else:
		labels[label] = [idNum]

for label in labels:
	print(label, end=" ")
	print(len(labels[label]))
	seen = []
	for num in labels[label]:
		if not num in seen:
			seen.append(num)
			print("\t", num, end=" ")
			print(labels[label].count(num))

