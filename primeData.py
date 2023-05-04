import csv
import matplotlib.pyplot as plt
import numpy as np
import os 
import subprocess
from pathlib import Path
import PIL as Image
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from struct import unpack
from keras.preprocessing import image

"""This program is designed to build fresh stats from a new model.
This is done in the following order:
1. load model
2. load image data
3. test the model against the image data
4. record stats of each species predicted on average
5. store as a table in a csv file"""
#model data
model = keras.models.load_model("Models")
img_height = 400
img_width = img_height
speciesClassifier = ['bald_uakari', 'black_headed_night_monkey', 
                     'common_squirrel_monkey', 'japanese_macaque', 
                     'mantled_howler', 'nilgiri_langur', 'patas_monkey', 
                     'pigmy_marmoset', 'silver_marmoset', 
                     'white_headed_capuchin']

def predictSpecies(imgurl):
    img = tf.keras.utils.load_img(
    imgurl, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted = speciesClassifier[np.argmax(score)]
    return predicted

#species list
species = ["mantled_howler", "mantled_howler", "bald_uakari", 
               "japanese_macaque", "pygmy_marmoset", 'white_headed_capuchin', 
               'white_headed_capuchin', 'common_squirrel_monkey', 
               'black_headed_night_monkey', 'nilgiri_langur']

#dict deffinition

primates = dict([('n0', 'mantled_howler'),
('n1', 'mantled_howler'),
('n2', 'bald_uakari'),
('n3', 'japanese_macaque'),
('n4', 'pygmy_marmoset'),
('n5', 'white_headed_capuchin'),
('n6', 'white_headed_capuchin'),
('n7', 'common_squirrel_monkey'),
('n8', 'black_headed_night_monkey'),
('n9', 'nilgiri_langur')])


imageLocal = "C:\\Users\\Brenden\\Desktop\\Primates"
#getting list of directories to explore
speciesList = []
accuracyList = []
#generating stats
for x in os.walk(imageLocal):
    totalAccuracy = 0
    splitName = x[0].split(imageLocal + "\\")
    if(len(splitName) == 2):
        speciesList.append(splitName[1])
    if(len(x[2]) == 0):
        continue
    for item in x[2]:
        image_url = x[0] + '\\' + item
        if(predictSpecies(image_url) == splitName[1]):
            totalAccuracy += 1
    #first three indeces detail the accuracy of predictions reguarding the dataset (training, testing)
    #last index details the dataset + userImages
    #[correctpredictions, numberOfImagesScanned, averageAccuracy, TotalImagesScanned]
    accuracyList.append([totalAccuracy, len(x[2]), totalAccuracy / len(x[2]), len(x[2])])   

#reducing to significant figures for percentage accuracy
for index in range(0, len(accuracyList)):
    val = float(accuracyList[index][2]) * 1000
    val = ((val * 1000) // 1) / 10 #calculating significant figures
    accuracyList[index][2] = val


with open('stats.csv', 'w') as file:
    write = csv.writer(file)
    write.writerow(speciesList)
    write.writerows(accuracyList)


