import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import subprocess
import csv
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import PIL as Image
import logging
logging.getLogger().disabled = True
import tensorflow as tf
tf.get_logger().setLevel('FATAL')
# tf.get_logger().setLevel('TEST')
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from struct import unpack
from keras.preprocessing import image
from tensorflow.python.util import deprecation

from ipywidgets import widgets

import pathlib    

class JPEG:
    marker_mapping = {
    0xffd8: "Start of Image",
    0xffe0: "Application Default Header",
    0xffdb: "Quantization Table",
    0xffc0: "Start of Frame",
    0xffc4: "Define Huffman Table",
    0xffda: "Start of Scan",
    0xffd9: "End of Image"
    }
    def __init__(self, image_file):
        with open(image_file, 'rb') as f:
            self.img_data = f.read()
    
    def decode(self):
        data = self.img_data
        while(True):
            marker, = unpack(">H", data[0:2])
            JPEG.marker_mapping.get(marker)
            if marker == 0xffd8:
                data = data[2:]
            elif marker == 0xffd9:
                return
            elif marker == 0xffda:
                data = data[-2:]
            else:
                lenchunk, = unpack(">H", data[2:4])
                data = data[2+lenchunk:]            
            if len(data)==0:
                break
               
def verifyImage(img_url):
    try:
        decoder = JPEG(img_url)
        decoder.decode()
    except:
        return False
    return True

class TensorModel():
    #species list
    class_names = ['bald_uakari', 'black_headed_night_monkey', 
                     'common_squirrel_monkey', 'japanese_macaque', 
                     'mantled_howler', 'nilgiri_langur', 'patas_monkey', 
                     'pigmy_marmoset', 'silver_marmoset', 
                     'white_headed_capuchin']
    
    def __init__(self, height=700, width=700, model="Models"):
        self.img_height = height
        self.img_width = width
        self.model = model
        self.model = keras.models.load_model("Models")

    def processImage(self, image_path):
        img = tf.keras.utils.load_img(
        image_path, target_size=(self.img_height, self.img_width)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, axis=0) # Create a batch

        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        predicted = TensorModel.class_names[np.argmax(score)]
        return predicted

class Stats():
    class_names = ['bald_uakari', 'black_headed_night_monkey', 
                    'common_squirrel_monkey', 'japanese_macaque', 
                    'mantled_howler', 'nilgiri_langur', 'patas_monkey', 
                    'pigmy_marmoset', 'silver_marmoset', 
                    'white_headed_capuchin']
    def __init__(self, imageLocal):
        self.fileName = imageLocal
    def updateStats(self, speciesName):
        stats = []
        head = []
        with open(self.fileName) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=",")
            #getting first row
            head = next(readCSV)
            for row in readCSV:
                if(row == []):
                    continue
                stats.append(row)
        index = Stats.class_names.index(speciesName)
        stats[index][3] = int(stats[index][3]) + 1

        #writing updated data
        if(stats != []):
            with open(self.fileName, 'w') as file:
                write = csv.writer(file)
                write.writerow(head)
                write.writerows(stats)

    def loadStats(self):
        species = []
        stats = []
        #opening csv file
        with open(self.fileName) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=",")
            #getting first row
            species = next(readCSV)
            for row in readCSV:
                if(row == []):
                    continue
                stats.append(row)
    




