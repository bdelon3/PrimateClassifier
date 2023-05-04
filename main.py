from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wx
import os 
import subprocess
import csv
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import PIL as Image
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from struct import unpack
from keras.preprocessing import image

import pathlib

marker_mapping = {
    0xffd8: "Start of Image",
    0xffe0: "Application Default Header",
    0xffdb: "Quantization Table",
    0xffc0: "Start of Frame",
    0xffc4: "Define Huffman Table",
    0xffda: "Start of Scan",
    0xffd9: "End of Image"
}


class JPEG:
    def __init__(self, image_file):
        with open(image_file, 'rb') as f:
            self.img_data = f.read()
    
    def decode(self):
        data = self.img_data
        while(True):
            marker, = unpack(">H", data[0:2])
            #print(marker_mapping.get(marker))
            marker_mapping.get(marker)
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

    

#model data
model = keras.models.load_model("Models")

#species list
class_names = ['bald_uakari', 'black_headed_night_monkey', 
                     'common_squirrel_monkey', 'japanese_macaque', 
                     'mantled_howler', 'nilgiri_langur', 'patas_monkey', 
                     'pigmy_marmoset', 'silver_marmoset', 
                     'white_headed_capuchin']

img_height = 400
img_width = img_height
image_url = "C:\\Users\\Brenden\\Desktop\\picachu.JPG"

def updateStats(speciesName):
    stats = []
    head = []
    with open('stats.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=",")
        #getting first row
        head = next(readCSV)
        for row in readCSV:
            if(row == []):
                continue
            stats.append(row)
    index = class_names.index(speciesName)
    stats[index][3] = int(stats[index][3]) + 1

    #writing updated data
    if(stats != []):
        with open('stats.csv', 'w') as file:
            write = csv.writer(file)
            write.writerow(head)
            write.writerows(stats)

#image verification
def verifyImage(imgUrl):

    try:
        img_data = None
        with open(imgUrl) as file:
            img_data = file.read()

        data = img_data
        while(True):
            marker, = unpack(">H", data[0:2])
            marker_mapping.get(marker)
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
            # img = JPEG(imgUrl)
            # img.decode()
    except:
        return False

    return True
#wx
class MainWindow(wx.Frame):
    def __init__(self, parent, title):
        wx.Frame.__init__(self, parent, title=title, size=(500, 500))
        panel = wx.Panel(self)
        #buttons add image, get stastics, process image
        #image selection
        self.addImage = wx.Button(parent=panel, label="Select Image", pos=(125, 90))
        self.addImage.Bind(event=wx.EVT_BUTTON, handler=self.addImageEvent)

        #image processing
        self.processImage = wx.Button(parent=panel, label="Process Image", pos=(125, 130))
        self.processImage.Bind(event=wx.EVT_BUTTON, handler=self.processImageEvent) 
        self.processImage.Disable()

        #Load stats
        self.viewStats = wx.Button(parent=panel, label="load all stats", pos=(125, 170))
        self.viewStats.Bind(event=wx.EVT_BUTTON, handler=self.loadStatsEvent)

        #close button
        exitButton = wx.Button(parent=panel, label="Close", pos=(125, 210))
        exitButton.Bind(event=wx.EVT_BUTTON, handler=self.closeEvent)

        #labels
        self.predictionLabel = wx.StaticText(parent=panel, label="Species Identified", pos=(250,110))
        self.imageWarning = wx.StaticText(parent=panel, label="", pos=(0,300))
        self.imageWarning.SetForegroundColour((255,0,0)) # set text color

        #text fields
        self.outputField = wx.TextCtrl(parent=panel, pos=(250, 130), size=(200, 25), value="Pending...", style=wx.TE_READONLY)

        self.imgLocal = ""

        self.Show(True)

    #event handlers
    def addImageEvent(self, event):
            root = tk.Tk()
            root.withdraw()

            files = filedialog.askopenfilenames(filetypes=[("JPEG", "*.jpg")])
            self.imageWarning.Label=""
            if(len(files) == 0):
                print("Faled to load")
                return
            if(not verifyImage(files[0])):
                self.imageWarning.Label="The Image is not the correct type or is corrupted!"
                return
            self.outputField.Value = "Pending..."
            file = files[0]
            self.imgLocal = file
            self.processImage.Enable(True)

    def processImageEvent(self, event):
        img = tf.keras.utils.load_img(
            self.imgLocal, target_size=(img_height, img_width)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, axis=0) # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        predicted = class_names[np.argmax(score)]
        print("PRedicted----------" + predicted)
        print(class_names)
        updateStats(predicted)
        self.outputField.Value = predicted

    def loadStatsEvent(self, event):
        species = []
        stats = []
        #opening csv file
        with open('stats.csv') as csvfile:
            readCSV = csv.reader(csvfile, delimiter=",")
            #getting first row
            species = next(readCSV)
            for row in readCSV:
                if(row == []):
                    continue
                stats.append(row)

        #calculating stats
        imagesPerCat = []
        accuracyStats = []
        numberOfSamples = []
        for category in stats:
            accuracyStats.append(float(category[2]))
            imagesPerCat.append(int(category[3]))
            numberOfSamples.append(int(category[1]))

        # #calculating percentage of species 
        # total = sum(imagesPerCat)
        # print(total)
        # print(imagesPerCat)
        # avgList = []
        # for cat in imagesPerCat:
        #     val = ((cat // total) * 100) // 1
        #     val = str(val) + "%"
        #     avgList.append(val)

        #barchart
        #Title: Percentage Accuracy According to Species
        data = {"Species":species, "Percentage of Accuracy":accuracyStats}   
        dataFrame = pd.DataFrame(data=data)
        dataFrame.plot.bar(x="Species", y="Percentage of Accuracy", rot=70, title="Percentage Accuracy According to Species", ylim=(85, 100))
        plt.subplots_adjust(bottom=.5)
        plt.show(block=True)

        #piechart
        #Title: Ratio of species classified
        df = pd.DataFrame(imagesPerCat, index=species, columns=['percentage'])
        df.plot(kind='pie', subplots=True, figsize=(8, 8), title="Ratio of species classified", legend=None)
        plt.show()

        #scatterchart
        # Title: Accuracy According by Sample Size
        data={'Image Sample Size':numberOfSamples,
            'Percentage Accuracy':accuracyStats}
        df = pd.DataFrame(data = data)
        df.plot.scatter(x = 'Image Sample Size', y = 'Percentage Accuracy', s = 100, title="Accuracy According by Sample Size")
        plt.show()

    def closeEvent(self, event):
        wx.Exit()
    
app = wx.App(False)
frame = MainWindow(None, "Primate Classifier")
app.MainLoop()