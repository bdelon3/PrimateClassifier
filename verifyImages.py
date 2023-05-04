
from struct import unpack
import os
import pathlib

data_dir = "Primates"
data_dir = pathlib.Path(data_dir)

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

imageList = list(data_dir.glob('*/*.jpg'))

# print(roses)
counting = 0
total = 0
for item in imageList:
    total += 1
    try:
        img = JPEG(item)
        img.decode()
    except:
        print(item)
        counting += 1
        os.remove(item)

print("total", total)
print("Removed: ", counting)
counting = 0
total = 0