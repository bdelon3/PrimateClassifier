import glob
index = 0
Images = []
for file in glob.glob("Images/*"):
    Images.append((file, index))
    index += 1

print(Images)
