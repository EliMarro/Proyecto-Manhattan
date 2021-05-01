# import numpy as np
# import png 
# import pydicom


# ds = pydicom.dcmread('D1-0001--1-1.dcm')
# shape = ds.pixel_array.shape
# print(shape)
# Convert to float to avoid overflow or underflow losses.
# image_2d = ds.pixel_array.astype(float)
# Rescaling grey scale between 0-255
# image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0
# Convert to uint
# image = np.uint8(image_2d_scaled)
# print(image_2d_scaled)

# image.save_as("Test4.npy")

import numpy as np
import matplotlib.pyplot as plt
from pydicom import dcmread
from pydicom.data import get_testdata_file
from pathlib import Path
from PIL import Image
import os
import shutil


# path = get_testdata_file("arriba")

# path= r"D1-0001--1-1.dcm"


imgList = os.listdir('arriba')
#print(imgList)

# leo imagen de referencia
# ruta = "1-1.dcm"
# imagen = dcmread(ruta)

# row, columns = imagen.pixel_array.shape[0], imagen.pixel_array.shape[1] # numero de filas y de columnas

#inicializo matriz de ceros
# train = np.zeros((len(imgList), row, columns), dtype = int)


# print(train.shape)
a=[]

for count in range(0, len(imgList)):
    im_name = imgList[count]
    #print(im_name)
    path = str("arriba"+"\\" + im_name)
    #print(path)
    
    ds= dcmread(path)

    arr = ds.pixel_array
    a.append(arr)
    #print(arr)
    #list=arr.tolist()
    #print(list)
    #algo = np.concatenate((list), axis=0)
    #algo = np.concatenate ([arr], axis=0)
   
    #print(algo.shape)
    #print(algo)
    #print(algo.shape)

algo = np.concatenate(([a]), axis=1)
print(algo.shape) #hay 5 matrices, y cada una tiene 2294 filas y 1914 columnas 
z,x,y =np.nonzero(algo)
x1 = x.min()
x2 = x.max()
y1 = y.min()
y2 = y.max()
z1 = z.min()
z2 = z.max()
algo1= algo[z1:z2+1, x1:x2, y1:y2]
print(algo1.shape)

#print(algo1)




#     plt.imshow(arr, cmap="gray")



#     np.save("train", arr)
    
#     # pil_image=Image.fromarray(arr)
#     # pil_image.show()
#     # plt.show()

# train = np.load('train.npy')
# print(train.shape)
# # para comprobar que lo guarda bien



