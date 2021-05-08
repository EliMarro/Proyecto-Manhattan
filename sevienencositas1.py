# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
# from tensorflow.keras.layers import Conv1D, MaxPool1D

# from tensorflow.keras.optimizers import Adam


# print(tf.__version__)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import datasets, metrics
from sklearn.preprocessing import LabelEncoder

file = 'CMMD.csv'

cancer = pd.read_csv(file)


#print(cancer)
#print (cancer.head(10))

# contamos el numero de cancerigenos y de benignos en la columna classification
#print(cancer['classification'].value_counts()) 

# para saber el tipo de dato que almacena cada columna
#print(cancer.dtypes)

# Ahora voy a transformar, de la columna "classification" (columna 5 empezando en 0) Benign en 0 y Malign en 1
encoder = LabelEncoder()
#cancer.iloc[:, 1] = encoder.fit_transform(cancer.iloc[:,1].values) 
encoder = encoder.fit_transform(cancer.iloc[:,5].values) 
print(encoder)
#print(cancer.shape)


arr = np.array(encoder)
print(arr.shape)