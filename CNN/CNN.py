# import numpy as np
# from keras.layers import Input, Dense, Activation, ZeroPadding2D, Flatten, Conv2D
# from keras.layers import MaxPooling2D
# from keras.models import Model
# from keras.preprocessing import image
# from keras.models import load_model
# from keras import metrics

# from keras.applications.imagenet_utils import preprocess_input
# from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot
# from keras.utils import plot_model
# from PIL import Image
# import keras.backend as K
# import tensorflow as tf
# import keras
# from keras.wrappers.scikit_learn import KerasClassifier


# from matplotlib.pyplot import imshow
# import matplotlib.pyplot as plt
# from pydicom import dcmread
# from pydicom.data import get_testdata_file
# from pathlib import Path
# from PIL import Image
# import os

# #######################################################################################################################

# numOfTestPoints = 2
# batchSize = 16
# numOfEpoches = 10
# #######################################################################################################################

# imgList = os.listdir('abajo')
# classes = []
# a = []

# def mean_pred(y_true, y_pred):
#     return K.mean(y_pred)

# # Get the softmax from folder name
# def getAsSoftmax(fname):
#     if (fname == 'benignum'):
#         return [1, 0]

#     else:
#         return [0, 1]

# # Return all images as numpy array, labels
# def get_imgs_frm_folder(path):
    
#     # y = np.empty(shape=[400],dtype=np.int8)
#     y = []
#     cnt = 0
    

#     for count in range(0, len(imgList)):
#         im_name = imgList[count]
#         #print(im_name)
#         path = str("abajo"+"\\" + im_name)
#         #print(path)
    
#         ds= dcmread(path)
#         arr = ds.pixel_array
#         #print(arr)
    

#         if np.all(arr[:,1913]==0):
#            arr = arr[::-1,::-1]
      
#         else:
#            arr = ds.pixel_array

#         a.append(arr)
#         y.append(getAsSoftmax("abajo"))
#         # print(getAsSoftmax(foldname))
    
    
#     algo = np.concatenate(([a]), axis=1)
#     #print(algo.shape) #hay 5 matrices, y cada una tiene 2294 filas y 1914 columnas 
#     z,x,y =np.nonzero(algo)
#     x1 = x.min()
#     x2 = x.max()
#     y1 = y.min()
#     y2 = y.max()
#     z1 = z.min()
#     z2 = z.max()
#     algo1= algo[z1:z2+1, x1:x2, y1:y2]
#     #print(algo1.shape)

#     return  y, cnt


# print(getAsSoftmax("abajo"))


import numpy as np
from keras.layers import Input, Dense, Activation, ZeroPadding2D, Flatten, Conv2D
from keras.layers import MaxPooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.models import load_model
from keras import metrics

from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from PIL import Image
import keras.backend as K
import tensorflow as tf
import keras
from keras.wrappers.scikit_learn import KerasClassifier

K.set_image_data_format('channels_last')
from matplotlib.pyplot import imshow
import os
import matplotlib.pyplot as plt 
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


# X_train = tf.expand_dims(X_train, axis=-1)
# print(X_train.shape)

#######################################################################################################################
# modelSavePath = 'my_model3.h5'
# numOfTestPoints = 2
batchSize = 16
numOfEpoches = 3
#######################################################################################################################

# defino funciones

def defModel(input_shape):
    X_input = Input(input_shape)

    # The max pooling layers use a stride equal to the pooling size
    normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

    X = Conv2D(16, (3, 3), strides=(1,1))(X_input)  # 'Conv.Layer(1)'

    X = Activation('relu')(X)

    X = MaxPooling2D((6, 1), strides=4)(X)  # MP Layer(2)

    X = Conv2D(16, (3, 3), strides=(1, 1))(X_input)  # 'Conv.Layer(1)'

    X = Activation('relu')(X)

    X = MaxPooling2D((3, 3), strides=3)(X)  # MP Layer(2)

    X = Conv2D(32, (3, 3), strides=(1, 1))(X)  # Conv.Layer(3)

    X = Activation('relu')(X)

    X = MaxPooling2D((2, 2), strides=2)(X)  # MP Layer(4)

    X = Conv2D(64, (2, 2), strides=(1, 1))(X)  # Conv.Layer(5)

    X = Activation('relu')(X)

    X = ZeroPadding2D(padding=(2, 2))(X)  # Output of convlayer(5) will be 82x82, we want 84x84

    X = MaxPooling2D((2, 2), strides=2)(X)  # MP Layer(6)

    X = Conv2D(64, (2, 2), strides=(1, 1))(X)  # Conv.Layer(7)

    X = Activation('relu')(X)

    X = ZeroPadding2D(padding=(2, 2))(X)  # Output of convlayer(7) will be 40x40, we want 42x42

    X = MaxPooling2D((3, 3), strides=3)(X)  # MP Layer(8)

    X = Conv2D(32, (3, 3), strides=(1, 1))(X)  # Con.Layer(9)

    X = Activation('relu')(X)

    X = Flatten()(X)  # Convert it to FC

    X = Dense(256, activation='relu')(X)  # F.C. layer(10)

    X = Dense(128, activation='relu')(X)  # F.C. layer(11)

    X = Dense(2, activation='softmax')(X)

    # ------------------------------------------------------------------------------

    model = Model(inputs=X_input, outputs=X, name='Model')

    return model
    

def train(batch_size, epochs):
    config = tf.compat.v1.ConfigProto()
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)

    model = defModel(X_train.shape[1:])

    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

    #history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)

    model.save('./modelo')

    preds = model.evaluate(X_test, Y_test_orig, batch_size=1, verbose=1, sample_weight=None)
    print(preds)

    print()
    print("Loss = " + str(preds[0]))
    print("Test Accuracy = " + str(preds[1]) + "\n\n\n\n\n")

    # acc = history.history['accuracy']
    # loss = history.history['loss']
    
    # epochs_range = range(numOfEpoches)

    # plt.figure(figsize=(8, 8))
    # #plt.subplot(1, 2, 1)
    # plt.plot(epochs_range, acc, label='Training Accuracy')
    # plt.plot(epochs_range, loss, label='Training Loss')
    # plt.legend(loc='lower right')
    # plt.title('Training Accuracy and loss')
    # plt.show()

    return model

classes = np.load('classes.npy')
print("Loading")
X_train = np.load('X_train.npy')
X_train = tf.expand_dims(X_train, axis=-1)
print(X_train.shape)
Y_train = np.load('Y_train.npy')
Y_train = tf.keras.utils.to_categorical(Y_train, 2)
X_test = np.load('X_test.npy')
X_test = tf.expand_dims(X_test, axis=-1)
Y_test_orig = np.load('Y_test_orig.npy')
Y_test_orig = tf.keras.utils.to_categorical(Y_test_orig, 2)
print(X_train.shape)

print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test_orig.shape))
model = train(batch_size=batchSize, epochs=numOfEpoches)





