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

from pydicom import dcmread
from pydicom.data import get_testdata_file
from PIL import Image


#######################################################################################################################
# modelSavePath = 'my_model3.h5'
# numOfTestPoints = 2
batchSize = 10
numOfEpoches = 1
#######################################################################################################################

# defino funciones

def defModel(input_shape):
    X_input = Input(input_shape)

    
    normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)


    X = layers.Conv2D(16, (3, 3), name='conv_second', strides=(1, 1))(X_input)  # 'Conv.Layer(1)'

    X = layers.Activation('relu')(X)

    X = layers.MaxPooling2D((3, 3), strides=3)(X)  # MP Layer(2)

    X = layers.Conv2D(32, (3, 3), name='conv_third', strides=(1, 1))(X)  # Conv.Layer(3)

    X = layers.Activation('relu')(X)

    X = layers.MaxPooling2D((2, 2), strides=2)(X)  # MP Layer(4)

    X = layers.Conv2D(64, (2, 2), name='conv_fourth', strides=(1, 1))(X)  # Conv.Layer(5)

    X = layers.Activation('relu')(X)

    X = layers.ZeroPadding2D(padding=(2, 2))(X)  # Output of convlayer(5) will be 82x82, we want 84x84

    X = layers.MaxPooling2D((2, 2), strides=2)(X)  # MP Layer(6)

    X = layers.Conv2D(64, (2, 2), name='conv_fifth', strides=(1, 1))(X)  # Conv.Layer(7)

    X = layers.Activation('relu')(X)

    X = layers.ZeroPadding2D(padding=(2, 2))(X)  # Output of convlayer(7) will be 40x40, we want 42x42

    X = layers.MaxPooling2D((3, 3), strides=3)(X)  # MP Layer(8)

    X = layers.Conv2D(32, (3, 3), name='conv_sixth', strides=(1, 1))(X)  # Con.Layer(9)

    X = layers.Activation('relu')(X)

    X = layers.Flatten()(X)  # Convert it to FC

    X = layers.Dense(256, activation='relu')(X)  # F.C. layer(10)

    X = layers.Dense(128, activation='relu')(X)  # F.C. layer(11)

    X = layers.Dense(2, activation='softmax')(X)

    # ------------------------------------------------------------------------------

    model = Model(inputs=X_input, outputs=X, name='Model')

    model.summary()

    return model


def train(batch_size, epochs):
    config = tf.compat.v1.ConfigProto()
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)

    model = defModel(X_train.shape[1:])

    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

    #history = model.fit(X_train, Y_train,(X_test, Y_test_orig), epochs=epochs, batch_size=batch_size)
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



def predict(img, savedModelPath, showImg=True):
    model = load_model(savedModelPath)
   
    x = img
    if showImg:
        Image.fromarray(np.array(img, np.float16), 'RGB')
    x = tf.expand_dims(x, axis=0)

    softMaxPred = model.predict(x)
    print("prediction from CNN: " + str(softMaxPred) + "\n")
    probs = softmaxToProbs(softMaxPred)

   
    maxprob = 0
    maxI = 0
    for j in range(len(probs)):
        # print(str(j) + " : " + str(round(probs[j], 4)))
        if probs[j] > maxprob:
            maxprob = probs[j]
            maxI = j
    # print(softMaxPred)
    print("prediction index: " + str(maxI))
    return maxI, probs

    

def softmaxToProbs(soft):
    z_exp = [np.math.exp(i) for i in soft[0]]
    sum_z_exp = sum(z_exp)
    return [(i / sum_z_exp) * 100 for i in z_exp]



def predictImage(img_path=get_testdata_file("1-2.dcm"), arrayImg=None, printData=True):
    a = []
    if arrayImg == None:
        train = np.load('X_train.npy')
        row, columns = train.shape[1], train.shape[2]
        

        img_path= r"1-2.dcm"
        ds= dcmread(img_path)
        arr= ds.pixel_array
        if np.all(arr[:,1913]==0):
            arr = arr[::-1,::-1]
        else:
            arr = ds.pixel_array
        
        
        arr1= arr[0:row, 0:columns]
        a.append(arr1)

    #Image.fromarray(arr1[0]).show()

    classes = []
    classes.append("Benign")
    classes.append("Malignant")
    

    compProbs = []
    compProbs.append(0)
    compProbs.append(0)
    

    for i in range(len(a)):
        ___, probs = predict(a[i], './modelo', showImg=False)

        for j in range(len(classes)):
            if printData:
                print(str(classes[j]) + " : " + str(round(probs[j], 2)) + "%")
            compProbs[j] += probs[j]


print("1. Do you want to train the network\n"
      "2. Test the model\n(Enter 1 or 2)?\n")
ch = int(input())
if ch == 1:
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

elif ch == 2:
    c = int(input("1. Test from random images\n2. Test your own custom image\n(Enter 1 or 2)\n"))
    if c == 1:
        classes = np.load('classes.npy')
        print("Loading")
        X_train = np.load('X_train.npy')
        X_train = tf.expand_dims(X_train, axis=-1)
        Y_train = np.load('Y_train.npy')
        Y_train = tf.keras.utils.to_categorical(Y_train, 2)
        X_test = np.load('X_test.npy')
        X_test = tf.expand_dims(X_test, axis=-1)
        Y_test_orig = np.load('Y_test_orig.npy')
        Y_test_orig = tf.keras.utils.to_categorical(Y_test_orig, 2)
   
        testImgsX = []
        testImgsY = []
        ran = []
        print("X_train shape: " + str(X_train.shape))
        print("Y_train shape: " + str(Y_train.shape))
   
        for i in range(2):
            ran.append(np.random.randint(0, X_train.shape[0] - 1))

        for ranNum in ran:
            testImgsX.append(X_train[ranNum])
            testImgsY.append(Y_train[ranNum])


        X_train = None
        Y_train = None

        print("testImgsX shape: " + str(len(testImgsX)))
        print("testImgsY shape: " + str(len(testImgsY)))
    

        cnt = 0.0

        classes = []
        classes.append("Benign")
        classes.append("Malignant")
        

        compProbs = []
        compProbs.append(0)
        compProbs.append(0)

        for i in range(len(testImgsX)):

            print("\n\nTest image " + str(i + 1) + " prediction:\n")

            predi, probs = predict(testImgsX[i], './modelo', showImg=False)

            for j in range(len(classes)):
                print(str(classes[j]) + " : " + str(round(probs[j], 4)) + "%")
                compProbs[j] += probs[j]

            maxi = 0
            for j in range(len(testImgsY[0])):
                if testImgsY[i][j] == 1:
                    maxi = j
                    break   
                
            if predi == maxi:
                cnt += 1

        print("% of images that are correct: " + str((cnt / len(testImgsX)) * 100))
    
    elif c == 2:
        predictImage()