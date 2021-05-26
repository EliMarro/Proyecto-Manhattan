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



#######################################################################################################################
# modelSavePath = 'my_model3.h5'
# numOfTestPoints = 2
batchSize = 16
numOfEpoches = 1
#######################################################################################################################

def defModel(input_shape):
    X_input = Input(input_shape)

    # The max pooling layers use a stride equal to the pooling size
    #normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

    # model = Sequential([
    #     layers.experimental.preprocessing.Rescaling(1./255, X_input),
    #     layers.Conv2D(16, 3, padding='same', activation='relu'),
    #     layers.MaxPooling2D(),
    #     layers.Conv2D(32, 3, padding='same', activation='relu'),
    #     layers.MaxPooling2D(),
    #     layers.Conv2D(64, 3, padding='same', activation='relu'),
    #     layers.MaxPooling2D(),
    #     layers.Flatten(),
    #     layers.Dense(128, activation='relu'),
    #     layers.Dense(2)
    #     ])
    # model.build


    # X = layers.Conv2D(16, (3, 3), name='conv_first', strides=(1,1))(X_input)  # 'Conv.Layer(1)'

    # X = layers.Activation('relu')(X)

    # X = layers.MaxPooling2D((6, 1), strides=4)(X)  # MP Layer(2)

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

    #X = layers.Dense(256, activation='relu')(X)  # F.C. layer(10)

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
  model = train(batch_size=batchSize, epochs=numOfEpoches)


  path = get_testdata_file("1-1.dcm")
  path= r"1-1.dcm"
  ds= dcmread(path)

  arr= ds.pixel_array

  if np.all(arr[:,1913]==0):
    arr = arr[::-1,::-1]
          
  else:
    arr = ds.pixel_array

  x,y =np.nonzero(arr)
  x1 = x.min()
  x2 = x.max()
  y1 = y.min()
  y2 = y.max()
  arr1= arr[x1:x2, y1:y2]

  arr1 = tf.expand_dims(arr1, 0) # Create a batch
  predictions = model.predict(arr1)
  score = tf.nn.softmax(predictions[0])
  print(score)
  


    































# batch_size = 32
# img_height = 180
# img_width = 180

# num_classes = 5
# class_names = train_ds.class_names

# model = Sequential([
#   layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
#   layers.Conv2D(16, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(32, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(64, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Flatten(),
#   layers.Dense(128, activation='relu'),
#   layers.Dense(num_classes)
# ])
    

# sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
# sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

# img = keras.preprocessing.image.load_img(
#     sunflower_path, target_size=(img_height, img_width)
# )
# img_array = keras.preprocessing.image.img_to_array(img)
# img_array = tf.expand_dims(img_array, 0) # Create a batch

# predictions = model.predict(img_array)
# score = tf.nn.softmax(predictions[0])

# print(
#     "This image most likely belongs to {} with a {:.2f} percent confidence."
#     .format(class_names[np.argmax(score)], 100 * np.max(score))
# )