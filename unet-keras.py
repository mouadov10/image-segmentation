import random
import tensorflow as tf
import numpy as np
from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img
import tensorflow
import scipy
import sklearn
import cv2
import os
from sklearn.feature_extraction import image
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm

from keras import backend as K
from skimage.io import imread, imshow
from skimage.transform import resize
import os; os.environ['KERAS_BACKEND'] = 'theano'
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from sklearn.utils import class_weight




input_path = os.path.join('..', 'projet/input', '2015_boe_chiu', '2015_BOE_Chiu')
subject_path = [os.path.join(input_path, 'Subject_0{}.mat'.format(i)) for i in range(1, 10)] + [os.path.join(input_path, 'Subject_10.mat')]
data_indexes = [10, 15, 20, 25, 28, 30, 32, 35, 40, 45, 50]

# mat = scipy.io.loadmat(subject_path[0])
# img_tensor = mat['images']
# manual_fluid_tensor_1 = mat['manualFluid1']


# img_array = np.transpose(img_tensor, (2, 0, 1))
# manual_fluid_array = np.transpose(manual_fluid_tensor_1, (2, 0, 1))


def thresh(x):
    if x == 0:
        return 0
    else:
        return 1


thresh = np.vectorize(thresh, otypes=[np.float])

width = 256
height = 256
dim = (width, height)


def create_dataset(paths):
    x = []
    y = []
    Z = np.zeros([61, 256, 256])
    L = np.zeros([61, 256, 256])

    for path in tqdm(paths):
        mat = scipy.io.loadmat(path)
        img_tensor = mat['images']
        fluid_tensor = mat['manualFluid1']

        img_array = np.transpose(img_tensor, (2, 0, 1)) / 255
        fluid_array = np.transpose(fluid_tensor, (2, 0, 1))
        fluid_array = thresh(fluid_array)
        for j in range(61):
            Z[j] = cv2.resize(img_array[j], dim, interpolation=cv2.INTER_AREA)
            L[j] = cv2.resize(fluid_array[j], dim, interpolation=cv2.INTER_AREA)

        for idx in data_indexes:
            x += [np.expand_dims(Z[idx], 0)]
            y += [np.expand_dims(L[idx], 0)]
    return np.array(x), np.array(y)


def create_datatest(paths):
    x = []
    
    T = np.zeros([61, 256, 256])
    for path in tqdm(paths):
        mat = scipy.io.loadmat(path)
        img_tensor = mat['images']
        img_array = np.transpose(img_tensor, (2, 0, 1)) / 255
        for j in range(61):
            T[j] = cv2.resize(img_array[j], dim, interpolation=cv2.INTER_AREA)

        for idx in data_indexes:
            x += [np.expand_dims(T[idx+1], 0)]
           
    return np.array(x)

x_train, y_train = create_dataset(subject_path[:9])
x_val, y_val = create_dataset(subject_path[9:])
x_test= create_datatest(subject_path[9:])


# print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)

# plt.imshow(img_array[25])

# plt.imshow(manual_fluid_array[25])

## Build Model ##

IMG_CHANNELS = 1

inputs = tf.keras.layers.Input((IMG_CHANNELS, height, width))
# K.set_image_data_format('channels_first')

c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',
                            data_format='channels_first')(inputs)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',
                            data_format='channels_first')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2), data_format='channels_first')(c1)

c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',
                            data_format='channels_first')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',
                            data_format='channels_first')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2), data_format='channels_first')(c2)

c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',
                            data_format='channels_first')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',
                            data_format='channels_first')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2), data_format='channels_first')(c3)

c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',
                            data_format='channels_first')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',
                            data_format='channels_first')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_first')(c4)

c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',
                            data_format='channels_first')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',
                            data_format='channels_first')(c5)

# Expansive path
u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', data_format='channels_first')(c5)
u6 = tf.keras.layers.concatenate([u6, c4], axis=1)
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',
                            data_format='channels_first')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',
                            data_format='channels_first')(c6)

u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', data_format='channels_first')(c6)
u7 = tf.keras.layers.concatenate([u7, c3], axis=1)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',
                            data_format='channels_first')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',
                            data_format='channels_first')(c7)

u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', data_format='channels_first')(c7)
u8 = tf.keras.layers.concatenate([u8, c2], axis=1)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',
                            data_format='channels_first')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',
                            data_format='channels_first')(c8)

u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same', data_format='channels_first')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=1)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',
                            data_format='channels_first')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',
                            data_format='channels_first')(c9)

outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', data_format="channels_first")(c9)


## Model Configurations ##
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
loss_fn = tf.keras.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])
model.summary()

#Modelcheckpoint
checkpointer = tf.keras.callbacks.ModelCheckpoint('unet_model.h5', verbose=1, save_best_only=True)
callbacks = [tf.keras.callbacks.EarlyStopping(patience=6, monitor='val_loss')]



## Training Model ##

class_weights = {0:0.11,1:0.89}

# weights = class_weight.compute_class_weight('balanced',
#                                             np.unique(y_train),
#                                             y_train)
# sample_weights = class_weight.compute_sample_weight('balanced', tuple(y_train))
results = model.fit(x_train,y_train, validation_data=(x_val, y_val), validation_split=0, batch_size=10, epochs=600, callbacks=callbacks)
result = model.predict(x_train)
result = result > 0.4

## Plot Train and Val Loss ##
plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

## Plot Train and Val Accuracy ##
plt.plot(results.history['accuracy'])
plt.plot(results.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

def plot_examples(datax, datay, num_examples=3):
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(18,5*num_examples))
    m = datax.shape[0]
    for row_num in range(num_examples):
        image_indx = np.random.randint(m)
        result = model.predict(datax)
        image_arr = result > 0.2
        ax[row_num][0].imshow(tf.keras.backend.squeeze(datax[image_indx], 0))
        ax[row_num][0].set_title("Orignal Image")
        ax[row_num][1].imshow(tf.keras.backend.squeeze(result[image_indx], 0))
        ax[row_num][1].set_title("Segmented Image")
        ax[row_num][2].imshow(tf.keras.backend.squeeze(datay[image_indx], 0))
        ax[row_num][2].set_title("Target image")
    plt.show()

def plot_test(datax, num_examples=3):
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(18,5*num_examples))
    m = datax.shape[0]
    for row_num in range(num_examples):
        image_indx =np.random.randint(m)
        result = model.predict(datax)
        image_arr = result > 0.2
        ax[row_num][0].imshow(tf.keras.backend.squeeze(datax[image_indx], 0))
        ax[row_num][0].set_title("Orignal test Image")
        ax[row_num][1].imshow(tf.keras.backend.squeeze(result[image_indx], 0))
        ax[row_num][1].set_title("Segmented test Image")
    plt.show()    

## Plot Training Examples ##
plot_examples(x_train, y_train)


## Plot Val Examples ##
plot_examples(x_val, y_val)

plot_test(x_test)