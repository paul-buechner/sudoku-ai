import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler

'''
# !! Printing usable training hardware !!
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
'''

(x_train, y_train), (x_test, y_test) = mnist.load_data()

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.10,
    width_shift_range=0.1,
    height_shift_range=0.1)


def model():
    model = Sequential()

    model.add(
        Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=5, strides=2,
                     padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=5, strides=2,
                     padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(128, kernel_size=4, activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax'))
    return model


model = model()

# Compile...
model.compile(optimizer="Adam",
              loss="categorical_crossentropy", metrics=["accuracy"])

# Decrease learning rate each epoch
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)


# Train model...
epochs = 2

X_train2, X_val2, Y_train2, Y_val2 = train_test_split(
    x_train, y_train, test_size=0.1)
model.fit(datagen.flow(X_train2, Y_train2, batch_size=64),
          epochs=epochs, steps_per_epoch=X_train2.shape[0]//64,
          validation_data=(x_test, y_test), callbacks=[annealer], verbose=1)


cnn_results = model.evaluate(x_test, y_test)


# Save Model
model.save('model/model_tf.h5')
