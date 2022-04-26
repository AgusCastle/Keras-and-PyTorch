import numpy as np
from torch import dropout
np.random.seed(42)

import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.layers import BatchNormalization
from keras.callbacks import TensorBoard

import tflearn.datasets.oxflower17 as oxflower17

X , Y = oxflower17.load_data(one_hot=True) 

class VGGKeras():
    def __init__(self, epochs, batch_size):
        self.epochs = epochs
        self.batch_size = batch_size

        self.model = Sequential()
        self.neuronalNetworkModel()

    def trainAndVal(self):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.history = self.model.fit(X, Y, batch_size=self.batch_size, epochs=self.epochs, verbose=1, validation_split=0.1, shuffle=True)

    def neuronalNetworkModel(self):
        self.model.add(Conv2D(64, 3, activation='relu', input_shape=(224, 224, 3)))
        self.model.add(Conv2D(64, 3, activation='relu'))
        self.model.add(MaxPooling2D(2, 2))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(128, 3, activation='relu'))
        self.model.add(Conv2D(128, 3, activation='relu'))
        self.model.add(MaxPooling2D(2, 2))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(256, 3, activation='relu'))
        self.model.add(Conv2D(256, 3, activation='relu'))
        self.model.add(Conv2D(256, 3, activation='relu'))
        self.model.add(MaxPooling2D(2, 2))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(512, 3, activation='relu'))
        self.model.add(Conv2D(512, 3, activation='relu'))
        self.model.add(Conv2D(512, 3, activation='relu'))
        self.model.add(MaxPooling2D(2, 2))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(512, 3, activation='relu'))
        self.model.add(Conv2D(512, 3, activation='relu'))
        self.model.add(Conv2D(512, 3, activation='relu'))
        self.model.add(MaxPooling2D(2, 2))
        self.model.add(BatchNormalization())

        self.model.add(Flatten())
        self.model.add(Dense(4096, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(4096, activation='relu'))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(17, activation='softmax'))

        self.model.sumary()

    def getAccuracyLoss_Train(self):
        return self.history.history['accuracy'], self.history.history['loss']

    def getAccuracyLoss_Val(self):
        return self.history.history['val_accuracy'], self.history.history['val_loss']
