import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import SGD
from keras import utils as np_utils

(X_train, y_train), (X_valid, y_valid) = mnist.load_data()

class Net_Keras:
    def __init__(self, epochs):
        self.X_train = X_train.reshape(60000, 784).astype('float32')
        self.X_valid = X_valid.reshape(10000, 784).astype('float32')

        self.X_train /= 255
        self.X_valid /= 255

        self.n_classes = 10
        self.y_train = keras.utils.np_utils.to_categorical(y_train, self.n_classes)
        self.y_valid = keras.utils.np_utils.to_categorical(y_valid, self.n_classes)

        self.model = Sequential()
        self.epochs = epochs
    
    def modelNet(self):
        self.model.add(Dense(64, activation='relu', input_shape=(784,)))
        self.model.add(BatchNormalization())

        self.model.add(Dense(64, activation='relu'))
        self.model.add(BatchNormalization())

        self.model.add(Dense(64, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.2))

        self.model.add(Dense(10, activation='softmax'))

        self.model.summary()

    def trainAndVal(self):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.history = self.model.fit(self.X_train, self.y_train, batch_size=128, epochs=self.epochs, verbose=1, validation_data=(self.X_valid, self.y_valid))

    def getAccuracyLoss_Train(self):
        return self.history.history['accuracy'], self.history.history['loss']

    def getAccuracyLoss_Val(self):
        return self.history.history['val_accuracy'], self.history.history['val_loss']