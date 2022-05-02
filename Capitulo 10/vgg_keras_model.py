import numpy as np
np.random.seed(42)
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.layers import BatchNormalization
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
import os; 
os.environ['CUDA_VISIBLE_DEVICES'] = 'cuda:0'


# Instantiate two image generator classes:
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    data_format='channels_last',
    rotation_range=30,
    horizontal_flip=True,
    fill_mode='reflect')

valid_datagen = ImageDataGenerator(
    rescale=1.0/255,
    data_format='channels_last',
    rotation_range=30,
    horizontal_flip=True,
    fill_mode='reflect')

# Define the train and validation generators: 
train_generator = train_datagen.flow_from_directory(
    directory='/home/bringascastle/Escritorio/datasets/cartoon_face/train',
    target_size=(224, 224),
    classes=['personai_01656','personai_01675','personai_01954','personai_02110','personai_03844','personai_04878'],
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=42)

valid_generator = valid_datagen.flow_from_directory(
    directory='/home/bringascastle/Escritorio/datasets/cartoon_face/test',
    target_size=(224, 224),
    classes=['personai_01656','personai_01675','personai_01954','personai_02110','personai_03844','personai_04878'],
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=42)



class VGGKeras():
    def __init__(self, epochs, batch_size):
        self.epochs = epochs
        self.batch_size = batch_size

        self.model = Sequential()
        self.neuronalNetworkModel()

    def trainAndVal(self):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.history = self.model.fit(
    train_generator, 
    steps_per_epoch=16, 
    epochs=self.epochs, 
    validation_data=valid_generator, 
    validation_steps=16
    )

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

        self.model.add(Dense(6, activation='softmax'))

        self.model.summary()

    def getAccuracyLoss_Train(self):
        return self.history.history['accuracy'], self.history.history['loss']

    def getAccuracyLoss_Val(self):
        return self.history.history['val_accuracy'], self.history.history['val_loss']
