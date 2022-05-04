from keras.applications.resnet import ResNet50
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator

resnet50 = ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=(224,224,3),
            pooling=None)

for layer in resnet50.layers:
    layer.trainable = False

# Instantiate the sequential model and add the ResNET50 model: 
model = Sequential()
model.add(resnet50)

# Add the custom layers atop the ResNET50 model: 
model.add(Flatten(name='flattened'))
model.add(Dropout(0.5, name='dropout'))
model.add(Dense(6, activation='softmax', name='predictions'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

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
    directory='/home/bringascastle/Escritorio/datasets/cartoon_face/train/',
    target_size=(224, 224),
    classes=['personai_01656','personai_01675','personai_01954','personai_02110','personai_03844','personai_04878'],
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=42)

valid_generator = valid_datagen.flow_from_directory(
    directory='/home/bringascastle/Escritorio/datasets/cartoon_face/test/',
    target_size=(224, 224),
    classes=['personai_01656','personai_01675','personai_01954','personai_02110','personai_03844','personai_04878'],
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=42)
history = model.fit(
    train_generator, 
    steps_per_epoch=16, 
    epochs=30, 
    validation_data=valid_generator, 
    validation_steps=16
    )

def getAccuracyLoss_Train():
        return history.history['accuracy'], history.history['loss']

def getAccuracyLoss_Val():
        return history.history['val_accuracy'], history.history['val_loss']