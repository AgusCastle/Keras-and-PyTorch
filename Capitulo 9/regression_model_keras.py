import numpy as np
from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense, Dropout 
from tensorflow.keras.layers import BatchNormalization

(X_train, y_train), (X_valid, y_valid) = boston_housing.load_data()

class RegressionModelKeras():
    
    def __init__(self):
        self.model = Sequential()

        self.model.add(Dense(32, input_dim=13, activation='relu'))
        self.model.add(BatchNormalization())

        self.model.add(Dense(16, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.2))

        self.model.add(Dense(1, activation='linear'))

        self.model.summary()
    
    def predictCost(self, i):
        return self.model.predict(np.reshape(X_valid[i], [1,13]))

    def trainAndVal(self):
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.history = self.model.fit(X_train, y_train, batch_size=8, epochs=32, verbose=1, validation_data=(X_valid, y_valid))

    def getLoss_Train(self):
        return self.history.history['loss']

    def getLoss_Val(self):
        return self.history.history['val_loss']