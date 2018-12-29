import os
import pickle
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, Reshape
from sklearn.model_selection import train_test_split

def get_trained_model(X, Y, e=50, bs=64):
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1)

    model = Sequential()
    model.add(Dense(32, input_shape=X[0].shape))
    model.add(Dense(64))
    model.add(Dense(64))
    model.add(Dense(len(Y[0][0])))
    model.add(Reshape(Y[0].shape))
    model.compile(loss='mse', optimizer='Adam')

    metrics = model.fit(X_train, 
            Y_train,
            epochs=e,
            batch_size=bs,
            verbose=0,
            validation_data=(X_val, Y_val))

    return model, metrics
