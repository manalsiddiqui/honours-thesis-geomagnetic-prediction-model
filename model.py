from tensorflow import keras
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Conv2D, MaxPooling2D, Flatten, GRU, Embedding
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score


def train_model(X_train, y_train, batch_size, epochs, model):
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)
    return model, history


def test_model(test_x, test_y, model):
    performance = model.evaluate(x=test_x, y=test_y)

    return performance

def predict_model(X_new, model):
    predictions = model.predict(X_new)
    return predictions


def basic_nn(input_size, lr):
    model = Sequential([
        Dense(64, input_dim=input_size, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='mean_squared_error',
                  metrics=['mean_absolute_error','accuracy'])

    return model



def basic_rnn(input_size, lr):
    model = Sequential([
        GRU(64, activation='relu', input_dim=input_size),
        Dense(1)
    ])

    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='mean_squared_error',
                  metrics=['mean_absolute_error','accuracy'])

    return model


def lstm_nn():
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(10, 1), return_sequences=True),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')


def lstm_model_two():
    model = Sequential([
        LSTM(64,dropout = 10 , stateful=True),
        LSTM(32,dropout = 10 , stateful=True),
        Dense(1)
    ])
    
    model.compile(optimizer='sgd', loss='mse')
    
