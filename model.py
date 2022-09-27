import tensorflow as tf
import keras
from tensorflow.keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, GRU, Flatten, Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, TimeDistributed, Masking, Activation, RepeatVector
import keras.backend as K
from keras.layers.core import Activation


def model_gru(num_gru, seq_length, num_features, num_labels):
    initial_unit = 256
    n = num_gru-1
    model = Sequential()
    model.add(GRU(
                input_shape=(seq_length, num_features),
                units=initial_unit,
                dropout=0.05,
                return_sequences=True))
    for i in range(n):
        if (i+1) == n:
          model.add(GRU(units=int(initial_unit/(2*(i+1))),
                        dropout=0.05,
                        return_sequences=False))
        else:
          model.add(GRU(units=int(initial_unit/(2*(i+1))),
                        dropout=0.05,
                        return_sequences=True))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=num_labels, activation='relu'))
    
    return model

def model_cnngru(num_cnn, num_gru, seq_length, num_features, num_labels):
    initial_unit = 256
    initial_filter = 128
    n = num_gru-1
    m = num_cnn-1
    model = Sequential()
    model.add(Conv1D(filters=initial_filter, kernel_size=3, strides=1 , activation='relu', padding='causal', 
                     input_shape=(seq_length, num_features)))
    model.add(MaxPooling1D(pool_size=2, strides=1, padding="valid"))
    model.add(Dropout(0.05))
    for j in range(m):
        model.add(Conv1D(filters=int(initial_filter/(2*(j+1))), kernel_size=3, strides=1 , activation='relu', padding='causal'))
        model.add(MaxPooling1D(pool_size=2, strides=1, padding="valid"))
        model.add(Dropout(0.05))
    model.add(GRU(
                units=initial_unit,
                dropout=0.05,
                return_sequences=True))
    for i in range(n):
        if (i+1) == n:
          model.add(GRU(units=int(initial_unit/(2*(i+1))),
                        dropout=0.05,
                        return_sequences=False))
        else:
          model.add(GRU(units=int(initial_unit/(2*(i+1))),
                        dropout=0.05,
                        return_sequences=True))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=num_labels, activation='relu'))
    
    return model
