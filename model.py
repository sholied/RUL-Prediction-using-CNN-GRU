import tensorflow as tf
import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, LearningRateScheduler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Flatten, Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, TimeDistributed, Masking, Activation, RepeatVector
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Activation


def model_gru(num_gru, seq_length, num_features, num_labels):
    initial_unit = 256
    n = num_gru - 1
    model = Sequential()
    
    # First GRU layer with input shape
    model.add(GRU(
        input_shape=(seq_length, num_features),
        units=initial_unit,
        dropout=0.05,
        return_sequences=True
    ))
    
    # Additional GRU layers
    for i in range(n):
        return_sequences = (i + 1) != n  # Last GRU should have return_sequences=False
        model.add(GRU(
            units=int(initial_unit / (2 * (i + 1))),
            dropout=0.05,
            return_sequences=return_sequences
        ))
    
    # Dense layers
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=32, activation='relu'))

    model.add(Dense(units=1))
    
    return model


def model_lstm(num_lstm, seq_length, num_features, num_labels):
    initial_unit = 256
    n = num_lstm - 1
    model = Sequential()
    
    # First LSTM layer with input shape
    model.add(LSTM(
        input_shape=(seq_length, num_features),
        units=initial_unit,
        dropout=0.05,
        return_sequences=True
    ))
    
    # Additional LSTM layers
    for i in range(n):
        return_sequences = (i + 1) != n  # Last LSTM should have return_sequences=False
        model.add(LSTM(
            units=int(initial_unit / (2 * (i + 1))),
            dropout=0.05,
            return_sequences=return_sequences
        ))
    
    # Dense layers
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=32, activation='relu'))
    
    model.add(Dense(units=num_labels))
    
    return model

def model_cnngru(num_cnn, num_gru, seq_length, num_features, num_labels):
    initial_unit = 256
    initial_filter = 128
    n = num_gru - 1
    m = num_cnn - 1
    model = Sequential()
    
    # First CNN layer
    model.add(Conv1D(filters=initial_filter, kernel_size=3, strides=1, activation='relu', padding='causal', 
                     input_shape=(seq_length, num_features)))
    model.add(MaxPooling1D(pool_size=2, strides=1, padding="valid"))
    model.add(Dropout(0.05))
    
    # Additional CNN layers
    for j in range(m):
        model.add(Conv1D(filters=int(initial_filter/(2*(j+1))), kernel_size=3, strides=1, activation='relu', padding='causal'))
        model.add(MaxPooling1D(pool_size=2, strides=1, padding="valid"))
        model.add(Dropout(0.05))
    
    # First GRU layer
    model.add(GRU(units=initial_unit, dropout=0.05, return_sequences=True))
    
    # Additional GRU layers
    for i in range(n):
        return_sequences = (i + 1) != n  # Last GRU should have return_sequences=False
        model.add(GRU(units=int(initial_unit / (2 * (i + 1))), dropout=0.05, return_sequences=return_sequences))
    
    # Dense layers
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=32, activation='relu'))

    model.add(Dense(units=num_labels))
    
    return model


def model_cnnlstm(num_cnn, num_lstm, seq_length, num_features, num_labels):
    initial_unit = 256
    initial_filter = 128
    n = num_lstm - 1
    m = num_cnn - 1
    model = Sequential()
    
    # First CNN layer
    model.add(Conv1D(filters=initial_filter, kernel_size=3, strides=1, activation='relu', padding='causal', 
                     input_shape=(seq_length, num_features)))
    model.add(MaxPooling1D(pool_size=2, strides=1, padding="valid"))
    model.add(Dropout(0.05))
    
    # Additional CNN layers
    for j in range(m):
        model.add(Conv1D(filters=int(initial_filter/(2*(j+1))), kernel_size=3, strides=1, activation='relu', padding='causal'))
        model.add(MaxPooling1D(pool_size=2, strides=1, padding="valid"))
        model.add(Dropout(0.05))
    
    # First LSTM layer
    model.add(LSTM(units=initial_unit, dropout=0.05, return_sequences=True))
    
    # Additional LSTM layers
    for i in range(n):
        return_sequences = (i + 1) != n  # Last LSTM should have return_sequences=False
        model.add(LSTM(units=int(initial_unit / (2 * (i + 1))), dropout=0.05, return_sequences=return_sequences))
    
    # Dense layers
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=32, activation='relu'))

    model.add(Dense(units=num_labels))
    
    return model