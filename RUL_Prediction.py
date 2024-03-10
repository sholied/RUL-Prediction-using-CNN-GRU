from email.policy import default
import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
from pathlib import Path
import joblib

# Set for reproducability
np.random.seed(101)  
PYTHONHASHSEED = 0

from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from tqdm import tqdm, tqdm_notebook

from data_generator import TSDataGenerator, split_data, create_generators
from util import set_log_dir, rmse
from util import LRDecay
from data_util import *
from model import *
import tensorflow.keras.backend as K

MODEL_DIR = os.path.abspath("model")

persist_run_stats = True # Enable for saving results to CouchDB
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

#Load Dataset from folder
cols = ['id', 'cycle' ]

# Three operational setting columns
setting_cols = ['setting' + str(i) for i in range(1,4)]
cols.extend(setting_cols)

# Twenty one sensor columns
sensor_cols = ['s' + str(i) for i in range(1,22)]
cols.extend(sensor_cols)

sort_cols = ['id','cycle']


def load_data(paths, col_names, sort_cols):
    # read data 
    df = pd.DataFrame()
    for p in paths:
        instance_df = pd.read_csv(p, sep=" ", header=None)
        instance_df.drop(instance_df.columns[[26, 27]], axis=1, inplace=True)
        instance_df.columns = col_names
        instance_df['filename'] = os.path.splitext(os.path.basename(p))[0]
        
        df = pd.concat((df, instance_df), sort=False) 

    df['condition'] = df['filename'].apply( lambda f: fn_condition_map[f])
    df['id'] = df['id'] + df['filename'].apply( lambda f: fn_id_map[f])
    df.drop(['filename'], axis=1, inplace=True)
    df = df.sort_values(sort_cols)
    return df

def r2_keras(y_true, y_pred):
    """Coefficient of Determination 
    """
    SS_res =  K.sum(K.square( y_true - y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

#Calculate Training DATA RUL
def calc_training_rul(df):
    # Data Labeling - generate column RUL
    rul = pd.DataFrame(df.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'max']
    df = df.merge(rul, on=['id'], how='left')
    df['RUL'] = df['max'] - df['cycle']
    df.drop('max', axis=1, inplace=True)
    return df

def train_model(inp_model, num_epochs):
    # Create the model
    if inp_model == "stacked_gru":
        model = stacked_gru(batch_size, sequence_length, num_features, num_labels)
    elif inp_model == "cnn_gru":
        model = model_cnngru(2, 2, sequence_length, num_features, num_labels)
    elif inp_model == "single_gru":
        model = model_gru(batch_size, sequence_length, num_features, num_labels)
    else:
        print("-------------------------------------------------------------------------\n")
        print("model incorrect, please choose one : single_gru or stacked_gru or cnn_gru !\n")
        print("-------------------------------------------------------------------------\n")

    model.compile(loss=rmse, optimizer='rmsprop',metrics=['mse',r2_keras])

    print(model.summary())

    start = time.time()
    training_runs = 4
    initial_epoch = 0
    epochs_before_decay = 15
    lrate = 1e-3
    patience = 100
        
    tensorboard = TensorBoard(log_dir=log_dir,
                            histogram_freq=0, write_graph=True, write_images=False)

    checkpointer = ModelCheckpoint(checkpoint_path + '.keras', verbose=1, save_best_only=True)

    epoch_loss_history = []
    epoch_val_history = []
    epoch_lr_history = []

    for tr_run in range(training_runs):

        print("Training run: {}, epoch: {}".format(tr_run, initial_epoch))

        t_df, v_df = split_data(train_df, randomize=True, train_pct=.8)
        train_data_generator, val_data_generator = create_generators(t_df, v_df, 
                                                                    feature_cols, 
                                                                    label_cols, 
                                                                    batch_size=batch_size, 
                                                                    sequence_length=sequence_length, 
                                                                    randomize=True, 
                                                                    loop=True,
                                                                    pad=False,
                                                                    verbose=True)
        
        # Callbacks
        lr_decay = LRDecay(initial_lrate=lrate, epochs_step=epochs_before_decay)
        lr_scheduler = LearningRateScheduler(lr_decay.step_decay, verbose=1)
        
        earlystopper = EarlyStopping(patience=patience, verbose=1)
        
        callbacks = [ tensorboard, checkpointer, lr_scheduler, earlystopper]

        # fit the network
        history = model.fit(
            train_data_generator.generate(), 
            validation_data=val_data_generator.generate(), 
            initial_epoch=initial_epoch,
            epochs=num_epochs, 
            steps_per_epoch=train_data_generator.summary()['max_iterations'],
            validation_steps=val_data_generator.summary()['max_iterations'],
            shuffle=False,
            verbose=1,
            callbacks=callbacks )
        
        # pick up after the last epoch
        if len(history.epoch) > 0:        
            initial_epoch = history.epoch[-1] + 1
        
        # TODO fix, sometimes Keras is returning an empty history dict.
        try:
            # Save loss/val metrics
            epoch_loss_history += history.history['loss']
            epoch_val_history += history.history['val_loss']
            epoch_lr_history += lr_decay.history_lr
        except:
            pass
    time_total = time.time() - start
    info_path = os.path.join(MODEL_DIR, 'information_training.txt')  
    with open(info_path, 'w') as f:
        f.write("The previous best weights : " + checkpoint_path + "\n")
        f.write("Epochs that used : " + str(num_epochs) + "\n")
        f.write("Model that used :" + str(inp_model) + "\n")
        f.write("Batch Size that used :" + str(batch_size) + "\n")
        f.write("Total time training :" + str(int(time_total)) + "seconds\n")
    print("The previous best weights: ", checkpoint_path)
    model.load_weights(checkpoint_path)
    print("Total time: ", int(time_total), "seconds")
    #print("Loading model: ", checkpoint_path)
    print("-------------------------Finish----------------------")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train model of GRU and CNN+GRU"
    )
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        default="single_gru",
        help="choose model what you want train, gru or cnn_gru",
    )
    parser.add_argument(
        "--input",
        required=True,
        metavar="~/CMAPSSData/",
        help="Full input path of dataset",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="~/model/",
        help="Full output path of dataset",
    )
    parser.add_argument(
        "--epochs",
        required=True,
        type=int,
        default="50",
        help="number of epochs for training",
    )

    args = parser.parse_args()
    DATA_DIR = args.input
    MODEL_DIR = args.output

    path = os.path.join(DATA_DIR, "train_FD*.txt")
    persist_run_stats = True # Enable for saving results to CouchDB
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    path = os.path.join(DATA_DIR, "train_FD*.txt")
    all_files = glob.glob(path)

    train_df = load_data(all_files, cols, sort_cols)

    train_df = calc_training_rul(train_df)

    #Data Transformation
    pipeline = Pipeline(steps=[
        # The default activation function for LSTM tanh, so we'll use a range of [-1,1].
        ('min_max_scaler', preprocessing.MinMaxScaler(feature_range=(-1, 1)))
    ])

    # Set up the columns that will be scaled
    train_df['cycle_norm'] = train_df['cycle']

    # Transform all columns except id, cycle, and RUL
    cols_transform = train_df.columns.difference(['id','cycle', 'RUL'])

    xform_train_df = pd.DataFrame(pipeline.fit_transform(train_df[cols_transform]), 
                                columns=cols_transform, 
                                index=train_df.index)
    join_df = train_df[train_df.columns.difference(cols_transform)].join(xform_train_df)
    train_df = join_df.reindex(columns = train_df.columns)

    # Build the feature column list 
    feature_cols = ['cycle_norm', 'condition']

    # Three operational setting columns
    setting_cols = ['setting' + str(i) for i in range(1,4)]
    feature_cols.extend(setting_cols)

    # Twenty one sensor columns
    sensor_cols = ['s' + str(i) for i in range(1,22)]
    feature_cols.extend(sensor_cols)

    # Build the label column list
    label_cols = ['RUL']

    # Size of the series time window.
    sequence_length = 25

    # Number of time series sequences that will be train on per batch.
    batch_size = 512

    num_features = len(feature_cols)
    num_labels = len(label_cols)


    # Setup log directory
    log_dir, checkpoint_path = set_log_dir(MODEL_DIR, "engine")

    print("Log dir: ", log_dir)
    print("Checkpoint path: ", checkpoint_path)

    # Save the pipeline for later use
    pipeline_path = os.path.join(log_dir, 'engine_pipeline.pkl') 
    joblib.dump(pipeline, pipeline_path) 

    train_model(args.model, args.epochs)
