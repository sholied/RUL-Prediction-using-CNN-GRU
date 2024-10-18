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

import sys
sys.path.insert(0,'../')

from data_generator import TSDataGenerator, split_data, create_generators
from util import set_log_dir, rmse, r2_keras, upload_to_drive, find_or_create_folder
from util import LRDecay
from data_util import *
from model import *
import datetime
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

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

# Google Drive setup
SERVICE_ACCOUNT_FILE = os.getenv('SERVICE_ACCOUNT_FILE')
# Define the folder name and path where the model will be saved
folder_name = f"engine_{datetime.datetime.now().strftime('%Y%m%dT%H%M')}"
parent_folder_id = os.getenv('FOLDER_ID')  # Get folder ID from the environment variable

SCOPES = ['https://www.googleapis.com/auth/drive.file']
creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=creds)
# Create or find the folder
folder_id = find_or_create_folder(drive_service, folder_name, parent_folder_id)

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

def countdown(seconds):
    while seconds > 0:
        print(f"Waiting for {seconds} seconds...", end='\r')
        time.sleep(1)
        seconds -= 1
    print() 

#Calculate Training DATA RUL
def calc_training_rul(df):
    # Data Labeling - generate column RUL
    rul = pd.DataFrame(df.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'max']
    df = df.merge(rul, on=['id'], how='left')
    df['RUL'] = df['max'] - df['cycle']
    df.drop('max', axis=1, inplace=True)
    return df

def train_model(inp_model, num_epochs, num_cnn=0, num_gru=0):
    # Create the model
    if inp_model in ["cnn_gru", "cnn_lstm"]:
        plot_filename = "plot_train_{}_{}_{}.png".format(inp_model, num_cnn, num_gru)
        results_filename = 'results_train_{}_{}_{}.txt'.format(inp_model, num_cnn, num_gru)
        model = model_cnngru(num_cnn, num_gru, sequence_length, num_features, num_labels) if inp_model == "cnn_gru" else model_cnnlstm(num_cnn, num_gru, sequence_length, num_features, num_labels)
    elif inp_model in ["single_gru", "single_lstm"]:
        plot_filename = "plot_train_{}_{}.png".format(inp_model, num_gru)
        results_filename = 'results_train_{}_{}.txt'.format(inp_model, num_gru)
        model = model_gru(num_gru, sequence_length, num_features, num_labels) if inp_model == "single_gru" else model_lstm(num_gru, sequence_length, num_features, num_labels)
    else:
        plot_filename = "plot_train_{}_{}_{}.png".format(inp_model, num_cnn, num_gru)
        results_filename = 'results_train_{}_{}_{}.txt'.format(inp_model, num_cnn, num_gru)
        print("-------------------------------------------------------------------------\n")
        print("model incorrect, please choose one: single_gru, cnn_gru, single_lstm, or cnn_lstm!\n")
        print("-------------------------------------------------------------------------\n")
        return

    model.compile(loss=rmse, optimizer='rmsprop', metrics=['mse', r2_keras])

    print(model.summary())

    start = time.time()
    training_runs = 4
    initial_epoch = 0
    epochs_before_decay = 15
    lrate = 1e-3
    patience = 100

    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False)
    checkpointer = ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=True)

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

        callbacks = [tensorboard, checkpointer, lr_scheduler, earlystopper]

        history = model.fit(
            train_data_generator.generate(), 
            validation_data=val_data_generator.generate(), 
            initial_epoch=initial_epoch,
            epochs=num_epochs, 
            steps_per_epoch=train_data_generator.summary()['max_iterations'],
            validation_steps=val_data_generator.summary()['max_iterations'],
            shuffle=False,
            verbose=0,
            callbacks=callbacks)

        if len(history.epoch) > 0:        
            initial_epoch = history.epoch[-1] + 1

        # TODO fix, sometimes Keras is returning an empty history dict.
        try:
            epoch_loss_history += history.history['loss']
            epoch_val_history += history.history['val_loss']
            epoch_lr_history += lr_decay.history_lr
        except:
            pass
    
    time_total = time.time() - start

    fig, ax1 = plt.subplots(figsize=(10,6))
    epochs = np.arange(len(epoch_loss_history))
    ax1.plot(epochs, epoch_loss_history, label='loss')
    ax1.plot(epochs, epoch_val_history, label='val_loss')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("RMSE")
    ax1.set_ylim(0)
    ax1.legend()

    ax2 = ax1.twinx()
    lr_epochs = []
    lr_steps = []
    for e1, l1, l2 in epoch_lr_history:
        lr_epochs.append(e1)
        lr_steps.append(l1)

    ax2.plot(lr_epochs, lr_steps, 'r.', label='lr')
    ax2.set_ylabel("Learning Rate")
    ax2.legend(loc='lower left')

    fig.tight_layout()
    plt.savefig(plot_filename)

    with open(results_filename, 'w') as f:
        f.write("The previous best weights : " + checkpoint_path + "\n")
        f.write("Epochs that used : " + str(num_epochs) + "\n")
        f.write("Model that used :" + str(inp_model) + "\n")
        f.write("Batch Size that used :" + str(batch_size) + "\n")
        f.write("Total time training :" + str(int(time_total)) + "seconds\n")
    print("The previous best weights: ", checkpoint_path)
    model.load_weights(checkpoint_path)
    print("Total time: ", int(time_total), " seconds")
    print("-------------------------Finish Training----------------------")

    # Uploading files to Google Drive
    upload_to_drive(checkpoint_path, folder_id, drive_service)  # Upload the best model weights
    upload_to_drive(plot_filename, folder_id, drive_service)  # Upload the loss/val_loss plot
    upload_to_drive(results_filename, folder_id, drive_service)  # Upload the training result summary

    # Testing phase
    print("Starting model testing...")
    list_dataset = ['FD001', 'FD002', 'FD003', 'FD004']

    # Create a single results file for all datasets
    compare_result_filename = 'compare_train_results_{}_cnn{}_gru{}.txt'.format(inp_model, num_cnn, num_gru)
    with open(compare_result_filename, 'w') as file:
        file.write('Comparison of Test Scores for model {} with CNN layers {} and GRU layers {}:\n'.format(inp_model, num_cnn, num_gru))
        file.write('=================================================================\n')

        for dataset_name in list_dataset:
            print("Starting testing dataset test{}.txt".format(dataset_name))

            test_X_path = os.path.join(DATA_DIR, 'test_' + dataset_name + '.txt')
            test_y_path = os.path.join(DATA_DIR, 'RUL_' + dataset_name + '.txt')

            # Read in the features
            test_df = load_data([test_X_path], cols, sort_cols)

            # Read in the labels (RUL)
            test_rul_df = load_rul_data([test_y_path], ['id', 'RUL_actual'])

            # Calculate the RUL and merge back to the test dataframe
            test_df = calc_test_rul(test_df, test_rul_df)

            # Transform dataset
            test_df['cycle_norm'] = test_df['cycle']

            norm_test_df = pd.DataFrame(pipeline.transform(test_df[cols_transform]), 
                                        columns=cols_transform, 
                                        index=test_df.index)
            test_join_df = test_df[test_df.columns.difference(cols_transform)].join(norm_test_df)
            test_df = test_join_df.reindex(columns=test_df.columns)
            test_df = test_df.reset_index(drop=True)

            test_data_generator = TSDataGenerator(test_df, 
                                                  feature_cols, 
                                                  label_cols,
                                                  batch_size=batch_size,
                                                  seq_length=sequence_length, 
                                                  randomize=False,
                                                  loop=False)

            print("summaryyyy--------------------")
            print(test_data_generator.print_summary())

            X = []
            y = []
            for p in tqdm(test_data_generator.generate(), total=test_data_generator.summary()['max_iterations']):
                X.append(p[0])
                y.append(p[1])

            test_X = np.vstack(X)
            test_y = np.vstack(y)

            # Evaluate the model
            score = model.evaluate(test_X, test_y, verbose=1, batch_size=batch_size)
            print('DATASET :: test{}.txt :: Test score for model {} with layer GRU {} and CNN {}:\n\tRMSE: {}\n\tMSE: {}\n\tR2: {}'.format(dataset_name, inp_model, num_gru, num_cnn, *score))

            # Append results to the comparison file
            file.write('DATASET :: test{}.txt Test score for model {} with layer GRU {} and CNN {}:\n'.format(dataset_name, inp_model, num_gru, num_cnn))
            file.write('-------------------------------------------------------------\n')
            file.write('\tRMSE: {}\n'.format(score[0]))
            file.write('\tMSE: {}\n'.format(score[1]))
            file.write('\tR2: {}\n\n'.format(score[2]))

        # # Optional: You can also upload intermediate results per dataset if needed
        # test_result_filename = 'evaluate_train_result_{}_{}_cnn{}_gru{}.txt'.format(dataset_name, inp_model, num_cnn, num_gru)
        # with open(test_result_filename, 'w') as individual_file:
        #     individual_file.write('DATASET :: test{}.txt Test score for model {} with layer GRU {} and CNN {}:\n'.format(dataset_name, inp_model, num_gru, num_cnn))
        #     individual_file.write('-------------------------------------------------------------\n')
        #     individual_file.write('\tRMSE: {}\n'.format(score[0]))
        #     individual_file.write('\tMSE: {}\n'.format(score[1]))
        #     individual_file.write('\tR2: {}\n'.format(score[2]))

        # # Upload individual test results
        # upload_to_drive(test_result_filename, folder_id, drive_service)

    # Upload the overall comparison file to the drive
    upload_to_drive(compare_result_filename, folder_id, drive_service)

    print("-------------------------Finish Testing----------------------")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train model of GRU and CNN+GRU"
    )
    parser.add_argument(
        "--model",
        required=False,
        type=str,
        default="single_gru",
        help="choose model what you want train, gru or cnn_gru",
    )
    parser.add_argument(
        "--input",
        required=False,
        metavar="./dataset/",
        default="./dataset/",
        help="Full input path of dataset",
    )
    parser.add_argument(
        "--output",
        required=False,
        metavar="./model/",
        default="./model/",
        help="Full output path of dataset",
    )
    parser.add_argument(
        "--epochs",
        required=False,
        type=int,
        default="1",
        help="number of epochs for training",
    )

    args = parser.parse_args()
    DATA_DIR = os.path.abspath(args.input)
    MODEL_DIR = os.path.abspath(args.output)
    persist_run_stats = True # Enable for saving results to CouchDB
    path = os.path.join(DATA_DIR, "train_FD*.txt")    
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
    batch_size = 256

    num_features = len(feature_cols)
    num_labels = len(label_cols)

    # num_cnn = 3
    # num_gru = 1


    print("========================START TRAINING MODEL===========================")
    # for num_gru in range(2, 5):
    for num_gru in range(1, 2):
        args.model = 'single_gru'
        print("PROCESS NUMBER LAYER GRU : ", num_gru)
        # Setup log directory
        log_dir, checkpoint_path = set_log_dir(MODEL_DIR, "engine_{}_{}".format(args.model, num_gru))

        print("Log dir: ", log_dir)
        print("Checkpoint path: ", checkpoint_path)

        # Save the pipeline for later use
        pipeline_path = os.path.join(log_dir, 'engine_pipeline.pkl') 
        joblib.dump(pipeline, pipeline_path) 

        train_model(args.model, args.epochs, num_gru=num_gru)
        time.sleep(1)

    # print("Wait before starting the next 2nd loop...")
    # countdown(5)

    # for num_lstm in range(2, 5):
    #     args.model = 'single_lstm'
    #     print("PROCESS NUMBER LAYER LSTM : ", num_lstm)
    #     # Setup log directory
    #     log_dir, checkpoint_path = set_log_dir(MODEL_DIR, "engine_{}_{}".format(args.model, num_lstm))

    #     print("Log dir: ", log_dir)
    #     print("Checkpoint path: ", checkpoint_path)

    #     # Save the pipeline for later use
    #     pipeline_path = os.path.join(log_dir, 'engine_pipeline.pkl') 
    #     joblib.dump(pipeline, pipeline_path) 

    #     train_model(args.model, args.epochs, num_gru=num_lstm)
    #     time.sleep(1)

    # print("Wait before starting the next 3rd loop...")
    # countdown(5)

    # for num_cnn in range(1, 4):
    #     # Iterate through GRU layers (2 to 3)
    #     for num_gru in range(1, 4):
    #         args.model = 'cnn_gru'
    #         print("PROCESS NUMBER LAYER CNN : {} AND LAYER GRU : {}".format(num_cnn, num_gru))
    #         # Setup log directory
    #         log_dir, checkpoint_path = set_log_dir(MODEL_DIR, "engine_{}_{}_{}".format(args.model, num_cnn, num_gru))

    #         print("Log dir: ", log_dir)
    #         print("Checkpoint path: ", checkpoint_path)

    #         # Save the pipeline for later use
    #         pipeline_path = os.path.join(log_dir, 'engine_pipeline.pkl') 
    #         joblib.dump(pipeline, pipeline_path) 

    #         train_model(args.model, args.epochs,num_cnn=num_cnn, num_gru=num_gru)
    #         time.sleep(1)

    # print("Wait before starting the next 4th loop...")
    # countdown(5)

    # for num_cnn in range(1, 4):
    #     # Iterate through GRU layers (2 to 3)
    #     for num_lstm in range(1, 4):
    #         args.model = 'cnn_lstm'
    #         print("PROCESS NUMBER LAYER CNN : {} AND LAYER LSTM : {}".format(num_cnn, num_lstm))
    #         # Setup log directory
    #         log_dir, checkpoint_path = set_log_dir(MODEL_DIR, "engine_{}_{}_{}".format(args.model, num_cnn, num_lstm))

    #         print("Log dir: ", log_dir)
    #         print("Checkpoint path: ", checkpoint_path)

    #         # Save the pipeline for later use
    #         pipeline_path = os.path.join(log_dir, 'engine_pipeline.pkl') 
    #         joblib.dump(pipeline, pipeline_path) 

    #         train_model(args.model, args.epochs,num_cnn=num_cnn, num_gru=num_lstm)
    #         time.sleep(1)