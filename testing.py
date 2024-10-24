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
from tqdm.notebook import tqdm

from data_generator import TSDataGenerator, split_data, create_generators
from util import set_log_dir, rmse, r2_keras, upload_to_drive, find_or_create_folder
from util import LRDecay
from data_util import *
from model import *
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
import datetime

#Load Dataset from folder
cols = ['id', 'cycle' ]

# Three operational setting columns
setting_cols = ['setting' + str(i) for i in range(1,4)]
cols.extend(setting_cols)

# Twenty one sensor columns
sensor_cols = ['s' + str(i) for i in range(1,22)]
cols.extend(sensor_cols)

sort_cols = ['id','cycle']

# # Google Drive setup
# SERVICE_ACCOUNT_FILE = os.getenv('SERVICE_ACCOUNT_FILE')
# # Define the folder name and path where the model will be saved
# folder_name = f"engine_test_{datetime.datetime.now().strftime('%Y%m%dT%H%M')}"
# parent_folder_id = os.getenv('FOLDER_ID')  # Get folder ID from the environment variable

# SCOPES = ['https://www.googleapis.com/auth/drive.file']
# creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
# drive_service = build('drive', 'v3', credentials=creds)
# # Create or find the folder
# folder_id = find_or_create_folder(drive_service, folder_name, parent_folder_id)

def load_rul_data(paths, col_names):
    
    # Filename is used to determine the condition
    col_names.append('filename')

    # read data 
    df = pd.DataFrame()
    for p in paths:
        instance_df = pd.read_csv(p, sep=" ", header=None)
        instance_df.drop(instance_df.columns[[1]], axis=1, inplace=True)
        instance_df['filename'] = os.path.splitext(os.path.basename(p))[0]
        instance_df = instance_df.reset_index()
        instance_df.columns = col_names
        
        df = pd.concat((df, instance_df), sort=False) 

    df['id'] = df['id'] + df['filename'].apply( lambda f: fn_id_map[f]) + 1
    df.drop(['filename'], axis=1, inplace=True)
    return df

def calc_test_rul( feature_df, label_df):
    # If index is not reset there will be int/str type issues when attempting the merge. 
    cycle_count_df = feature_df.groupby('id').count().reset_index()[['id','cycle']].rename(index=str, columns={"cycle":"cycles"}).reset_index(drop=True)
    print(cycle_count_df.shape)

    # Join cycle and RUL dataframes
    assert cycle_count_df.shape[0] == label_df.shape[0]
    tmp_df = cycle_count_df.merge(label_df, on="id", how='left')

    # The RUL actual column contains the value for the last cycle.
    # Adding the cycles column will give us the RUL for the first cycle.
    tmp_df['RUL_actual'] = tmp_df['cycles'] + tmp_df['RUL_actual']
    tmp_df.drop('cycles',  axis=1, inplace=True)

    # Join the two data frames
    feature_df = feature_df.merge(tmp_df, on='id', how='left')


    # Use the cycle to decrement the RUL until the ground truth is reached.
    feature_df['RUL'] = feature_df['RUL_actual'] - feature_df['cycle']
    feature_df.drop('RUL_actual',  axis=1, inplace=True)
    
    return feature_df

def plot_prediction(rul_actual, rul_predicted):  
    fig = plt.figure(figsize=(25,5))
    cycles = np.arange(len(rul_actual))
    plt.scatter(cycles, rul_predicted, marker='.', label="Predicted")
    plt.plot(cycles, rul_actual, 'r', label="Actual")
    plt.xlabel("Cycle")
    plt.xlim(0)
    plt.ylabel("RUL")
    plt.ylim(0)

    plt.legend()
    plot_filename = "plot_prediction.png"
    plt.savefig(plot_filename)
    # upload_to_drive(plot_filename, folder_id, drive_service)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Testing model of GRU and CNN+GRU"
    )
    parser.add_argument(
        "--dataset",
        required=False,
        metavar="./dataset/",
        default="./dataset/",
        help="Full input path of dataset",
    )
    parser.add_argument(
        "--input_model_weight",
        required=False,
        metavar="./model/",
        default="./model/",
        help="Full output path of dataset",
    )
    parser.add_argument(
        "--dataset_name",
        required=False,
        metavar="FD002",
        default="FD002",
        help="Full input path of dataset",
    )

    args = parser.parse_args()
    DATA_DIR = os.path.abspath(args.dataset)
    MODEL_DIR = os.path.abspath(args.input_model_weight)
    persist_run_stats = True # Enable for saving results to CouchDB

    #load test dataset
    dataset_name = args.dataset_name
    dataset_name = str(dataset_name)


    # Setup log directory
    checkpoint_path = sorted(glob.glob(MODEL_DIR+"/*/*.keras"), reverse=True)[0]

    print("Checkpoint path: ", checkpoint_path)
    #Data Transformation
    pipeline = Pipeline(steps=[
        # The default activation function for LSTM tanh, so we'll use a range of [-1,1].
        ('min_max_scaler', preprocessing.MinMaxScaler(feature_range=(-1, 1)))
    ])

    # Save the pipeline for later use
    import joblib 
    pipeline_path = sorted(glob.glob(MODEL_DIR+"/*/*.pkl"), reverse=True)[0]
    joblib.dump(pipeline, pipeline_path) 

    print("Loading model: ", checkpoint_path)
    custom_objects={'rmse':rmse, 'r2_keras':r2_keras}
    inf_model = load_model(checkpoint_path, custom_objects=custom_objects)

#    print(inf_model.summary())
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


    test_X_path = os.path.join(DATA_DIR, 'test_' + dataset_name + '.txt')
    test_y_path = os.path.join(DATA_DIR, 'RUL_' + dataset_name + '.txt')

    # Read in the features
    test_df = load_data([test_X_path], cols, sort_cols)

    # Read in the labels (RUL)
    test_rul_df = load_rul_data([test_y_path], ['id', 'RUL_actual'])

    # Calculate the RUL and merge back to the test dataframe
    test_df = calc_test_rul(test_df, test_rul_df)
#    print(test_df.head())

    #transform dataset
    test_df['cycle_norm'] = test_df['cycle']

    norm_test_df = pd.DataFrame(pipeline.transform(test_df[cols_transform]), 
                                columns=cols_transform, 
                                index=test_df.index)
    test_join_df = test_df[test_df.columns.difference(cols_transform)].join(norm_test_df)
    test_df = test_join_df.reindex(columns = test_df.columns)
    test_df = test_df.reset_index(drop=True)

#    print(test_df.head()[feature_cols])
#    print(test_df.head()[label_cols])

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
    # Evaluation metrics are RMSE, MSE, MAE
    score = inf_model.evaluate(test_X, test_y, verbose=1, batch_size=batch_size)
    print('Test score:\n\tRMSE: {}\n\tMSE: {}\n\tR2: {}'.format(*score))

    # Saving scores to a text file
    model_name = os.path.basename(checkpoint_path)
    test_result_filename = 'test_result_{}.txt'.format(model_name)
    with open(test_result_filename, 'w') as file:
        file.write('Test score:\n')
        file.write('\tRMSE: {}\n'.format(score[0]))
        file.write('\tMSE: {}\n'.format(score[1]))
        file.write('\tR2: {}\n'.format(score[2]))

    # upload_to_drive(test_result_filename, folder_id, drive_service)

    test_data_generator = TSDataGenerator(test_df, feature_cols, label_cols, batch_size=batch_size, seq_length=sequence_length, loop=False)

    g = test_data_generator.generate()
    test_X, test_y = next(g)
    y_pred_array = inf_model.predict_on_batch(test_X)
    plot_prediction(test_y, y_pred_array)

    # Evaluation metrics are RMSE, MSE, MAE
    score = inf_model.evaluate(test_X, test_y, verbose=1, batch_size=batch_size)
    print('Test score:\n\tRMSE: {}\n\tMSE: {}\n\tR2: {}'.format(*score))