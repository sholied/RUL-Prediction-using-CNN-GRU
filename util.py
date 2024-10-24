import os
import sys
import datetime
import errno
import glob
import numpy as np
from keras import backend as K
import tensorflow as tf
from googleapiclient.http import MediaFileUpload


# Check if the folder exists
def create_folder(service, folder_name, parent_folder_id):
    folder_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder',
        'parents': [parent_folder_id]  # Specify the parent folder ID
    }
    folder = service.files().create(body=folder_metadata, fields='id').execute()
    return folder.get('id')

# Try to find the folder first; if not found, create it
def find_or_create_folder(service, folder_name, parent_folder_id):
    query = f"'{parent_folder_id}' in parents and name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
    results = service.files().list(q=query, fields='files(id, name)').execute()
    folders = results.get('files', [])
    
    if folders:
        # Folder exists, return the ID
        return folders[0].get('id')
    else:
        # Folder does not exist, create it
        return create_folder(service, folder_name, parent_folder_id)

def upload_to_drive(file_name, folder_id, service):
    """Upload a file to Google Drive using resumable upload."""
    file_metadata = {'name': os.path.basename(file_name), 'parents': [folder_id]}
    
    # Open file and use MediaIoBaseUpload for resumable upload
    try:
        media = MediaFileUpload(file_name, mimetype='application/octet-stream', resumable=True)
        request = service.files().create(body=file_metadata, media_body=media, fields='id')
        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                print(f"Upload progress: {int(status.progress() * 100)}%")
        print(f"Uploaded {file_name} to Google Drive with file ID: {response.get('id')}")
    except Exception as e:
        print(f"An error occurred while uploading {file_name}: {e}")

def set_log_dir(model_dir, name, per_epoch=False, val_loss=False, create_dir=True):
    # Directory for training logs
    now = datetime.datetime.now()
    now_str = "{:%Y%m%dT%H%M}".format(now)
    log_dir = os.path.join(model_dir, "{}{}".format(name.lower(), now_str))

    # Create log_dir if not exists
    if not os.path.exists(log_dir):
        if create_dir:
            os.makedirs(log_dir)
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), log_dir)

    # Path to save after each epoch. Include epoch and val loss
    checkpoint_path = os.path.join(log_dir, "{}_model".format(name.lower()))

    if val_loss:
        checkpoint_path += "_{val_loss:.2f}"  # Include val_loss in filename
        per_epoch = True

    if per_epoch:
        checkpoint_path += "_{epoch:04d}"  # Include epoch in filename

    checkpoint_path += ".keras"  # Ensure the new format is used

    return log_dir, checkpoint_path


def find_model_file(model_dir, by_val_loss=True):
    # file names are expected in format: <name>_model_<timestamp>_<epoch>_<val_loss>.keras
    # _<epoch> and _<val_loss> are optional
    
    path = os.path.join(model_dir, "*.keras")
    all_model_paths = sorted(glob.glob(path))

    if by_val_loss:
        model_path = all_model_paths[0]
        val_loss = float(sys.maxsize)
        for path in all_model_paths:
            filename = os.path.basename(path)
            file = os.path.splitext(filename)[0]
            file_val_loss = file.split('_')[-1]
            if val_loss > float(file_val_loss):
                val_loss = float(file_val_loss)
                model_path = path

    else:
        model_path = all_model_paths[-1]

    return model_path


# Root Mean Squared Loss Function
def rmse(y_true, y_pred):
    #tf.cast(y_pred, tf.int64)
    #tf.cast(y_true, tf.float32)
    return K.sqrt(K.mean(K.square(float(y_pred) - float(y_true))))

def r2_keras(y_true, y_pred):
    """Coefficient of Determination 
    """
    SS_res =  K.sum(K.square( y_true - y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


class LRDecay:

    def __init__(self, initial_lrate=1e-3, decay_multiple=0.5, epochs_step=25, patience_multiple=3):
        self.initial_lrate = initial_lrate
        self.decay_multiple = decay_multiple
        self.epochs_step = epochs_step
        self.patience_multiple = patience_multiple
        self.patience = self.epochs_step*self.patience_multiple
        self.history_lr = []
        self.r_epoch = 0

    def linear_decay(self, x):
        return self.decay_multiple - (x/1500)

    def reset(self, r_epoch=0):
        self.r_epoch = r_epoch

    # learning rate schedule
    def step_decay(self, epoch, current_lr):
        lrate = current_lr
    
        if self.r_epoch == 0:
            lrate = self.initial_lrate
    
        elif (1+self.r_epoch) % self.epochs_step == 0:
            lrate = current_lr * self.linear_decay(self.r_epoch)
    
        lrate = np.around(lrate, 8)

        # Use the actual epoch to track history rather then
        # this runs epoch
        self.history_lr.append((epoch, current_lr, lrate))

        self.r_epoch += 1
    
        return lrate
