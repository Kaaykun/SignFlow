import cv2
import shutil
import sys
import os
import time
import glob
from tensorflow import keras
from google.cloud import storage

from backend.params import FRAMES_PER_VIDEO, TARGET_SIZE, TRAIN_SIZE, N_CLASSES, NUMBER_OF_AUGMENTATIONS
from backend.params import LOCAL_REGISTRY_PATH


def generate_processed_videos(X):
    """
    Generate processed videos from sampled frames.

    Parameters:
    - X (numpy.ndarray): Array containing sampled frames for multiple videos.
    - output_folder (str): Path to the folder to store processed videos. Defaults to '../data/processed_videos/'.

    Returns:
    - None
    """
    def frames_to_video(sampled_frames, output_path, fps=FRAMES_PER_VIDEO):
        height, width, _ = sampled_frames[0].shape
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in sampled_frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video.write(frame_rgb)

        video.release()

    output_folder = os.path.dirname('../data/processed_videos/')

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    os.makedirs(output_folder)

    for i, sampled_frames in enumerate(X):
            video_path = f'../data/processed_videos/processed_{i}.mp4'
            frames_to_video(sampled_frames, video_path)


def generate_csv(list_of_dataframes):
    """
    Generate CSV files from a list of DataFrames.

    Parameters:
    - list_of_dataframes (list): List containing pandas DataFrames.

    Returns:
    - None
    """
    output_folder = os.path.dirname('../data/csv/')

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    os.makedirs(output_folder)

    for i, dataframe in enumerate(list_of_dataframes):
        file_path = f'{output_folder}/dataframe_{i}.csv'
        dataframe.to_csv(file_path, index=False)


def test_shape_X_y(selected_df, input_length, X, y_cat):
    if X.shape == (len(selected_df), FRAMES_PER_VIDEO, *TARGET_SIZE, 3):
        print(f'✅ X has been initialized with Shape {X.shape}!')
    else:
        sys.exit('❌ X has not been initialized properly!')

    if y_cat.shape == (input_length, N_CLASSES):
        print(f'✅ y has been initialized with Shape {y_cat.shape}!')
    else:
        sys.exit('❌ y has not been initialized properly!')


def test_shape_X_y_val(X, X_val, y_cat, y_cat_val):
    if X_val.shape == (round(len(X) * (1 - TRAIN_SIZE)), FRAMES_PER_VIDEO, *TARGET_SIZE, 3):
        print(f'\n✅ X_val has been initialized with Shape {X_val.shape}!')
    else:
        sys.exit('\n❌ X_val has not been initialized properly!')

    if y_cat_val.shape == (round(len(y_cat) * (1 - TRAIN_SIZE)), N_CLASSES):
        print(f'✅ y_cat_val has been initialized with Shape {y_cat_val.shape}!')
    else:
        sys.exit('❌ y_cat_val has not been initialized properly!')


def test_shape_X_y_aug(X_aug, X_train, y_aug, y_cat_train):
    if X_aug.shape == ((NUMBER_OF_AUGMENTATIONS + 1) * len(X_train), FRAMES_PER_VIDEO, *TARGET_SIZE, 3):
        print(f'\n✅ X_aug has been initialized with Shape {X_aug.shape}!')
    else:
        sys.exit('\n❌ X_aug has not been initialized properly!')

    if y_aug.shape == ((NUMBER_OF_AUGMENTATIONS + 1) * len(y_cat_train), N_CLASSES):
        print(f'✅ y_aug has been initialized with Shape {y_aug.shape}!')
    else:
        sys.exit('❌ y_aug has not been initialized properly!')


def save_model(model):
    """
    Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}{timestamp}.h5"
    - if MODEL_TARGET='gcs', also persist it in your bucket on GCS at "models/{timestamp}.h5"
    - if MODEL_TARGET='mlflow', also persist it on MLflow instead of GCS
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    model_path = os.path.join(LOCAL_REGISTRY_PATH, f"{timestamp}.h5")
    model.save(model_path)
    print(model_path, 'updated function 2')
    print(model_path.split("/")[-1])
    print("✅ Model saved locally")

    # if MODEL_TARGET == "gcs":
    #     print('successfully entered function')
    #     model_filename = model_path.split("/")[-1]
    #     print(f'model filename = {model_filename}')
    #     client = storage.Client()
    #     print(f'client = {client}')
    #     bucket = client.bucket(BUCKET_NAME)
    #     print(f'bucket = {bucket}')
    #     blob = bucket.blob(f'models/{model_filename}')
    #     print(f'blob = {blob}')
    #     blob.upload_from_filename(model_path)

    #     print("✅ Model saved to GCS")

    #     return None

    return None


def load_model():
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from GCS (most recent one) if MODEL_TARGET=='gcs'  --> for unit 02 only
    - or from MLFLOW (by "stage") if MODEL_TARGET=='mlflow' --> for unit 03 only

    Return None (but do not Raise) if no model is found

    """
    # Get the latest model version name by the timestamp on disk
    local_model_directory = os.path.join(LOCAL_REGISTRY_PATH)
    local_model_paths = glob.glob(f"{local_model_directory}/*")

    if not local_model_paths:
        return None

    most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

    latest_model = keras.models.load_model(most_recent_model_path_on_disk)

    print("✅ Model loaded from local disk")

    return latest_model
