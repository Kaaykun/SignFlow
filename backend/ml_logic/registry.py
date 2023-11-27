import os
# Suppress WARNING, INFO, and DEBUG messages related to tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import shutil
import sys
import os
import time
import glob
import mediapipe as mp
from tensorflow import keras

from backend.params import FRAMES_PER_VIDEO, TARGET_SIZE, TRAIN_SIZE, N_CLASSES, NUMBER_OF_AUGMENTATIONS
from backend.params import LOCAL_REGISTRY_PATH, CUSTOM_VIDEO_PATH


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


def draw_landmarks(results, frame):
    """
    Draw landmarks and connections for hands and pose on an image frame.

    Parameters:
    - results (dict): Results obtained from holistic processing.
    - frame (numpy.ndarray): Image frame.

    Returns:
    - numpy.ndarray: Annotated image with landmarks and connections drawn.
    """
    mp_holistic = mp.solutions.holistic # type: ignore
    mp_drawing = mp.solutions.drawing_utils # type: ignore
    annotated_image = frame.copy()

    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=results.right_hand_landmarks,
            connections=mp_holistic.HAND_CONNECTIONS)
        print("✅ Right hand annotated")
    else:
        print("❌ Right hand not annotated")

    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=results.left_hand_landmarks,
            connections=mp_holistic.HAND_CONNECTIONS)
        print("✅ Left hand annotated")
    else:
        print("❌ Left hand not annotated")


    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=results.pose_landmarks,
            connections=mp_holistic.POSE_CONNECTIONS)
        print("✅ Pose annotated")
    else:
        print("❌ Pose not annotated")

    return annotated_image


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


def record_videos(word, name, video_duration, num_videos):
    """
    Record multiple videos using the default camera for a specified duration.

    Parameters:
    - word (str): Word or label for the recorded videos.
    - video_duration (float): Duration of each recorded video in seconds.
    - num_videos (int): Number of videos to record.
    - custom_video_path (str): Path to store the recorded videos. Defaults to '../data/custom_videos/'.

    Returns:
    - None
    """
    output_folder = os.path.dirname(f'{CUSTOM_VIDEO_PATH}')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for video_num in range(1, num_videos + 1):
        output_file = f"{word}_{name}_{video_num}.mp4"

        # Open a video capture stream (use 0 for default camera)
        cap = cv2.VideoCapture(0)

        # Set video resolution (optional)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 360)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

        # Get the frames per second (fps) of the video capture stream
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Calculate the number of frames needed to capture for the specified duration
        num_frames_to_capture = int(fps * video_duration)

        # Create a VideoWriter object to save the video
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

        # Record video for the specified durations
        start_time = time.time()
        frame_count = 0

        while frame_count < num_frames_to_capture:
            ret, frame = cap.read()

            if not ret:
                break

            # Write the frame to the output video file
            out.write(frame)

            frame_count += 1

        # Release the video capture and writer objects
        cap.release()
        out.release()

        # Print information about the recorded video
        elapsed_time = time.time() - start_time
        print(f"Video recorded: {output_file}")
        print(f"Duration: {elapsed_time:.2f} seconds")
        print(f"Number of frames: {frame_count}")
        print(f"Frames per second: {fps}")
        print()


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
