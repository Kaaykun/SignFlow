import cv2
import shutil
import sys
import os

from params import FRAMES_PER_VIDEO, TARGET_SIZE, TRAIN_SIZE, N_CLASSES, NUMBER_OF_AUGMENTATIONS


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
