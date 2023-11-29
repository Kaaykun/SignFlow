import os
# Suppress WARNING, INFO, and DEBUG messages related to tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import mediapipe as mp
import cv2
import time
import tensorflow as tf
from tensorflow import keras
from keras import Sequential, layers
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from backend.params import FRAMES_PER_VIDEO, N_CLASSES


def detect_landmarks(frame):
    """
    Performs holistic detection on a frame using MediaPipe's Holistic model.

    Parameters:
    - frame (numpy.ndarray): The frame to be processed (in BGR format).

    Returns:
    - dict: Results obtained from holistic processing containing landmarks, poses, etc.
    """
    mp_holistic = mp.solutions.holistic # type: ignore

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, static_image_mode=False) as holistic:
        results = holistic.process(frame_rgb)
    return results


def get_coord_for_R_hand(results):
    """
    Extracts x, y, and z coordinates for the right hand landmarks from the results.

    Parameters:
    - results (dict): Results obtained from holistic processing.

    Returns:
    - tuple: Three lists containing x, y, and z coordinates for the right hand landmarks.
    """
    x_coord_R_hand = []
    y_coord_R_hand = []
    z_coord_R_hand = []

    if results.right_hand_landmarks:
        # Append coordinates if landmarks are detected
        for landmark in results.right_hand_landmarks.landmark:
            if landmark:
                x_coord_R_hand.append(landmark.x)
                y_coord_R_hand.append(landmark.y)
                z_coord_R_hand.append(landmark.z)
            else:
                x_coord_R_hand.append(0)
                y_coord_R_hand.append(0)
                z_coord_R_hand.append(0)
        #print("✅ Right hand detected")

    else:
        for _ in range(FRAMES_PER_VIDEO + 1):
            x_coord_R_hand.append(0)
            y_coord_R_hand.append(0)
            z_coord_R_hand.append(0)
        #print("❌ Right hand not detected")

    return x_coord_R_hand, y_coord_R_hand, z_coord_R_hand


def get_coord_for_L_hand(results):
    """
    Extracts x, y, and z coordinates for the left hand landmarks from the results.

    Parameters:
    - results (dict): Results obtained from holistic processing.

    Returns:
    - tuple: Three lists containing x, y, and z coordinates for the left hand landmarks.
    """
    x_coord_L_hand = []
    y_coord_L_hand = []
    z_coord_L_hand = []

    if results.left_hand_landmarks:
        # Append coordinates if landmarks are detected
        for landmark in results.left_hand_landmarks.landmark:
            if landmark:
                x_coord_L_hand.append(landmark.x)
                y_coord_L_hand.append(landmark.y)
                z_coord_L_hand.append(landmark.z)
            else:
                x_coord_L_hand.append(0)
                y_coord_L_hand.append(0)
                z_coord_L_hand.append(0)
        #print("✅ Left hand detected")
    else:
        for _ in range(FRAMES_PER_VIDEO + 1):
            x_coord_L_hand.append(0)
            y_coord_L_hand.append(0)
            z_coord_L_hand.append(0)
        #print("❌ Left hand not detected")

    return x_coord_L_hand, y_coord_L_hand, z_coord_L_hand


def get_coord_for_pose(results):
    """
    Extracts x, y, and z coordinates for the pose landmarks from the results.

    Parameters:
    - results (dict): Results obtained from holistic processing.

    Returns:
    - tuple: Three lists containing x, y, and z coordinates for the pose landmarks.
    """
    x_coord_pose = []
    y_coord_pose = []
    z_coord_pose = []

    if results.pose_landmarks:
        # Loop over all landmarks for the pose
        for landmark in results.pose_landmarks.landmark:
            if landmark:
                x_coord_pose.append(landmark.x)
                y_coord_pose.append(landmark.y)
                z_coord_pose.append(landmark.z)
            else:
                x_coord_pose.append(0)
                y_coord_pose.append(0)
                z_coord_pose.append(0)
        #print("✅ Pose detected")

    else:
        x_coord_pose = [0] * 33
        y_coord_pose = [0] * 33
        z_coord_pose = [0] * 33
        #print("❌ Pose not detected")

    x_coord_pose = x_coord_pose[0:25]
    y_coord_pose = y_coord_pose[0:25]
    z_coord_pose = z_coord_pose[0:25]

    return x_coord_pose, y_coord_pose, z_coord_pose


def get_norm_coord(x, y, z, results):
    """
    Normalize x, y, and z coordinates with respect to the pose landmarks.

    Parameters:
    - x (list): List of x coordinates.
    - y (list): List of y coordinates.
    - z (list): List of z coordinates.
    - results (dict): Results obtained from holistic processing.

    Returns:
    - tuple: Three lists containing normalized x, y, and z coordinates.
    """
    x_coord_pose, y_coord_pose, z_coord_pose = get_coord_for_pose(results)

    x_norm = []
    y_norm = []
    z_norm = []

    for coord_x, coord_y, coord_z in zip(x, y, z):
        if all(coord == 0 for coord in (coord_x, coord_y, coord_z)):
            x_norm.append(0)
            y_norm.append(0)
            z_norm.append(0)
        else:
            x_norm.append((coord_x - x_coord_pose[0])/(abs(x_coord_pose[11]-x_coord_pose[12])))
            y_norm.append((coord_y - y_coord_pose[0])/(abs(x_coord_pose[11]-x_coord_pose[12])))
            z_norm.append((coord_z - z_coord_pose[0])/(abs(x_coord_pose[11]-x_coord_pose[12])))

    return x_norm, y_norm, z_norm


def coordinates_per_frame(results):
    """
    Aggregates x, y, and z coordinates for hand and pose landmarks into a single list for a frame.

    Parameters:
    - results (dict): Results obtained from holistic processing.

    Returns:
    - list: Combined list of x, y, and z coordinates for hand and pose landmarks.
    """
    x_coord_R_hand, y_coord_R_hand, z_coord_R_hand = get_coord_for_R_hand(results)
    x_coord_L_hand, y_coord_L_hand, z_coord_L_hand = get_coord_for_L_hand(results)
    x_coord_pose, y_coord_pose, z_coord_pose = get_coord_for_pose(results)

    X_frame = x_coord_R_hand + y_coord_R_hand + z_coord_R_hand + \
              x_coord_L_hand + y_coord_L_hand + z_coord_L_hand + \
              x_coord_pose + y_coord_pose + z_coord_pose

    return X_frame


def normalized_coordinates_per_frame(results):
    """
    Gathers normalized x, y, and z coordinates for hand and pose landmarks into a single list for a frame.

    Parameters:
    - results (dict): Results obtained from holistic processing.

    Returns:
    - list: Combined list of normalized x, y, and z coordinates for hand and pose landmarks.
    """
    # Retrieve original coordinates
    x_coord_R_hand, y_coord_R_hand, z_coord_R_hand = get_coord_for_R_hand(results)
    x_coord_L_hand, y_coord_L_hand, z_coord_L_hand = get_coord_for_L_hand(results)
    x_coord_pose, y_coord_pose, z_coord_pose = get_coord_for_pose(results)

    # Calculate normalized coordinates
    x_norm_coord_R_hand, y_norm_coord_R_hand, z_norm_coord_R_hand  = get_norm_coord(x_coord_R_hand,
                                                                                    y_coord_R_hand,
                                                                                    z_coord_R_hand,
                                                                                    results)
    x_norm_coord_L_hand, y_norm_coord_L_hand, z_norm_coord_L_hand = get_norm_coord(x_coord_L_hand,
                                                                                   y_coord_L_hand,
                                                                                   z_coord_L_hand,
                                                                                   results)
    x_norm_coord_pose, y_norm_coord_pose, z_norm_coord_pose = get_norm_coord(x_coord_pose,
                                                                             y_coord_pose,
                                                                             z_coord_pose,
                                                                             results)

    # Combine normalized coordinates into a single list
    X_frame_norm = x_norm_coord_R_hand + y_norm_coord_R_hand + z_norm_coord_R_hand + \
                   x_norm_coord_L_hand + y_norm_coord_L_hand + z_norm_coord_L_hand + \
                   x_norm_coord_pose + y_norm_coord_pose + z_norm_coord_pose

    return X_frame_norm


def mediapipe_video_to_coord(X):
    """
    Converts a list of MediaPipe videos into a tensor of coordinates.

    Parameters:
    - X (list): List of video frames.

    Returns:
    - tf.Tensor: Tensor of coordinates extracted from the videos.
    """
    X_coord = []

    for video_count in range(len(X)):
        video = X[video_count]
        video_frames = []

        for frame_count in range(FRAMES_PER_VIDEO):
            # start = time.time()
            frame = video[frame_count]
            results = detect_landmarks(frame)
            # print('Frame Count:', frame_count, 'Landmark:', time.time() - start)
            # middle = time.time()
            X_frame = normalized_coordinates_per_frame(results)
            # print('Normalize:', time.time() - middle)
            video_frames.append(X_frame)

        X_coord.append(video_frames)
    X_coord = tf.convert_to_tensor(X_coord)

    return X_coord


def train_model(X_aug_coord, X_val_coord, y_aug, y_cat_val):
    """
    Trains a simple LSTM model on the provided coordinate data.

    Parameters:
    - X_aug_coord (np.ndarray): Augmented coordinate data for training.
    - X_val_coord (np.ndarray): Coordinate data for validation.
    - y_aug (np.ndarray): Augmented labels for training.
    - y_cat_val (np.ndarray): Categorical labels for validation.

    Returns:
    - tf.keras.Model: Trained LSTM model.
    """
    def model_initialize_simple(dim):
        model = Sequential()
        model.add(layers.Masking(input_shape=(FRAMES_PER_VIDEO, dim), mask_value=0))
        model.add(layers.LSTM(units=512, activation="tanh", return_sequences=True))
        model.add(layers.LSTM(units=256, activation="tanh"))
        model.add(layers.Dense(N_CLASSES, activation="softmax"))

        return model

    def model_compile(model):
        learning_rate = 1e-4

        model.compile(
            loss="categorical_crossentropy",
            optimizer=Adam(learning_rate=learning_rate),
            metrics=["accuracy"]
        )

        return model

    def model_fit(model, epoch, batch_size):
        es = EarlyStopping(patience=30, restore_best_weights=True)

        history = model.fit(
            X_aug_coord,
            y_aug,
            epochs=epoch,
            batch_size=batch_size,
            validation_data=(X_val_coord, y_cat_val),
            verbose=1,
            callbacks=[es]
        )

        return model

    # 21 Landmarks per hand (42)
    # 25 Landmarks per pose
    # dim = (42 + 25) * 3 (X, Y, Z coords)
    dim = 201

    model = model_initialize_simple(dim)
    model = model_compile(model)
    model = model_fit(model, epoch=300, batch_size=16)

    return model
