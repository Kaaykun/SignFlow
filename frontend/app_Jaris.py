import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

from backend.ml_logic.preprocessor import sample_frames
from backend.ml_logic.registry import load_model
from backend.params import FRAMES_PER_VIDEO, TARGET_SIZE, VIDEO_PATH

mp_holistic = mp.solutions.holistic

def detect_landmarks(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, static_image_mode=True) as holistic:
        results = holistic.process(frame_rgb)
    return results

def get_coord_for_R_hand(results):
    x_coord_R_hand = []
    y_coord_R_hand = []
    z_coord_R_hand = []

    if results.right_hand_landmarks:
        #append coordinates
        for landmark in results.right_hand_landmarks.landmark:
            if landmark:
                x_coord_R_hand.append(landmark.x)
                y_coord_R_hand.append(landmark.y)
                z_coord_R_hand.append(landmark.z)
            else:
                x_coord_R_hand.append(0)
                y_coord_R_hand.append(0)
                z_coord_R_hand.append(0)
        print("✅ Right hand detected")

    else:
        for index in range(21):
            x_coord_R_hand.append(0)
            y_coord_R_hand.append(0)
            z_coord_R_hand.append(0)
        print("❌ Right hand not detected")

    return x_coord_R_hand, y_coord_R_hand, z_coord_R_hand

def get_coord_for_L_hand(results):
    x_coord_L_hand = []
    y_coord_L_hand = []
    z_coord_L_hand = []

    if results.left_hand_landmarks:
        # Append coordinates
        for landmark in results.left_hand_landmarks.landmark:
            if landmark:
                x_coord_L_hand.append(landmark.x)
                y_coord_L_hand.append(landmark.y)
                z_coord_L_hand.append(landmark.z)
            else:
                x_coord_L_hand.append(0)
                y_coord_L_hand.append(0)
                z_coord_L_hand.append(0)
        print("✅ Left hand detected")
    else:
        # No left hand detected, populate with NaN values
        for index in range(21):
            x_coord_L_hand.append(0)
            y_coord_L_hand.append(0)
            z_coord_L_hand.append(0)
        print("❌ Left hand not detected")

    return x_coord_L_hand, y_coord_L_hand, z_coord_L_hand

def get_coord_for_pose(results):
    x_coord_pose = []
    y_coord_pose = []
    z_coord_pose = []

    if results.pose_landmarks:
        # loop over all landmarks for each hand
        for landmark in results.pose_landmarks.landmark:
            if landmark:
                x_coord_pose.append(landmark.x)
                y_coord_pose.append(landmark.y)
                z_coord_pose.append(landmark.z)
                # print(f"Landmark X: {landmark.x}, Y: {landmark.y}, Z: {landmark.z}")
            else:
                x_coord_pose.append(0)
                y_coord_pose.append(0)
                z_coord_pose.append(0)
        print("✅ Pose detected")

    else:
        x_coord_pose = [0] * 33
        y_coord_pose = [0] * 33
        z_coord_pose = [0] * 33
        print("❌ Pose not detected")

    x_coord_pose = x_coord_pose[11:25]
    y_coord_pose = y_coord_pose[11:25]
    z_coord_pose = z_coord_pose[11:25]

    return x_coord_pose, y_coord_pose, z_coord_pose

def coordinates_per_frame(results):
    x_coord_R_hand, y_coord_R_hand, z_coord_R_hand = get_coord_for_R_hand(results)
    x_coord_L_hand, y_coord_L_hand, z_coord_L_hand = get_coord_for_L_hand(results)
    x_coord_pose, y_coord_pose, z_coord_pose = get_coord_for_pose(results)

    X_frame = x_coord_R_hand + y_coord_R_hand + z_coord_R_hand + \
          x_coord_L_hand + y_coord_L_hand + z_coord_L_hand + \
          x_coord_pose + y_coord_pose + z_coord_pose
    return X_frame

def mediapipe_video_to_coord(X):
    X_coord = []

    for video_count in range(len(X)):
        video = X[video_count]
        video_frames = []

        for frame_count in range(10):
            frame = video[frame_count]
            results = detect_landmarks(frame)
            X_frame = coordinates_per_frame(results)
            video_frames.append(X_frame)

        X_coord.append(video_frames)

    X_coord = tf.convert_to_tensor(X_coord)
    return X_coord

# Function to perform frame sampling
def sample_frames(video_path, total_frames):
    frames = []
    cap = cv2.VideoCapture(video_path)

    frame_indices = []

    while len(set(frame_indices)) != FRAMES_PER_VIDEO:
        frame_indices = sorted(np.random.uniform(0, total_frames-5, FRAMES_PER_VIDEO).astype(int))

    frame_counter = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            if frame_counter in frame_indices:
                # Resize frame to required size
                frame = cv2.resize(frame, (150, 150))
                # CV2 output BGR -> converting to RGB
                #frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                # Append to list of frames
                frames.append(frame)

            frame_counter += 1

            if len(frames) == FRAMES_PER_VIDEO:
                break

    finally:
        cap.release()

    return frames

def create_X(selected_df, input_length):
    """
    Create an array X for sampled frames from selected videos.

    Parameters:
    - selected_df (pandas.DataFrame): DataFrame containing selected video information.
    - input_length (int): Length of the selected DataFrame.

    Returns:
    - numpy.ndarray: An array containing sampled frames from selected videos.
    """

    np.random.seed(9)

    X = np.empty((1, FRAMES_PER_VIDEO, *TARGET_SIZE, 3), dtype=np.uint8)

    for i, row in selected_df.iterrows():
        video_id = row['video_id']
        total_frames = row['video_length']
        video_path = f'{VIDEO_PATH}{video_id}.mp4'

        sampled_frames = sample_frames(video_path, total_frames)

        X[i] = np.array(sampled_frames)

    return X

# Streamlit app
def main():
    st.title("Sign Flow")
    st.markdown('Welcome to our project streamlit!')

    X = np.empty((1, FRAMES_PER_VIDEO, *TARGET_SIZE, 3), dtype=np.uint8)

    # Allow user to upload a video file
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        # Display the uploaded video
        st.video(uploaded_file)

        video_path = f'{VIDEO_PATH}20982.mp4'
        # Process the video using the model
        frames = sample_frames(video_path, 36)
        X[0] = np.array(frames)
        X_coord = mediapipe_video_to_coord(X)


        model = load_model()
        if model is not None:
            st.write('Model loaded')
        else:
            st.write('Failed to load the model')

        st.write(model.predict(X_coord))


        # Display the processed frames (you can customize this part based on your needs)
        for frame in frames:
            st.image(frame, channels="BGR", caption="Processed Frame")

if __name__ == "__main__":
    main()
