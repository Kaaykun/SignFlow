import numpy as np
import cv2

from params import FRAMES_PER_VIDEO, TARGET_SIZE, VIDEO_PATH


def sample_frames(video_path, total_frames):
    """
    Sample frames from a video file located at 'video_path'.

    Parameters:
    - video_path (str): Path to the video file.
    - total_frames (int): Total number of frames in the video.

    Returns:
    - list: A list containing sampled frames from the video file.
    """
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
                frame = cv2.resize(frame, TARGET_SIZE)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

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

    X = np.empty((input_length, FRAMES_PER_VIDEO, *TARGET_SIZE, 3), dtype=np.uint8)

    for i, row in selected_df.iterrows():
        video_id = row['video_id']
        total_frames = row['video_length']
        video_path = f'{VIDEO_PATH}{video_id}.mp4'

        sampled_frames = sample_frames(video_path, total_frames)

        X[i] = np.array(sampled_frames)

    return X
