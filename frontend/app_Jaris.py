import streamlit as st
import cv2
import numpy as np

from backend.ml_logic.preprocessor import sample_frames
from backend.params import FRAMES_PER_VIDEO, TARGET_SIZE, VIDEO_PATH

X = np.empty((1, FRAMES_PER_VIDEO, *TARGET_SIZE, 3), dtype=np.uint8)

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


# Streamlit app
def main():
    st.title("Sign Flow")
    st.markdown('Welcome to our project streamlit!')

    # Allow user to upload a video file
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        # Display the uploaded video
        st.video(uploaded_file)

        video_path = f'{VIDEO_PATH}00335.mp4'
        # Process the video using the model
        frames = sample_frames(video_path, 58)

        # Display the processed frames (you can customize this part based on your needs)
        for frame in frames:
            st.image(frame, channels="BGR", caption="Processed Frame")

if __name__ == "__main__":
    main()
