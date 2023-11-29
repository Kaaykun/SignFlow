import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import av
import os
import logging
import matplotlib.pyplot as plt

import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

from backend.ml_logic.preprocessor import sample_frames
from backend.ml_logic.model import mediapipe_video_to_coord, detect_landmarks
from backend.ml_logic.registry import load_model, draw_landmarks
from backend.params import FRAMES_PER_VIDEO, TARGET_SIZE, VIDEO_PATH



def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    results = detect_landmarks(img)
    coords = mediapipe_video_to_coord(img)
    annotated_img = draw_landmarks(results, img)
    plt.imshow(annotated_img)

    return av.VideoFrame.from_ndarray(coords, format="bgr24")

def main():

    webrtc_streamer(key="example", video_frame_callback=video_frame_callback)

if __name__ == "__main__":
    main()














# Streamlit app
# def main():
    # st.sidebar.title("Pages")
    # pages = ["Video Uploading Sign detection", "Live Sign Detection"]
    # choice = st.sidebar.radio("Sign Flow", pages)

    # # if choice == "Video Uploading Sign detection":
    # #     video_uploading_page()
    # # elif choice == "Live Sign Detection":
    # #     video_streaming_page()

    # st.title("Sign Flow")
    # st.markdown('Welcome to our project streamlit!')

    # X = np.empty((1, FRAMES_PER_VIDEO, *TARGET_SIZE, 3), dtype=np.uint8)

    # # Allow user to upload a video file
    # uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

    # if uploaded_file is not None:
    #     # Display the uploaded video
    #     st.video(uploaded_file)

    #     video_path = f'{VIDEO_PATH}20982.mp4'
    #     # Process the video using the model
    #     frames = sample_frames(video_path, 36)
    #     X[0] = np.array(frames)
    #     X_coord = mediapipe_video_to_coord(X)


    #     model = load_model()
    #     if model is not None:
    #         st.write('Model loaded')
    #     else:
    #         st.write('Failed to load the model')

    #     st.write(model.predict(X_coord)) #type:ignore

    #     # Display the processed frames (you can customize this part based on your needs)
    #     for frame in frames:
    #         st.image(frame, channels="BGR", caption="Processed Frame")

    # webrtc_streamer(key="example", video_frame_callback=video_frame_callback)


# if __name__ == "__main__":
#     main()
