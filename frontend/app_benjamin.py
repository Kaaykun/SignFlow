import streamlit as st
import cv2
import tempfile
import numpy as np
import mediapipe as mp
import os
import pandas as pd

from backend.ml_logic.model import mediapipe_video_to_coord
from backend.ml_logic.preprocessor import sample_frames
from backend.ml_logic.registry import load_model

from backend.params import VIDEO_PATH

mp_holistic = mp.solutions.holistic


def get_num_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    return num_frames

def preprocess_video(uploaded_file):

    temp_video_path = os.path.join(tempfile.gettempdir(), "temp_video.mp4")
    with open(temp_video_path, "wb") as temp_file:
            temp_file.write(uploaded_file.read())

    num_frames = get_num_frames(temp_video_path)
    frames = sample_frames(temp_video_path, num_frames)
    X= np.array(frames)
    # X shape is (20, 512, 512, 3)
    X = np.expand_dims(X, axis=0)
    # X shape is (1, 20, 512, 512, 3)
    X_coord = mediapipe_video_to_coord(X)

    return X_coord

def video_uploading_page():
    st.title("Video Uploading Sign detection")

    classes = ['beer','bye','drink','go','hello','love','many','no','thank you','what','work','world','yes','you']

    video_urls = {
        'beer': f'{VIDEO_PATH}05712.mp4',
        'bye': f'{VIDEO_PATH}08510.mp4',
        'drink': f'{VIDEO_PATH}69302.mp4',
        'go': f'{VIDEO_PATH}69345.mp4',
        'hello': f'{VIDEO_PATH}27184.mp4',
        'love': f'{VIDEO_PATH}34123.mp4',
        'many': f'{VIDEO_PATH}69396.mp4',
        'no': f'{VIDEO_PATH}69411.mp4',
        'thankyou':f'{VIDEO_PATH}69502.mp4',
        'what':f'{VIDEO_PATH}69531.mp4',
        'work':f'{VIDEO_PATH}63806.mp4',
        'world':f'{VIDEO_PATH}63836.mp4',
        'yes':f'{VIDEO_PATH}69546.mp4',
        'you':f'{VIDEO_PATH}69547.mp4'
    }

    chosen_word = st.selectbox("Choose a word:",classes)

    if st.button("Show Video"):
        if chosen_word in video_urls:
            video_url = video_urls[chosen_word]
            st.video(video_url)

    st.write("Upload a video file:")
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        st.video(uploaded_file)
        X_coord = preprocess_video(uploaded_file)

        model = load_model()


        if model is not None:
            st.write('Model running')
        else:
            st.write('Failed to load the model')

        prediction = pd.DataFrame(model.predict(X_coord))
        prediction.columns = ['I','beer','bye','drink','go','hello','love','many','no','thankyou','what','work','world','yes','you']

        max_acuracy_word = prediction.idxmax(axis=1).iloc[0]
        max_accuracy = prediction[max_acuracy_word].iloc[0]

        st.write(prediction)
        st.write(f'{max_acuracy_word} with probability {max_accuracy}')


def video_streaming_page():
    st.title("Live Sign Detection")


def main():
    st.sidebar.title("Pages")
    pages = ["Video Uploading Sign detection", "Live Sign Detection"]
    choice = st.sidebar.radio("Sign Flow", pages)

    if choice == "Video Uploading Sign detection":
        video_uploading_page()
    elif choice == "Live Sign Detection":
        video_streaming_page()

if __name__ == "__main__":
    main()
