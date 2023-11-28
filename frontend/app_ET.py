import streamlit as st
import cv2
import tempfile
import numpy as np
import mediapipe as mp
import os
import pandas as pd
from streamlit_webrtc import webrtc_streamer
import av
import queue
import time

from backend.ml_logic.model import mediapipe_video_to_coord, detect_landmarks
from backend.ml_logic.preprocessor import sample_frames
from backend.ml_logic.registry import load_model, draw_landmarks

from backend.params import VIDEO_PATH

mp_holistic = mp.solutions.holistic
model = load_model()



frame_accumulator = []
# lock = threading.Lock()
# prediction_list = [""]

mapping = {'bye': 2,
 'love': 6,
 'many': 7,
 'world': 12,
 'thankyou': 9,
 'work': 11,
 'hello': 5,
 'go': 4,
 'yes': 13,
 'you': 14,
 'beer': 1,
 'I': 0,
 'drink': 3,
 'what': 10,
 'no': 8}
mapping = {v: k for k, v in mapping.items()}

result_queue: "queue.Queue[List[Detection]]" = queue.Queue()


def frames_to_predicton(frames):
    frames_resized = [cv2.resize(frame, (480, 480)) for frame in frames]
    frames_resized = np.array(frames_resized)
    frames_resized = np.expand_dims(frames_resized, axis=0)
    X_coord = mediapipe_video_to_coord(frames_resized)
    prediction = model.predict(X_coord)[0]
    max_index = np.argmax(prediction)
    word_detected = mapping[max_index]

    print(frames_resized.shape)
    print(X_coord.shape)
    print(prediction)
    print(max_index)
    print(word_detected)
    return word_detected


def video_frame_callback(frame):
    #annotate the frame
    frame = frame.to_ndarray(format="bgr24")
    results = detect_landmarks(frame)
    annotated_image = draw_landmarks(results, frame)
    print(frame.shape)

    # Accumulate frames for 2 seconds (20 frames)
    global frame_accumulator
    frame_accumulator.append(frame)
    if len(frame_accumulator) == 20:
        print("------------- AI running.... -------------")
        # print(np.array(frame_accumulator).shape)

        word_detected = frames_to_predicton(frame_accumulator)
        print(word_detected)
        st.write(word_detected)
        # st.session_state["prediction"] = word_detected
        result_queue.put(word_detected)

        frame_accumulator = []

        # with lock:
        #     prediction_list.append(word_detected)
        #     prediction = word_detected
    return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")


# def video_streaming_page():
#     st.title("Live Sign Detection")
#     webrtc_streamer(key="example", video_frame_callback=video_frame_callback)


##########################



def main():
    st.sidebar.title("Pages")
    ctx = webrtc_streamer(key="example", video_frame_callback=video_frame_callback)

    if ctx.state.playing:
        result = ""
        prediction_placeholder = st.empty()
        while True:
            time.sleep(0.5)
            result += result_queue.get() + " → "
            prediction_placeholder.write(result)





    # if "prediction" not in st.session_state:
    #     st.session_state["prediction"] = ""

    # if st.session_state["prediction"] != "":
    #     # st.write(st.session_state["prediction"])
    #     prediction_placeholder.text(st.session_state["prediction"])

    # global prediction
    # st.write(prediction)
    # # print(prediction_list)


    # while ctx.state.playing:
    #     with lock:
    #         prediction = prediction_list[-1]



    # pages = ["Video Uploading Sign detection", "Live Sign Detection"]
    # choice = st.sidebar.radio("Sign Flow", pages)

    # if choice == "Video Uploading Sign detection":
    #     video_uploading_page()
    # elif choice == "Live Sign Detection":
    #     video_streaming_page()


if __name__ == "__main__":
    main()
