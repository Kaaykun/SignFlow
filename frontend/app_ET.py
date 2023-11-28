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
second_queue: "queue.Queue[List[Detection]]" = queue.Queue()

def frames_to_predicton(frames):
    frames_resized = [cv2.resize(frame, (480, 480)) for frame in frames]
    frames_resized = np.array(frames_resized)
    frames_resized = np.expand_dims(frames_resized, axis=0)
    X_coord = mediapipe_video_to_coord(frames_resized)
    prediction = model.predict(X_coord)[0]
    if np.max(prediction) > 0.4:
        max_index = np.argmax(prediction)
        word_detected = mapping[max_index]
    else:
        word_detected = "..."

    print(frames_resized.shape)
    print(X_coord.shape)
    print(prediction)
    # print(max_index)
    print(word_detected)
    return word_detected



def video_frame_callback(frame):
    #measure time
    start_time = time.time()

    #annotate the frame
    frame = frame.to_ndarray(format="bgr24")
    results = detect_landmarks(frame)
    annotated_image = draw_landmarks(results, frame)
    print(frame.shape)

    # Accumulate 20 frames
    global frame_accumulator
    frame_accumulator.append(frame)
    if len(frame_accumulator) == 20:
        print("------------- AI running.... -------------")
        end_time = time.time()
        elapsed_time = end_time - start_time
        # print(np.array(frame_accumulator).shape)

        word_detected = frames_to_predicton(frame_accumulator)
        # word_detected = "Prediction = " + word_detected + ", " + str(round(elapsed_time,3)) + "seconds"
        print(word_detected)
        # st.session_state["prediction"] = word_detected
        result_queue.put(word_detected)

        frame_accumulator = []

        # with lock:
        #     prediction_list.append(word_detected)
        #     prediction = word_detected
    second_queue.put(len(frame_accumulator))

    return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")


def video_streaming_page():
    st.title("Live Sign Detection")


##########################



def main():
    st.sidebar.title("Pages")

    ctx = webrtc_streamer(key="example", video_frame_callback=video_frame_callback)

    if ctx.state.playing:
        # return recording frame %
        frame_count = 0
        frame_counter_placeholder = st.empty()
        frame_counter_placeholder.text("Recording... 0%")

        # return a string of predictions
        result = ""
        prediction_placeholder = st.empty()

        while True:
            # time.sleep(0.5)

            # return recording frame %
            frame_count = (second_queue.get() + 1) * 100 / 20
            frame_counter_placeholder.text(f"Recording... {frame_count: .0f}%")

            if frame_count == 100:
                # return a string of predictions
                result += result_queue.get() + " → "
                prediction_placeholder.markdown(f"<h1>{result}</h1>", unsafe_allow_html=True)



    # if ctx.state.playing:
    #     # return a string of predictions
    #     # return recording frame %
    #     frame_counter_placeholder = st.empty()
    #     frame_counter_placeholder.text("Recording...: 0%")
    #     while True:
    #         # time.sleep(0.5)
    #         frame_count = second_queue.get() * 100 / 20
    #         frame_counter_placeholder.text(f"Recording...: {frame_count: .0f}%")

    # if ctx.state.playing:
    #     result = ""
    #     prediction_placeholder = st.empty()
    #     while True:
    #         # time.sleep(0.5)
    #         result += result_queue.get() + " → "
    #         prediction_placeholder.write(result)
    #         # frame_counter_placeholder.text(f"Frame Accumulator Length: {len(frame_accumulator)}")
    #         # prediction_placeholder.markdown(f"<h1>{result}</h1>", unsafe_allow_html=True)


    # pages = ["Upload your video", "Sign live"]
    # choice = st.sidebar.radio("Sign Flow", pages)

    # if choice == "Upload your video":
    #     video_uploading_page()
    # elif choice == "Sign live":
    #     video_streaming_page()

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

if __name__ == "__main__":
    main()
