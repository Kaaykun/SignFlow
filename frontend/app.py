# Import dependencies
from streamlit_webrtc import webrtc_streamer
import streamlit as st
import pandas as pd
import numpy as np
import threading
import tempfile
import queue
import cv2
import os
import av
import sys

# Enable module importing from backend
root_path = os.path.join(
    os.path.dirname(__file__),
    "..",
)
sys.path.append(root_path)

# Import modules from backend
from backend.ml_logic.model import mediapipe_video_to_coord, detect_landmarks
from backend.ml_logic.preprocessor import sample_frames
from backend.ml_logic.registry import load_model, draw_landmarks

# Cach the LSTM model
@st.cache_resource
def preload_model():
    model = load_model()
    return model

# Global variable initialization
model = preload_model()

file_path = os.path.dirname(os.path.abspath(__file__))

pause_threshold = 3
pred_accumulator = []
pause_accumulator = []
pause:list[bool] = [False]

result_queue: "queue.Queue[List[Detection]]" = queue.Queue() #type:ignore
second_queue: "queue.Queue[List[Detection]]" = queue.Queue() #type:ignore

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

######################## Logic for video_uploading_page ########################

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
    X = np.expand_dims(X, axis=0)
    X_coord = mediapipe_video_to_coord(X)

    return X_coord

def display_videos_for_word(chosen_word, video_urls):
    col1, col2 = st.columns(2)
    with col1:
        st.video(video_urls[chosen_word][0])
    with col2:
        st.video(video_urls[chosen_word][1])

def video_uploading_page():
    st.markdown("""
    <div style="display: flex; justify-content: center; align-items: center;">
        <h1>Sign Flow</h1>
    </div>
    """, unsafe_allow_html=True)
    banner_image = os.path.join(file_path, 'sign_banner.jpg')

    st.image(banner_image, use_column_width=True)
    st.markdown("""
    <div style="display: flex; justify-content: center; align-items: center;">
        <h2>Select a word and learn its sign!</h2>
    </div>
    """, unsafe_allow_html=True)

    classes = ['Select a sign','beer','bye','drink','go','hello',
               'love','many','no','thank you','what','work','world',
               'yes','you']
    video_urls = {
        'beer': [os.path.join(file_path,'videos_demo','05707.mp4'),
                 os.path.join(file_path,'videos_demo','05708.mp4')],
        'bye': [os.path.join(file_path,'videos_demo','08512.mp4'),
                os.path.join(file_path,'videos_demo','08517.mp4')],
        'drink': [os.path.join(file_path,'videos_demo','69302.mp4'),
                  os.path.join(file_path,'videos_demo','65539.mp4')],
        'go': [os.path.join(file_path,'videos_demo', '24946.mp4'),
               os.path.join(file_path,'videos_demo','24941.mp4')],
        'hello': [os.path.join(file_path,'videos_demo', '27184.mp4'),
                  os.path.join(file_path,'videos_demo', '27172.mp4')],
        'love': [os.path.join(file_path,'videos_demo','34123.mp4'),
                 os.path.join(file_path,'videos_demo','34124.mp4')],
        'many': [os.path.join(file_path,'videos_demo','69396.mp4'),
                 os.path.join(file_path,'videos_demo','34824.mp4')],
        'no': [os.path.join(file_path,'videos_demo','69411.mp4'),
               os.path.join(file_path,'videos_demo','38525.mp4')],
        'thank you': [os.path.join(file_path,'videos_demo', '69502.mp4'),
                      os.path.join(file_path,'videos_demo','66598.mp4')],
        'what': [os.path.join(file_path,'videos_demo','69531.mp4'),
                 os.path.join(file_path,'videos_demo','62968.mp4')],
        'work': [os.path.join(file_path,'videos_demo','63790.mp4'),
                 os.path.join(file_path,'videos_demo','63789.mp4')],
        'world': [os.path.join(file_path,'videos_demo', '63836.mp4'),
                  os.path.join(file_path,'videos_demo','63837.mp4')],
        'yes': [os.path.join(file_path,'videos_demo','69546.mp4'),
                os.path.join(file_path,'videos_demo','64287.mp4')],
        'you': [os.path.join(file_path,'videos_demo','69547.mp4'),
                os.path.join(file_path,'videos_demo','64385.mp4')]
    }

    chosen_word = st.selectbox("Choose a word:",classes, index = 0,
                               label_visibility = 'collapsed')

    if chosen_word in video_urls:
        display_videos_for_word(chosen_word, video_urls)

    uploaded_file = st.file_uploader("**Upload a video file:**",
                                     type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        st.video(uploaded_file)

    if st.button("What is this sign?"):
        X_coord = preprocess_video(uploaded_file)

        if model is not None:
            st.write('🛠️ AI at work... 🦾')
            st.write('✅ Sign detected')
        else:
            st.write('Failed to load the model')

        prediction = pd.DataFrame(model.predict(X_coord)) #type:ignore
        prediction.columns = ['I','beer','bye','drink','go','hello',
                              'love','many','no','thank you','what',
                              'work','world','yes','you']
        st.write('**Prediction of the sign :ok_hand: :wave: :+1: :open_hands: ...**')

        max_probability_word = prediction.idxmax(axis=1).iloc[0]
        max_probability = prediction[max_probability_word].iloc[0]

        max_probability_word = max_probability_word.capitalize()
        max_probability = round(float(max_probability), 2)

        st.markdown(f"""
        <div style="display: flex; justify-content: center; align-items: center;">
            <h2 style="font-size: 2em;">{max_probability_word}</h2>
        </div>
        """, unsafe_allow_html=True)

######################## Logic for video_streaming_page ########################

def frames_to_predicton(frames):
        frames_resized = [cv2.resize(frame, (480, 480)) for frame in frames]
        frames_resized = np.expand_dims(np.array(frames_resized), axis=0)
        X_coord = mediapipe_video_to_coord(frames_resized)

        prediction = model.predict(X_coord)[0] #type:ignore

        if np.max(prediction) > 0.4:
            max_index = np.argmax(prediction)
            word_detected = mapping[max_index]
        else:
            word_detected = "..."

        return word_detected

def process_frames(frames):
    global result_queue
    word_detected = frames_to_predicton(frames)
    result_queue.put(word_detected)

def video_streaming_page():
    def video_frame_callback(frame):
        global pred_accumulator
        global pause_accumulator
        global pause

        frame = frame.to_ndarray(format="bgr24")
        results = detect_landmarks(frame)
        annotated_image = draw_landmarks(results, frame)

        if len(pred_accumulator) < 20 and not pause[0]:
            pred_accumulator.append(frame)

        elif len(pred_accumulator) == 20 and not pause[0]:
            pause[0] = True
            print("------------- AI running.... -------------")

            processing_thread = threading.Thread(target=process_frames, args=(pred_accumulator,))
            processing_thread.start()

        elif len(pred_accumulator) == 20 and pause[0]:
            pause_accumulator.append(frame)
            annotated_image = np.zeros_like(annotated_image)

        if len(pred_accumulator) == 20 and pause[0] and len(pause_accumulator) == pause_threshold:
            pred_accumulator.clear()
            pause_accumulator.clear()
            pause[0] = False

        second_queue.put(len(pred_accumulator))

        return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

    ctx = webrtc_streamer(key="example",
                    video_frame_callback=video_frame_callback,
                    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                    media_stream_constraints={"video": True, "audio": False},
                    async_processing=True)

    if ctx.state.playing:
        frame_count = 0
        frame_counter_placeholder = st.empty()
        frame_counter_placeholder.text("Recording... 0%")

        result = ""
        prediction_placeholder = st.empty()

        while True:
            frame_count = (second_queue.get() + 1) * 100 / 20
            frame_counter_placeholder.text(f"📽️ Recording... {frame_count: .0f}%")

            if frame_count == 100:
                frame_counter_placeholder.text("🛠️ AI at work... 🦾")
                result += result_queue.get() + " -> "
                prediction_placeholder.markdown(f"<h1>{result}</h1>", unsafe_allow_html=True)

##################################### Main #####################################

def main():

    st.sidebar.title("SignFlow")
    pages = ["Sign Detection: Upload Video", "Sign Detection: Real Time"]
    choice = st.sidebar.selectbox("Page selector:", pages)

    if choice == "Sign Detection: Upload Video":
        video_uploading_page()
    elif choice == "Sign Detection: Real Time":
        video_streaming_page()



if __name__ == "__main__":
    main()
