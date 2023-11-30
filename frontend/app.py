# Importing necessary dependencies
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

# Setting Streamlit page configuration
st.set_page_config(page_title='SignFlow', page_icon='👋', layout="centered", initial_sidebar_state="auto", menu_items=None)

# Cach the LSTM model 15 classes
@st.cache_resource
def preload_model_uploading():
    model = load_model(target='uploading')
    return model

# Cach the LSTM model 7 classes
@st.cache_resource
def preload_model_live():
    model = load_model(target='live')
    return model

# Global variable initialization
model_uploading = preload_model_uploading()
model_live = preload_model_live()

file_path = os.path.dirname(os.path.abspath(__file__))

pause_threshold = 5
pred_accumulator = []
pause_accumulator = []
pause:list[bool] = [False]

result_queue: "queue.Queue[List[Detection]]" = queue.Queue() #type:ignore
second_queue: "queue.Queue[List[Detection]]" = queue.Queue() #type:ignore

mapping = {'I': 0,
           'beer': 1,
           'drink': 2,
           'go': 3,
           'hello': 4,
           'many': 5,
           'world': 6}

mapping = {v: k for k, v in mapping.items()}

######################## Logic for video_uploading_page ########################

def get_num_frames(video_path):
    """Get the number of frames in a video file"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    return num_frames


def preprocess_video(uploaded_file):
    """Preprocess the uploaded video file"""
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
    """Display videos associated with a selected word"""
    col1, col2 = st.columns(2)
    with col1:
        st.video(video_urls[chosen_word][0])
    with col2:
        st.video(video_urls[chosen_word][1])

def video_uploading_page():
    """Create the UI for video uploading and processing"""
    # Streamlit UI configuration
    banner_image = os.path.join(file_path, 'SignFlowLogo.png')
    st.image(banner_image, use_column_width=True, width=100)

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

        if model_uploading is not None:
            st.write('**Prediction of the sign :ok_hand: :wave: :+1: :open_hands: ...**')
        else:
            st.write('Failed to load the model')

        prediction = pd.DataFrame(model_uploading.predict(X_coord)) #type:ignore
        prediction.columns = ['I','beer','bye','drink','go','hello',
                              'love','many','no','thank you','what',
                              'work','world','yes','you']

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
        """Function to process frames and predict sign based on the frames"""
        frames_resized = [cv2.resize(frame, (480, 480)) for frame in frames]
        frames_resized = np.expand_dims(np.array(frames_resized), axis=0)
        X_coord = mediapipe_video_to_coord(frames_resized)

        prediction = model_live.predict(X_coord)[0] #type:ignore

        if np.max(prediction) > 0.6:
            max_index = np.argmax(prediction)
            word_detected = mapping[max_index] # Mapping the predicted index to the sign word
        else:
            word_detected = "..." # If prediction confidence is low

        return word_detected

def process_frames(frames):
    """Function to process frames and put the detected word in the result queue"""
    global result_queue
    word_detected = frames_to_predicton(frames)
    result_queue.put(word_detected)

def video_streaming_page():
    """Function to create the video streaming page"""
    # Streamlit UI configuration
    banner_image = os.path.join(file_path, 'SignFlowLogo.png')
    st.image(banner_image, use_column_width=True, width=100)

    st.markdown("""
    <div style="display: flex; justify-content: center; align-items: center;">
        <h2>Translate signs in real time!</h2>
    </div>
    """, unsafe_allow_html=True)

    def video_frame_callback(frame):
        """Callback function for processing each video frame"""
        global pred_accumulator
        global pause_accumulator
        global pause

        if len(pred_accumulator) == 0:
            print("📽️ START")

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

    # WebRTC streaming setup
    ctx = webrtc_streamer(key="example",
                    video_frame_callback=video_frame_callback,
                    #rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                    media_stream_constraints={"video": True, "audio": False},
                    async_processing=True)

    # Continuously update UI based on video processing
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
    """Main function to select and render different pages"""
    st.sidebar.title("Page selector")
    pages = ["Sign Detection: Upload Video", "Sign Detection: Real Time"]
    choice = st.sidebar.selectbox("Page selector:", pages, label_visibility='collapsed')

    if choice == "Sign Detection: Upload Video":
        video_uploading_page()
    elif choice == "Sign Detection: Real Time":
        video_streaming_page()

if __name__ == "__main__":
    main()
