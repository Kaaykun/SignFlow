import streamlit as st
import cv2
import tempfile
import numpy as np
import mediapipe as mp
import os
import sys
import pandas as pd
import tempfile

root_path = os.path.join(
    os.path.dirname(__file__),
    "..",
)
sys.path.append(root_path)

file_path = os.path.dirname(os.path.abspath(__file__))

from backend.ml_logic.model import mediapipe_video_to_coord, detect_landmarks
from backend.ml_logic.preprocessor import sample_frames
from backend.ml_logic.registry import load_model, draw_landmarks

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def get_num_frames(video_path):
    '''
    Give the number of frames of a particular video.
    Input: local video path
    Ouput: Number of frames
    '''
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    return num_frames

def preprocess_video(uploaded_file):
    '''
    This functions aims to proprocess the video uploaded on streamlit in order to perfrom prediction on it.
    1. Save the video in a temporary file
    2. Get the number of frames of this video and take a sample of 20 frames from them
    3. Convert the frames to a numpy array. The shape of the frames is now (20,512,512,3)
    4. Expand the dimension of the frames to give them a shape of (1,20,512,512,3)
    5. Calling the mediapipe_video_to_coord to have for each frame a list of coordinates.

    Input: a video file.mp4
    Output: a tensor of (1,20,201) representing the coordinates for the 20 frames of the video
    '''
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

def display_videos_for_word(chosen_word, video_urls):
    '''
    This function aims to give example of two videos for a chosen word.
    Input:
    - chosen word as a string
    - local path to videos'''

    col1, col2 = st.columns(2)
    with col1:
        st.video(video_urls[chosen_word][0])
    with col2:
        st.video(video_urls[chosen_word][1])

def show_annotated_video(uploaded_file):
        temp_video_path = os.path.join(tempfile.gettempdir(), "temp_video.mp4")
        with open(temp_video_path, "wb") as temp_file:
            temp_file.write(uploaded_file.read())

        temp_dir = tempfile.TemporaryDirectory()
        output_video_path = os.path.join(temp_dir.name, "video.mp4")

        cap = cv2.VideoCapture(temp_video_path)

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename=output_video_path, fourcc=fourcc, fps=fps, frameSize=(frame_width, frame_height))

        while cap.isOpened():
            ret, frame_video = cap.read()
            if not ret:
                break
            # Process the frame for landmark detection
            results_video = detect_landmarks(frame_video)

            # Draw landmarks on the frame
            annotated_image = draw_landmarks(results_video, frame_video)

            # Write the frame with annotations to the output video
            out.write(annotated_image)


        cap.release()
        out.release()
        cv2.destroyAllWindows()

        st.video(output_video_path)
        temp_dir.cleanup()

def show_annotated_frame(uploaded_file):
    temp_video_path = os.path.join(tempfile.gettempdir(), "temp_video.mp4")
    with open(temp_video_path, "wb") as temp_file:
            temp_file.write(uploaded_file.read())

    num_frames = get_num_frames(temp_video_path)
    frames = sample_frames(temp_video_path, num_frames)
    X= np.array(frames)
    # X shape is (20, 512, 512, 3)

    sampled_frames = (5,10,15)
    annotated_frame = []
    for i in sampled_frames:
        results = detect_landmarks(X[i])
        image = draw_landmarks(results, X[i])
        annotated_frame.append(image)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(annotated_frame[0])
    with col2:
        st.image(annotated_frame[1])
    with col3:
        st.image(annotated_frame[2])

def video_uploading_page():
    '''
    This function controls the presentation of the first page of the streamlit.
    It displays buttons and allow the user to select words and calls the necessary functions
    to make a prediction after importing a model
    '''
    model = load_model()

    st.title("Sign detection")
    banner_image = os.path.join(file_path, 'sign_banner.jpg')

    st.image(banner_image, use_column_width=True)
    st.markdown("""
    <div style="display: flex; justify-content: center; align-items: center;">
        <h2>Select a word and learn its sign!</h2>
    </div>
    """, unsafe_allow_html=True)

    classes = ['Select a sign','beer','bye','drink','go','hello','love','many','no','thank you','what','work','world','yes','you']
    video_urls = {
        'beer': [os.path.join(file_path,'videos_demo','05707.mp4'), os.path.join(file_path,'videos_demo','05708.mp4')],
        'bye': [os.path.join(file_path,'videos_demo','08512.mp4'), os.path.join(file_path,'videos_demo','08517.mp4')],
        'drink': [os.path.join(file_path,'videos_demo','69302.mp4'), os.path.join(file_path,'videos_demo','65539.mp4')],
        'go': [os.path.join(file_path,'videos_demo', '24946.mp4'), os.path.join(file_path,'videos_demo','24941.mp4')],
        'hello': [os.path.join(file_path,'videos_demo', '27184.mp4'), os.path.join(file_path,'videos_demo', '27172.mp4')],
        'love': [os.path.join(file_path,'videos_demo','34123.mp4'), os.path.join(file_path,'videos_demo','34124.mp4')],
        'many': [os.path.join(file_path,'videos_demo','69396.mp4'), os.path.join(file_path,'videos_demo','34824.mp4')],
        'no': [os.path.join(file_path,'videos_demo','69411.mp4'), os.path.join(file_path,'videos_demo','38525.mp4')],
        'thank you': [os.path.join(file_path,'videos_demo', '69502.mp4'), os.path.join(file_path,'videos_demo','66598.mp4')],
        'what': [os.path.join(file_path,'videos_demo','69531.mp4'), os.path.join(file_path,'videos_demo','62968.mp4')],
        'work': [os.path.join(file_path,'videos_demo','63790.mp4'), os.path.join(file_path,'videos_demo','63789.mp4')],
        'world': [os.path.join(file_path,'videos_demo', '63836.mp4'), os.path.join(file_path,'videos_demo','63837.mp4')],
        'yes': [os.path.join(file_path,'videos_demo','69546.mp4'), os.path.join(file_path,'videos_demo','64287.mp4')],
        'you': [os.path.join(file_path,'videos_demo','69547.mp4'), os.path.join(file_path,'videos_demo','64385.mp4')]
    }

    chosen_word = st.selectbox("Choose a word:",classes, index = 0, label_visibility = 'collapsed')

    # if st.button("Show videos"):
    if chosen_word in video_urls:
        display_videos_for_word(chosen_word, video_urls)

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        st.video(uploaded_file)

    if st.button("What is this sign?"):
        if uploaded_file is not None:
            X_coord = preprocess_video(uploaded_file)

            if model is not None:
                st.write('**🛠️ AI at work... 🦾**')
            else:
                st.write('Failed to load the model')

            prediction = pd.DataFrame(model.predict(X_coord))
            prediction.columns = ['I','beer','bye','drink','go','hello','love','many','no','thank you','what','work','world','yes','you']
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

            # st.write(f'Confidence : {max_probability}')
        else:
            st.write("Please upload a video file...")

    #  if st.button("Coordinates extraction to detect the sign"):
    #     if uploaded_file is not None:
    #         show_annotated_frame(uploaded_file)
    #     else:
    #         st.write("Please upload a video file...")

def video_streaming_page():
    '''
    This function controls the presentation of the second page of the streamlit.
    '''
    st.title("Live Sign Detection")


def main():
    st.sidebar.title("SIGN FLOW PAGES")
    pages = ["Video Uploading Sign detection", "Live Sign Detection"]
    choice = st.sidebar.selectbox("Choose a page:", pages)

    if choice == "Video Uploading Sign detection":
        video_uploading_page()
    elif choice == "Live Sign Detection":
        video_streaming_page()

if __name__ == "__main__":
    main()
