import streamlit as st
import cv2
import tempfile
import numpy as np
import mediapipe as mp
import os
import pandas as pd

from backend.ml_logic.model import mediapipe_video_to_coord, detect_landmarks
from backend.ml_logic.preprocessor import sample_frames
from backend.ml_logic.registry import load_model, draw_landmarks

from backend.params import VIDEO_PATH

# from mediapipe.solutions.drawing_utils import mp_drawing
mp_holistic = mp.solutions.holistic

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

def detect_landmarks_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    with mp_holistic.Holistic(min_detection_confidence=0.2, min_tracking_confidence=0.2, static_image_mode=False) as holistic:
        results = holistic.process(frame_rgb)
    return results

# # def draw_(results, frame):
#     annotated_image = frame.copy()

#     if results.right_hand_landmarks:
#         mp_drawing.draw_landmarks(
#             image=annotated_image,
#             landmark_list=results.right_hand_landmarks,
#             connections=mp_holistic.HAND_CONNECTIONS)
#         print("✅ Right hand annotated")
#     else:
#         print("❌ Right hand not annotated")

#     if results.left_hand_landmarks:
#         mp_drawing.draw_landmarks(
#             image=annotated_image,
#             landmark_list=results.left_hand_landmarks,
#             connections=mp_holistic.HAND_CONNECTIONS)
#         print("✅ Left hand annotated")
#     else:
#         print("❌ Left hand not annotated")


#     if results.pose_landmarks:
#         mp_drawing.draw_landmarks(
#             image=annotated_image,
#             landmark_list=results.pose_landmarks,
#             connections=mp_holistic.POSE_CONNECTIONS)
#         print("✅ Pose annotated")
#     else:
#         print("❌ Pose not annotated")

#     return annotated_image

# def annotate_video(frames, video_path, temp_video_path):

#     for frame in frames:
#         cap = cv2.VideoCapture(video_path)

#         frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         fps = int(cap.get(cv2.CAP_PROP_FPS))

#         # Define the codec and create VideoWriter object
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         out = cv2.VideoWriter(filename=temp_video_path, fourcc=fourcc, fps=20, frameSize=(frame_width, frame_height))

#         if not out.isOpened():
#             print("Could not open output video file for writing.")

#         while cap.isOpened():
#             print("------")
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             # Process the frame for landmark detection
#             results = detect_landmarks_frame(frame)

#             # Draw landmarks on the frame
#             annotated_image = draw_(results, frame)

#             # Write the frame with annotations to the output video
#             out.write(annotated_image)

#             # cv2.imshow('Video', annotated_image)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         cap.release()
#         out.release()
#         cv2.destroyAllWindows()

#         return temp_video_path

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

def video_uploading_page():
    '''
    This function controls the presentation of the first page of the streamlit.
    It displays buttons and allow the user to select words and calls the necessary functions
    to make a prediction after importing a model
    '''

    st.title("Sign detection")
    banner_image = "sign_banner.jpg"
    st.image(banner_image, use_column_width=True)
    st.markdown("""
    <div style="display: flex; justify-content: center; align-items: center;">
        <h2>Select a word and learn its sign!</h2>
    </div>
    """, unsafe_allow_html=True)

    classes = ['Select a sign', 'beer','bye','drink','go','hello','love','many','no','thank you','what','work','world','yes','you']

    video_urls = {
        'beer': [f'{VIDEO_PATH}05707.mp4', f'{VIDEO_PATH}05708.mp4'],
        'bye': [f'{VIDEO_PATH}08512.mp4', f'{VIDEO_PATH}08517.mp4'],
        'drink': [f'{VIDEO_PATH}69302.mp4', f'{VIDEO_PATH}65539.mp4'],
        'go': [f'{VIDEO_PATH}24946.mp4', f'{VIDEO_PATH}24941.mp4'],
        'hello': [f'{VIDEO_PATH}27184.mp4', f'{VIDEO_PATH}27172.mp4'],
        'love': [f'{VIDEO_PATH}34123.mp4', f'{VIDEO_PATH}34124.mp4'],
        'many': [f'{VIDEO_PATH}69396.mp4', f'{VIDEO_PATH}34824.mp4'],
        'no': [f'{VIDEO_PATH}69411.mp4', f'{VIDEO_PATH}38525.mp4'],
        'thank you': [f'{VIDEO_PATH}69502.mp4', f'{VIDEO_PATH}66598.mp4'],
        'what': [f'{VIDEO_PATH}69531.mp4', f'{VIDEO_PATH}62968.mp4'],
        'work': [f'{VIDEO_PATH}63790.mp4', f'{VIDEO_PATH}63789.mp4'],
        'world': [f'{VIDEO_PATH}63836.mp4', f'{VIDEO_PATH}63837.mp4'],
        'yes': [f'{VIDEO_PATH}69546.mp4', f'{VIDEO_PATH}64287.mp4'],
        'you': [f'{VIDEO_PATH}69547.mp4', f'{VIDEO_PATH}64385.mp4']
    }

    chosen_word = st.selectbox("Choose a word:",classes, index = 0, label_visibility = 'collapsed')
    model = load_model()

    # if st.button("Show videos"):
    if chosen_word in video_urls:
        display_videos_for_word(chosen_word, video_urls)
        # video_url = video_urls[chosen_word]
        # st.video(video_url)

    st.write("Upload a video file:")
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        st.video(uploaded_file)

    if st.button("What is this sign?"):
        X_coord = preprocess_video(uploaded_file)

        if model is not None:
            st.write('**Model running**')
            st.write('**✅ Sign detected**')
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

        st.write(f'Confidence : {max_probability}')

    # if st.button("Draw landmarks"):
    #     video = annotate_video()


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
