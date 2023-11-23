import streamlit as st
import cv2
import numpy as np

# Load your pre-existing model
# model = load_model()

# video preprocessing
# def process_video(video_file):
#     # Replace the following lines with code to process the video using your model
#     cap = cv2.VideoCapture(video_file.name)
#     frames = []
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         # Perform processing on the frame using your model
#         # processed_frame = model.predict(frame)
#         frames.append(frame)
#     cap.release()
#     return frames

# Initialize empty array of desired shape
X = np.empty((1, 10, *(150, 150), 3), dtype=np.uint8)

# Function to perform frame sampling
def sample_frames(video_path, frames_per_video, total_frames):
    frames = []
    cap = cv2.VideoCapture(video_path)

    frame_indices = []

    while len(set(frame_indices)) != frames_per_video:
        frame_indices = sorted(np.random.uniform(0, total_frames-5, frames_per_video).astype(int))

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
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Append to list of frames
                frames.append(frame_rgb)

            frame_counter += 1

            if len(frames) == frames_per_video:
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

        video_path = '../backend/data/videos/00335.mp4'
        # Process the video using the model
        frames = sample_frames(video_path, 10, 20)

        # Display the processed frames (you can customize this part based on your needs)
        for frame in frames:
            st.image(frame, channels="BGR", caption="Processed Frame")

if __name__ == "__main__":
    main()
