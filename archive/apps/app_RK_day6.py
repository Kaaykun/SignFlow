import streamlit as st
import os
import cv2
from backend.params import FRAMES_PER_VIDEO, TARGET_SIZE, VIDEO_PATH, CUSTOM_VIDEO_PATH
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.compositing.concatenate import concatenate_videoclips
from PIL import Image


# # Change background color
# st.markdown(
#     """
#     <style>
#         body {
#             background-color: #D3D3D3;
#         }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # Change text color
# st.markdown(
#     """
#     <style>
#         div.stButton > button {
#             color: white;
#             background-color: #3498db;
#         }
#     </style>
#     """,
#     unsafe_allow_html=True
# )


def video_uploading_page():
    st.title("Video Sign detection")

    # Create a 3x3 grid layout using st.beta_columns
    col1, col2, col3 = st.columns(3)


    video_path = [f'{VIDEO_PATH}27184.mp4',
                    f'{VIDEO_PATH}08510.mp4',
                    f'{VIDEO_PATH}66807.mp4',
                    f'{VIDEO_PATH}34123.mp4',
                    f'{VIDEO_PATH}05705.mp4',
                    f'{VIDEO_PATH}69502.mp4'
        ]
    # Function to display video
    def display_video(video_path):
        st.video(video_path)

    # Button in the first column
    button1_click = col1.button("Hello")
    if button1_click:
        # Call the function to display the video
        display_video(video_path[0])

    # Button in the second column
    button2_click = col1.button("Bye")
    if button2_click:
        # Call the function to display the video
        display_video(video_path[1])

    # Button in the third column
    button3_click = col2.button("World")
    if button3_click:
        # Call the function to display the video
        display_video(video_path[2])

    # Button in the forth column
    button4_click = col2.button("Love")
    if button4_click:
        # Call the function to display the video
        display_video(video_path[3])

    # Button in the fifth column
    button5_click = col3.button("Beer")
    if button5_click:
        # Call the function to display the video
        display_video(video_path[4])

    # Button in the sixth column
    button6_click = col3.button("thank you")
    if button6_click:
        # Call the function to display the video
        display_video(video_path[5])


    #Placeholder previews (replace with your own)

    video_previews = [f'{CUSTOM_VIDEO_PATH}hello_Benjamin_4.mp4',
                      f'{CUSTOM_VIDEO_PATH}bye_Benjamin_4.mp4',
                      f'{CUSTOM_VIDEO_PATH}world_Benjamin_5.mp4',
                      f'{CUSTOM_VIDEO_PATH}love_Benjamin_5.mp4',
                      f'{CUSTOM_VIDEO_PATH}beer_Eigo_3.mp4',
                      f'{CUSTOM_VIDEO_PATH}thankyou_Benjamin_4.mp4'
    ]



    # Display video previews with a smaller size
    for preview in video_previews:
        clip = VideoFileClip(preview)
        frame = clip.get_frame(0)  # Get the first frame as a preview
        small_preview = Image.fromarray((frame * 255).astype('uint8'))
        st.image(small_preview, caption="Video Preview", width=200)

    # Allow the user to upload a new video file
    uploaded_file = st.file_uploader('Upload a new video file', type=['mp4'])

    if uploaded_file is not None:
        # Display the uploaded video preview with a smaller size
        clip = VideoFileClip(uploaded_file)
        frame = clip.get_frame(0)
        small_uploaded_preview = Image.fromarray((frame * 255).astype('uint8'))
        st.image(small_uploaded_preview, caption="Uploaded Video Preview", width=200)
        st.video(uploaded_file)



    # #Display video previews
    # for preview in video_previews:
    #     st.video(preview)


    # uploaded_file = st.file_uploader('Upload a new video file', type=['mp4'])

    # if uploaded_file is not None:
    #     st.video(uploaded_file)


def video_streming_page():
    st.title("Live Sign Detection")

    # Create a VideoCapture object to capture video from the default camera (0)
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    stop_button_pressed = st.button("Stop")
    while cap.isOpened() and not stop_button_pressed:
        ret, frame = cap.read()
        if not ret:
            st.write("Video Capture Ended")
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame,channels="RGB")
        if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed:
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    st.sidebar.title("Pages")
    pages = ["Video Uploading Sign detection", "Live Sign Detection"]
    choice = st.sidebar.radio("Sign Flow", pages)

    if choice == "Video Uploading Sign detection":
        video_uploading_page()
    elif choice == "Live Sign Detection":
        video_streming_page()

if __name__ == "__main__":
    main()
