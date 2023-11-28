import cv2
import numpy as np
import mediapipe as mp
import streamlit as st

from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
from backend.ml_logic.registry import load_model

# Initialize MediaPipe models
def mediapipe_detection(frame, model):
    mp_holistic = mp.solutions.holistic # type: ignore

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, static_image_mode=False) as holistic:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
        frame.flags.writeable = False                  # Image is no longer writeable
        results = model.process(frame)                 # Make prediction
        frame.flags.writeable = True                   # Image is now writeable
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
        return frame, results

def draw_styled_landmarks(frame, results):
    mp_holistic = mp.solutions.holistic # type: ignore
    mp_drawing = mp.solutions.drawing_utils # type: ignore

    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             )
    # Draw left hand connections
    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             )
    # Draw right hand connections
    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3) #, res.visibility
    #face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh]) #, face

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

    return output_frame



# Define Streamlit-WebRTC video transformer
class VideoTransformer(VideoTransformerBase):
    mp_holistic = mp.solutions.holistic # Holistic model
    mp_drawing = mp.solutions.drawing_utils # Drawing utilities
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
# 1. New detection variables
    sequence = []
    sequence_length = 30
    sentence = []
    predictions = []
    actions = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
    threshold = 0.50
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(1)

    #cap.set(3,640) # adjust width
    #cap.set(4,480) # adjust height
    # Set mediapipe model
    model = load_model()

    #with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        #print(results)

        # Draw landmarks
        draw_styled_landmarks(image, results)

        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-sequence_length:]

        if len(sequence) == sequence_length:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            #print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))


        #3. Viz logic
            if len(np.unique(predictions[-5:]))==1:
            #np.unique(predictions[-10:])[0]==np.argmax(res):
                if res[np.argmax(res)] > threshold:

                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5:
                sentence = sentence[-5:]

            # Viz probabilities
            colors = [(245,117,16), (117,245,16), (16,117,245), (20,16,117), (20,16,117), (20,16,117), (20,16,117), (20,16,117), (20,16,117), (20,16,117)]
            image = prob_viz(res, actions, image, colors)

        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

# Streamlit app
def main():
    st.title("Streamlit-WebRTC with OpenCV and MediaPipe")
    


if __name__ == "__main__":
    main()








# import streamlit as st
# import cv2
# import numpy as np
# import mediapipe as mp
# import tensorflow as tf

# from backend.ml_logic.preprocessor import sample_frames
# from backend.ml_logic.registry import load_model
# from backend.params import FRAMES_PER_VIDEO, TARGET_SIZE, VIDEO_PATH

# mp_holistic = mp.solutions.holistic

# def detect_landmarks(frame):
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, static_image_mode=True) as holistic:
#         results = holistic.process(frame_rgb)
#     return results

# def get_coord_for_R_hand(results):
#     x_coord_R_hand = []
#     y_coord_R_hand = []
#     z_coord_R_hand = []

#     if results.right_hand_landmarks:
#         #append coordinates
#         for landmark in results.right_hand_landmarks.landmark:
#             if landmark:
#                 x_coord_R_hand.append(landmark.x)
#                 y_coord_R_hand.append(landmark.y)
#                 z_coord_R_hand.append(landmark.z)
#             else:
#                 x_coord_R_hand.append(0)
#                 y_coord_R_hand.append(0)
#                 z_coord_R_hand.append(0)
#         print("✅ Right hand detected")

#     else:
#         for index in range(21):
#             x_coord_R_hand.append(0)
#             y_coord_R_hand.append(0)
#             z_coord_R_hand.append(0)
#         print("❌ Right hand not detected")

#     return x_coord_R_hand, y_coord_R_hand, z_coord_R_hand

# def get_coord_for_L_hand(results):
#     x_coord_L_hand = []
#     y_coord_L_hand = []
#     z_coord_L_hand = []

#     if results.left_hand_landmarks:
#         # Append coordinates
#         for landmark in results.left_hand_landmarks.landmark:
#             if landmark:
#                 x_coord_L_hand.append(landmark.x)
#                 y_coord_L_hand.append(landmark.y)
#                 z_coord_L_hand.append(landmark.z)
#             else:
#                 x_coord_L_hand.append(0)
#                 y_coord_L_hand.append(0)
#                 z_coord_L_hand.append(0)
#         print("✅ Left hand detected")
#     else:
#         # No left hand detected, populate with NaN values
#         for index in range(21):
#             x_coord_L_hand.append(0)
#             y_coord_L_hand.append(0)
#             z_coord_L_hand.append(0)
#         print("❌ Left hand not detected")

#     return x_coord_L_hand, y_coord_L_hand, z_coord_L_hand

# def get_coord_for_pose(results):
#     x_coord_pose = []
#     y_coord_pose = []
#     z_coord_pose = []

#     if results.pose_landmarks:
#         # loop over all landmarks for each hand
#         for landmark in results.pose_landmarks.landmark:
#             if landmark:
#                 x_coord_pose.append(landmark.x)
#                 y_coord_pose.append(landmark.y)
#                 z_coord_pose.append(landmark.z)
#                 # print(f"Landmark X: {landmark.x}, Y: {landmark.y}, Z: {landmark.z}")
#             else:
#                 x_coord_pose.append(0)
#                 y_coord_pose.append(0)
#                 z_coord_pose.append(0)
#         print("✅ Pose detected")

#     else:
#         x_coord_pose = [0] * 33
#         y_coord_pose = [0] * 33
#         z_coord_pose = [0] * 33
#         print("❌ Pose not detected")

#     x_coord_pose = x_coord_pose[11:25]
#     y_coord_pose = y_coord_pose[11:25]
#     z_coord_pose = z_coord_pose[11:25]

#     return x_coord_pose, y_coord_pose, z_coord_pose

# def coordinates_per_frame(results):
#     x_coord_R_hand, y_coord_R_hand, z_coord_R_hand = get_coord_for_R_hand(results)
#     x_coord_L_hand, y_coord_L_hand, z_coord_L_hand = get_coord_for_L_hand(results)
#     x_coord_pose, y_coord_pose, z_coord_pose = get_coord_for_pose(results)

#     X_frame = x_coord_R_hand + y_coord_R_hand + z_coord_R_hand + \
#           x_coord_L_hand + y_coord_L_hand + z_coord_L_hand + \
#           x_coord_pose + y_coord_pose + z_coord_pose
#     return X_frame

# def mediapipe_video_to_coord(X):
#     X_coord = []

#     for video_count in range(len(X)):
#         video = X[video_count]
#         video_frames = []

#         for frame_count in range(10):
#             frame = video[frame_count]
#             results = detect_landmarks(frame)
#             X_frame = coordinates_per_frame(results)
#             video_frames.append(X_frame)

#         X_coord.append(video_frames)

#     X_coord = tf.convert_to_tensor(X_coord)
#     return X_coord

# # Function to perform frame sampling
# def sample_frames(video_path, total_frames):
#     frames = []
#     cap = cv2.VideoCapture(video_path)

#     frame_indices = []

#     while len(set(frame_indices)) != FRAMES_PER_VIDEO:
#         frame_indices = sorted(np.random.uniform(0, total_frames-5, FRAMES_PER_VIDEO).astype(int))

#     frame_counter = 0

#     try:
#         while cap.isOpened():
#             ret, frame = cap.read()

#             if not ret:
#                 break

#             if frame_counter in frame_indices:
#                 # Resize frame to required size
#                 frame = cv2.resize(frame, (150, 150))
#                 # CV2 output BGR -> converting to RGB
#                 #frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#                 # Append to list of frames
#                 frames.append(frame)

#             frame_counter += 1

#             if len(frames) == FRAMES_PER_VIDEO:
#                 break

#     finally:
#         cap.release()

#     return frames

# def create_X(selected_df, input_length):
#     """
#     Create an array X for sampled frames from selected videos.

#     Parameters:
#     - selected_df (pandas.DataFrame): DataFrame containing selected video information.
#     - input_length (int): Length of the selected DataFrame.

#     Returns:
#     - numpy.ndarray: An array containing sampled frames from selected videos.
#     """

#     np.random.seed(9)

#     X = np.empty((1, FRAMES_PER_VIDEO, *TARGET_SIZE, 3), dtype=np.uint8)

#     for i, row in selected_df.iterrows():
#         video_id = row['video_id']
#         total_frames = row['video_length']
#         video_path = f'{VIDEO_PATH}{video_id}.mp4'

#         sampled_frames = sample_frames(video_path, total_frames)

#         X[i] = np.array(sampled_frames)

#     return X

# # Streamlit app
# def main():
#     st.title("Sign Flow")
#     st.markdown('Welcome to our project streamlit!')

#     X = np.empty((1, FRAMES_PER_VIDEO, *TARGET_SIZE, 3), dtype=np.uint8)

#     # Allow user to upload a video file
#     uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

#     if uploaded_file is not None:
#         # Display the uploaded video
#         st.video(uploaded_file)

#         video_path = f'{VIDEO_PATH}20982.mp4'
#         # Process the video using the model
#         frames = sample_frames(video_path, 36)
#         X[0] = np.array(frames)
#         X_coord = mediapipe_video_to_coord(X)


#         model = load_model()
#         if model is not None:
#             st.write('Model loaded')
#         else:
#             st.write('Failed to load the model')

#         st.write(model.predict(X_coord))


#         # Display the processed frames (you can customize this part based on your needs)
#         for frame in frames:
#             st.image(frame, channels="BGR", caption="Processed Frame")

# if __name__ == "__main__":
#     main()
