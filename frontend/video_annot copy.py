# import numpy as np
# import streamlit as st
# import cv2
# import av
# import queue
# import time
# from streamlit_webrtc import webrtc_streamer

# from backend.ml_logic.model import detect_landmarks, mediapipe_video_to_coord
# from backend.ml_logic.registry import draw_landmarks, load_model


# frame_buffer = []


# mapping = {
#     'I': 0,
#     'beer': 1,
#     'bye': 2,
#     'drink': 3,
#     'go': 4,
#     'hello': 5,
#     'love': 6,
#     'many': 7,
#     'no': 8,
#     'thankyou': 9,
#     'what': 10,
#     'work': 11,
#     'world': 12,
#     'yes': 13,
#     'you': 14
# }
# mapping = {v: k for k, v in mapping.items()}

# result_queue: "queue.Queue[List[Detection]]" = queue.Queue() #type:ignore

###############################################################

# def video_frame_callback(frame):

#     frame = frame.to_ndarray(format='bgr24')

#     results = detect_landmarks(frame)
#     annotated_image = draw_landmarks(results, frame)

#     print('1) Started process_and_display')
#     global frame_buffer
#     frame_buffer.append(frame)

#     if len(frame_buffer) == 20:
#         print('2) Started process_frames')

#         resized_frames = [cv2.resize(frame, (480, 480)) for frame in frame_buffer]

#         frames_array = np.array([resized_frames]) # 1, 20, 480, 480, 3

#         X_coord = mediapipe_video_to_coord(frames_array)

#         prediction = model.predict(X_coord) #type:ignore
#         max_index = np.argmax(prediction)
#         predicted_word = mapping[max_index]

#         st.write(predicted_word)
#         result_queue.put(predicted_word)

#         frame_buffer = []

#     return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

################################################################
# def mediapipe_video_to_coord(X):
#     X_coord = []

#     for video_count in range(len(X)):
#         video = X[video_count]
#         video_frames = []

#         for frame_count in range(FRAMES_PER_VIDEO):
#             start = time.time()
#             frame = video[frame_count]
#             results = detect_landmarks(frame)
#             print('Frame Count:', frame_count, 'Landmark:', time.time() - start)
#             middle = time.time()
#             X_frame = normalized_coordinates_per_frame(results)
#             print('Normalize:', time.time() - middle)
#             video_frames.append(X_frame)

#         X_coord.append(video_frames)

#     X_coord = tf.convert_to_tensor(X_coord)
#     return X_coord
# model = load_model()


# def frames_to_predicton(frames):
#     frames_resized = [cv2.resize(frame, (480, 480)) for frame in frames]
#     frames_resized = np.expand_dims(np.array(frames_resized), axis=0)
#     print(frames_resized.shape)
#     X_coord = mediapipe_video_to_coord(frames_resized)
#     start = time.time()
#     prediction = model.predict(X_coord)[0] #type:ignore
#     print(time.time() - start)
#     max_index = np.argmax(prediction)
#     word_detected = mapping[max_index]

#     return word_detected


# def video_frame_callback2(frame):
#     #annotate the frame
#     frame = frame.to_ndarray(format="bgr24")
#     results = detect_landmarks(frame)
#     annotated_image = draw_landmarks(results, frame)
#     # annotated_image = frame
#     # Accumulate frames for 2 seconds (20 frames)
#     global frame_buffer
#     frame_buffer.append(frame)
#     if len(frame_buffer) == 20:
#         print("------------- AI running.... -------------")
#         # print(np.array(frame_buffer).shape)
#         word_detected = frames_to_predicton(frame_buffer)
#         print(word_detected)
#         # st.session_state["prediction"] = word_detected
#         # result_queue.put(word_detected)

#         frame_buffer = []

#         # with lock:
#         #     prediction_list.append(word_detected)
#         #     prediction = word_detected
#     return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")


# # Streamlit
# def main():
#     # if 'model' not in st.session_state:
#     #     st.session_state['model'] = load_model()
#     ctx = webrtc_streamer(key="example",
#                           video_frame_callback=video_frame_callback2)

#     # ctx = webrtc_streamer(key="example",
#     #                       video_frame_callback=video_frame_callback2)

#     # if ctx.state.playing:
#     #     result = ""
#     #     prediction_placeholder = st.empty()
#     #     while True:
#     #         time.sleep(0.5)
#     #         result += result_queue.get() + " → "
#     #         prediction_placeholder.write(result)

# if __name__ == "__main__":
#     main()

##############################################################################

import numpy as np
import streamlit as st
import cv2
import av
import queue
import time
from streamlit_webrtc import webrtc_streamer
import mediapipe as mp

from backend.ml_logic.model import detect_landmarks, mediapipe_video_to_coord
from backend.ml_logic.registry import draw_landmarks, load_model

frame_buffer = []
start = time.time()

def video_frame_callback(frame):
    frame = frame.to_ndarray(format="bgr24")

    mp_drawing = mp.solutions.drawing_utils # type: ignore
    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)

    mp_holistic = mp.solutions.holistic # type: ignore

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0, #Light model
        ) as holistic:

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw pose, left and right hands landmarks on the frame.
        mp_drawing.draw_landmarks(image,
                                    results.right_hand_landmarks,
                                    mp_holistic.HAND_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=2),
                                    mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                    )

        # 3. Left Hand
        mp_drawing.draw_landmarks(image,
                                    results.left_hand_landmarks,
                                    mp_holistic.HAND_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=2),
                                    mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                    )

        # 4. Pose Detections
        mp_drawing.draw_landmarks(image,
                                    results.pose_landmarks,
                                    mp_holistic.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                    )

    global frame_buffer
    frame_buffer.append(frame)
    if (time.time() - start) >= 60:
    # if len(frame_buffer) == 1:
        print(len(frame_buffer))
        print(len(frame_buffer) / 60)
        # print(time.time() - start) # takes about 6 seconds to collect 20 frames

    return av.VideoFrame.from_ndarray(image, format="bgr24")


# Streamlit
def main():

    webrtc_streamer(key="example",
                    video_frame_callback=video_frame_callback)

if __name__ == "__main__":
    main()
