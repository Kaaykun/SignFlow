import cv2
import numpy as np
import time
import mediapipe as mp

from backend.ml_logic.model import detect_landmarks, mediapipe_video_to_coord
from backend.ml_logic.registry import draw_landmarks, load_model

model = load_model()

def frames_to_predicton(frames):
    start = time.time()
    frames_resized = [cv2.resize(frame, (480, 480)) for frame in frames]
    frames_resized = np.array(frames_resized)
    # frames_resized = np.expand_dims(frames_resized, axis=0)
    print('start to coord', start)
    # X_coord = mediapipe_video_to_coord(frames_resized)
    mp_holistic = mp.solutions.holistic # type: ignore

    print('finish to coord', time.time() - start)

    # prediction = model.predict(X_coord)[0] #type:ignore
    # max_index = np.argmax(prediction)
    # word_detected = mapping[max_index]

    # return max_index

def main(frame):
    cap = cv2.VideoCapture(frame)

    # Get video details
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out = cv2.VideoWriter(filename='test.mp4', fourcc=fourcc, fps=20, frameSize=(frame_width, frame_height))

    if not out.isOpened():
        print("Could not open output video file for writing.")

    frame_buffer = []
    BG_COLOR = (192, 192, 192)
    mp_drawing = mp.solutions.drawing_utils # type: ignore
    mp_drawing_styles = mp.solutions.drawing_styles # type: ignore
    mp_holistic = mp.solutions.holistic # type: ignore

    while cap.isOpened():
        print("------")
        ret, frame = cap.read()
        if not ret:

            break


        with mp_holistic.Holistic(
            static_image_mode=True,
            model_complexity=0,
            enable_segmentation=True,
            refine_face_landmarks=True) as holistic:

            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            annotated_image = frame.copy()
            # Draw segmentation on the frame.
            # To improve segmentation around boundaries, consider applying a joint
            # bilateral filter to "results.segmentation_mask" with "frame".
            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
            bg_image = np.zeros(frame.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR
            annotated_image = np.where(condition, annotated_image, bg_image)
            # Draw pose, left and right hands, and face landmarks on the frame.
            mp_drawing.draw_landmarks(
                annotated_image,
                results.face_landmarks,
                mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.
                get_default_pose_landmarks_style())
        # Process the frame for landmark detection
        # results = detect_landmarks(frame)

        # # Draw landmarks on the frame
        # annotated_image = draw_landmarks(results, frame)

        # frame_buffer.append(frame)
        # if len(frame_buffer) == 20:
        #     print("------------- AI running.... -------------")
        #     # print(np.array(frame_buffer).shape)

        #     word_detected = frames_to_predicton(frame_buffer)
        #     # st.session_state["prediction"] = word_detected

        #     frame_buffer = []

        # Write the frame with annotations to the output video
        out.write(annotated_image)
        cv2.imshow('title', annotated_image)

        # cv2.imshow('Video', annotated_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = 'backend/data/custom_videos/beer_Benjamin_1.mp4'
    main(video_path)
