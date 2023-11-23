import numpy as np
import cv2

from params import TARGET_SIZE, FRAMES_PER_VIDEO, NUMBER_OF_AUGMENTATIONS


def augment_frame_params(frame_width, frame_height):
    """
    Generate random parameters for image augmentation.

    Args:
    - frame_width (int): Width of the frame.
    - frame_height (int): Height of the frame.

    Returns:
    Variables containing randomly generated parameters for frame augmentation.
        - angle (float): Random rotation between 0 and 15 degrees.
        - flip (float): Random value for horizontal mirroring.
        - x_trans (int): Random translation along the x-axis between -20 and 20 pixels.
        - y_trans (int): Random translation along the y-axis between -20 and 20 pixels.
        - scale (float): Random zoom factor between 0.8 and 1.2.
        - crop_size (int): Random size for cropping within the frame.
        - alpha (float): Random value for brightness and contrast adjustment between 0.7 and 1.3.
        - beta (int): Random value for brightness and contrast adjustment between -20 and 20.
    """
    # Random rotation between -15 and 15 degrees
    angle = np.random.uniform(0, 15)
    # Random horizontal mirroring
    flip = np.random.rand()
    # Random translation
    x_trans = np.random.randint(-20, 20)
    y_trans = np.random.randint(-20, 20)
    # Random zoom
    scale = np.random.uniform(0.8, 1.2)
    # Random cropping (with centralized region)
    crop_size = np.random.randint(0.8 * min(frame_width, frame_height), min(frame_width, frame_height))
    # Changes in brightness, contrast, and saturation
    alpha = np.random.uniform(0.7, 1.3)
    beta = np.random.randint(-20, 20)

    return angle, flip, x_trans, y_trans, scale, crop_size, alpha, beta


def augment_frame(frame, angle, flip, x_trans, y_trans, scale, crop_size, alpha, beta):
    """
    Apply various random transformations to an input image/frame.

    Args:
    - frame (numpy.ndarray): Input image/frame to be augmented.
    - angle (float): Angle for random rotation.
    - flip (float): Value for horizontal flipping (50% chance).
    - x_trans (int): Random translation along the x-axis.
    - y_trans (int): Random translation along the y-axis.
    - scale (float): Random zoom factor.
    - crop_size (int): Random size for cropping within the frame.
    - alpha (float): Value for brightness and contrast adjustment.
    - beta (int): Value for brightness and contrast adjustment.

    Returns:
    numpy.ndarray: Augmented image/frame after applying random transformations.
    """
    # Random rotation by an angle
    rows, cols, _ = frame.shape
    M_rot = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    frame = cv2.warpAffine(frame, M_rot, (cols, rows))
    # Horizontal flipping
    if flip > 0.5:  # 50% chance of flipping
        frame = cv2.flip(frame, 1)
    # Random translation
    M_trans = np.float32([[1, 0, x_trans], [0, 1, y_trans]]) # type: ignore
    frame = cv2.warpAffine(frame, M_trans, (cols, rows)) # type: ignore
    # Random zoom
    frame = cv2.resize(frame, None, fx=scale, fy=scale)
    # Random cropping (with centralized region)
    x = int((rows - crop_size) / 2)
    y = int((cols - crop_size) / 2)
    frame = frame[x:x + crop_size, y:y + crop_size]
    # Changes in brightness, contrast, and saturation
    frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    # Resize frame back to height 150, width 150
    frame = cv2.resize(frame, TARGET_SIZE)

    return frame


def multiply_data(X, frames_per_video):
    X_temp = np.empty((len(X), frames_per_video, *TARGET_SIZE, 3), dtype=np.uint8)
    frame_height = X.shape[2]
    frame_width = X.shape[3]

    for i in range(len(X)):
        angle, flip, x_trans, y_trans,\
        scale, crop_size, alpha, beta = augment_frame_params(frame_height, frame_width)
        for j in range(frames_per_video):
            sampled_frame = X[i][j]
            aug_frame = augment_frame(sampled_frame,
                                      angle,
                                      flip,
                                      x_trans,
                                      y_trans,
                                      scale,
                                      crop_size,
                                      alpha,
                                      beta)
            X_temp[i][j] = aug_frame

    return X_temp


def augment_data(X_train, y_cat_train):
    X_aug = X_train.copy()
    y_aug = y_cat_train.copy()

    # Multiply dataset by defined param
    for _ in range(NUMBER_OF_AUGMENTATIONS):
        X_temp = multiply_data(X_train, FRAMES_PER_VIDEO)
        # Returns X_aug with shape (n * 219, 10, 150, 150, 3)
        X_aug = np.concatenate((X_aug, X_temp), axis=0)
        # Returns y_aug with shape (n * 219, 20)
        y_aug = np.concatenate((y_aug, y_cat_train), axis=0)

    return X_aug, y_aug
