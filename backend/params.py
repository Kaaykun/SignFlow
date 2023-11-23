import os


##################  VARIABLES  ##################
# Class definition
SELECTED_WORDS = ['work','study', 'write', 'hot', 'cold', 'family']
N_CLASSES = len(SELECTED_WORDS)
# Frame sampling parameters
FRAMES_PER_VIDEO = 10
TARGET_SIZE = (150, 150)
# Dataset multiplier
NUMBER_OF_AUGMENTATIONS = 3
# Train split parameters
TRAIN_SIZE = 0.7


##################  CONSTANTS  #####################
MAIN_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data', '')
VIDEO_PATH = os.path.join(MAIN_PATH, 'videos', '')
