import os

##################  VARIABLES  ##################
# Class definition
SELECTED_WORDS = ['work', 'study', 'write', 'hot', 'cold', 'family']
N_CLASSES = len(SELECTED_WORDS)
# Frame sampling parameters
FRAMES_PER_VIDEO = 20
TARGET_SIZE = (512, 512)
# Dataset multiplier
NUMBER_OF_AUGMENTATIONS = 3
# Train split parameters
TRAIN_SIZE = 0.7

###############  GCP VARIABLES  #################
# DATA_SIZE = os.environ.get("DATA_SIZE")
MODEL_TARGET = os.environ.get("MODEL_TARGET")
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_REGION = os.environ.get("GCP_REGION")
# BQ_DATASET = os.environ.get("BQ_DATASET")
# BQ_REGION = os.environ.get("BQ_REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
# INSTANCE = os.environ.get("INSTANCE")


#################  CONSTANTS  ####################
MAIN_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data', '')
VIDEO_PATH = os.path.join(MAIN_PATH, 'videos', '')
CUSTOM_VIDEO_PATH = os.path.join(MAIN_PATH, 'custom_videos', '')
LOCAL_REGISTRY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
