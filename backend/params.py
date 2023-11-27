import os

##################  VARIABLES  ##################
# Class definition
# Available classese: ['hello', 'bye', 'world', 'yes', 'no', 'I', 'you', 'go',
#                      'work', 'drink', 'beer', 'many', 'what', 'thankyou', 'love']

SELECTED_WORDS = ['hello', 'bye', 'world', 'yes', 'no', 'I', 'you', 'go', 'work', 'drink', 'beer', 'many', 'what', 'thankyou', 'love']
N_CLASSES = len(SELECTED_WORDS)
# Frame sampling parameters
FRAMES_PER_VIDEO = 20
TARGET_SIZE = (480, 480)
# Dataset multiplier
NUMBER_OF_AUGMENTATIONS = 1
# Train split parameters
TRAIN_SIZE = 0.7

###############  GCP VARIABLES  #################
MODEL_TARGET = os.environ.get("MODEL_TARGET")
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_REGION = os.environ.get("GCP_REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
INSTANCE = os.environ.get("INSTANCE")


#################  CONSTANTS  ####################
MAIN_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data', '')
VIDEO_PATH = os.path.join(MAIN_PATH, 'videos', '')
CUSTOM_VIDEO_PATH = os.path.join(MAIN_PATH, 'custom_videos', '')
LOCAL_REGISTRY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
