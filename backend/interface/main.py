import os
# Suppress WARNING, INFO, and DEBUG messages related to tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.model_selection import train_test_split

from backend.params import TRAIN_SIZE
from backend.ml_logic.data import load_wlas_df, load_features_df, load_selected_df, load_custom_features_df, load_custom_selected_df
from backend.ml_logic.preprocessor import create_X, create_custom_X
from backend.ml_logic.encoders import categorize_y
from backend.ml_logic.augment import augment_data
from backend.ml_logic.registry import test_shape_X_y, test_shape_X_y_val, test_shape_X_y_aug, save_model
from backend.ml_logic.model import mediapipe_video_to_coord, train_model

def main():
    # Load dataframes from backend.ml_logic.data (when using original videos)
    #wlas_df = load_wlas_df()
    #features_df = load_features_df(wlas_df)
    #selected_df, input_length = load_selected_df(features_df)

    # Load dataframes from backend.ml_logic.data (when using custom videos)
    features_df = load_custom_features_df()
    selected_df, input_length = load_custom_selected_df(features_df)

    # Create feature matrix X and categorical labels y_cat
    #X = create_X(selected_df, input_length)
    X = create_custom_X(selected_df, input_length)
    y_cat = categorize_y(selected_df)

    # Check for correct shapes of X, y_cat
    test_shape_X_y(selected_df, input_length, X, y_cat)

    # Split dataset into training and validation sets
    X_train, X_val, y_cat_train, y_cat_val = train_test_split(X,
                                                            y_cat,
                                                            train_size=TRAIN_SIZE,
                                                            random_state=1,
                                                            stratify=y_cat)

    # Check for correct shapes of X_val, y_cat_val
    test_shape_X_y_val(X, X_val, y_cat, y_cat_val)

    # Augment training data
    X_aug, y_aug = augment_data(X_train, y_cat_train)

    # Check for correct shapes of X_aug, y_aug
    test_shape_X_y_aug(X_aug, X_train, y_aug, y_cat_train)

    # Convert augmented and validation data to coordinates
    X_aug_coord = mediapipe_video_to_coord(X_aug)
    X_val_coord = mediapipe_video_to_coord(X_val)

    # Train the model using coordinates
    model = train_model(X_aug_coord, X_val_coord, y_aug, y_cat_val)

    # Save the trained model
    save_model(model)

if __name__ == '__main__':
    main()
