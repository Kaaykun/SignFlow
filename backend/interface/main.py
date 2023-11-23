import os
# Suppress WARNING, INFO, and DEBUG messages related to tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.model_selection import train_test_split

from backend.params import TRAIN_SIZE
from backend.ml_logic.data import load_wlas_df, load_features_df, load_selected_df
from backend.ml_logic.preprocessor import create_X
from backend.ml_logic.encoders import categorize_y
from backend.ml_logic.augment import augment_data
from backend.ml_logic.registry import test_shape_X_y, test_shape_X_y_val, test_shape_X_y_aug


def main():
    wlas_df = load_wlas_df()

    features_df = load_features_df(wlas_df)

    selected_df, input_length = load_selected_df(features_df)

    X = create_X(selected_df, input_length)
    y_cat = categorize_y(selected_df, input_length)

    test_shape_X_y(selected_df, input_length, X, y_cat)

    X_train, X_val, y_cat_train, y_cat_val = train_test_split(X,
                                                            y_cat,
                                                            train_size=TRAIN_SIZE,
                                                            random_state=1,
                                                            stratify=y_cat)

    test_shape_X_y_val(X, X_val, y_cat, y_cat_val)

    X_aug, y_aug = augment_data(X_train, y_cat_train)

    test_shape_X_y_aug(X_aug, X_train, y_aug, y_cat_train)


if __name__ == '__main__':
    main()
