import os
# Suppress WARNING, INFO, and DEBUG messages related to tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

from backend.params import N_CLASSES


def categorize_y(selected_df):
    """
    Categorize labels from the 'word' column in selected DataFrame.

    Parameters:
    - selected_df (pandas.DataFrame): DataFrame containing selected video information.

    Returns:
    - numpy.ndarray: Categorized labels in one-hot encoded form.
    """
    label_encoder = LabelEncoder()

    selected_df['encoded_word'] = label_encoder.fit_transform(selected_df['word'])
    y_cat = tf.keras.utils.to_categorical(selected_df['encoded_word'], num_classes=N_CLASSES)

    return y_cat
