from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

from backend.params import N_CLASSES


def categorize_y(selected_df, input_length):
    """
    Categorize labels from the 'word' column in selected DataFrame.

    Parameters:
    - selected_df (pandas.DataFrame): DataFrame containing selected video information.
    - input_length (int): Length of the selected DataFrame.

    Returns:
    - numpy.ndarray: Categorized labels in one-hot encoded form.
    """
    label_encoder = LabelEncoder()

    selected_df['encoded_word'] = label_encoder.fit_transform(selected_df['word'])
    y_cat = tf.keras.utils.to_categorical(selected_df['encoded_word'], num_classes=N_CLASSES)

    return y_cat
