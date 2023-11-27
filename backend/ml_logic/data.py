import os
# Suppress WARNING, INFO, and DEBUG messages related to tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import cv2
import os

from backend.params import MAIN_PATH, SELECTED_WORDS, CUSTOM_VIDEO_PATH


def get_videos_ids(json_list):
    """
    Extracts video IDs from a list of JSON entries, considering video existence in the directory.

    Parameters:
    - json_list (list): A list of dictionaries containing video information.

    Returns:
    - list: A list of video IDs that correspond to existing video files in the specified directory.
    """
    videos_list = []
    for ins in json_list:
        video_id = ins['video_id']
        if os.path.exists(f'{MAIN_PATH}videos/{video_id}.mp4'):
            videos_list.append(video_id)
    return videos_list


def get_json_features(json_list):
    """
    Extracts video IDs and URLs from a list of JSON entries based on existing video files.

    Parameters:
    - json_list (list): A list of dictionaries containing video information.

    Returns:
    - tuple: A tuple containing two lists - the first list contains video IDs of existing videos,
             and the second list contains URLs corresponding to those existing videos.
    """
    videos_ids = []
    videos_urls = []
    for ins in json_list:
        video_id = ins['video_id']
        video_url = ins['url']
        if os.path.exists(f'{MAIN_PATH}videos/{video_id}.mp4'):
            videos_ids.append(video_id)
            videos_urls.append(video_url)
    return videos_ids, videos_urls


def load_wlas_df():
    """
    Process video data from JSON and update DataFrame with video IDs.

    Returns:
    - pandas.DataFrame: A DataFrame updated with a 'videos_ids' column based on existing video files.
    """
    wlas_df = pd.read_json(MAIN_PATH + 'WLASL_v0.3.json')
    wlas_df['videos_ids'] = wlas_df['instances'].apply(get_videos_ids)
    return wlas_df


def load_features_df(wlas_df):
    """
    Load features from a DataFrame and generate a new DataFrame with word, video ID, and URL.

    Parameters:
    - wlas_df (pandas.DataFrame): DataFrame containing video information.

    Returns:
    - pandas.DataFrame: A new DataFrame containing word, video ID, and URL columns.
    """
    features_df = pd.DataFrame(columns=['word', 'video_id', 'url'])

    for row in wlas_df.iterrows():
        ids, urls = get_json_features(row[1][1])
        word = [row[1][0]] * len(ids)
        df = pd.DataFrame(list(zip(word, ids, urls)), columns = features_df.columns)
        features_df = pd.concat([features_df, df], ignore_index=True)

    features_df.index.name = 'index'
    return features_df


def load_custom_features_df():
    """
    Load videos from a data folder and generate a DataFrame with word and video ID.

    Returns:
    - pandas.DataFrame: A new DataFrame containing word and video ID columns.
    """
    features_df = pd.DataFrame(columns=['word', 'video_id'])

    for filename in os.listdir(CUSTOM_VIDEO_PATH):
        word = filename.split('_')[0]
        filename = filename.replace('.mp4', '')
        df = pd.DataFrame([[word, filename]], columns=features_df.columns)
        # Append temporary df to feature_df
        features_df = pd.concat([features_df, df], ignore_index=True)

    return features_df


def load_selected_df(features_df):
    """
    Load selected data from a features DataFrame and compute video lengths.

    Parameters:
    - features_df (pandas.DataFrame): DataFrame containing features information.

    Returns:
    - tuple: A tuple containing two elements:
        1. pandas.DataFrame: Selected DataFrame with updated video length information.
        2. int: Length of the selected DataFrame.
    """
    selected_df = features_df[features_df['word'].isin(SELECTED_WORDS)]

    for video_id in selected_df['video_id']:
        if os.path.exists(f'{MAIN_PATH}videos/{video_id}.mp4'):
            cap = cv2.VideoCapture(f'{MAIN_PATH}videos/{video_id}.mp4')
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            selected_df.loc[selected_df['video_id'] == video_id, ['video_length']] = int(length)
        pass

    selected_df = selected_df.reset_index(drop=True)
    input_length = len(selected_df)
    return selected_df, input_length


def load_custom_selected_df(features_df):
    """
    Load selected data from a features DataFrame and compute video lengths.

    Parameters:
    - features_df (pandas.DataFrame): DataFrame containing features information.

    Returns:
    - tuple: A tuple containing two elements:
        1. pandas.DataFrame: Selected DataFrame with updated video length information.
        2. int: Length of the selected DataFrame.
    """
    selected_df = features_df[features_df['word'].isin(SELECTED_WORDS)]

    for video_id in selected_df['video_id']:
        if os.path.exists(f'{CUSTOM_VIDEO_PATH}{video_id}.mp4'):
            cap = cv2.VideoCapture(f'{CUSTOM_VIDEO_PATH}{video_id}.mp4')
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            selected_df.loc[selected_df['video_id'] == video_id, ['video_length']] = int(length)
        pass

    selected_df = selected_df.reset_index(drop=True)
    input_length = len(selected_df)
    return selected_df, input_length
