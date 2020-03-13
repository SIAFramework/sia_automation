"""
Created on Fri Nov  8 12:15:19 2019
@author: Yashwanth Ramachandra
"""

import pandas as pd


"""
Merge themes data with their mapping data
"""
def cluster_theme_keywords(theme_data, themes_map_data):
    theme_clustered_data = pd.merge(theme_data, themes_map_data, on='themes_keyword', how='left')

    return theme_clustered_data


"""
Merge emotions data with their mapping data
"""
def cluster_emotion_keywords(emotion_data, emotion_map_data):
    emotion_clustered_data = pd.merge(emotion_data, emotion_map_data, on='emotion_keyword', how='left')

    return emotion_clustered_data