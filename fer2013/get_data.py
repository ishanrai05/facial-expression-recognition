import os
import pandas as pd
import math
import numpy as np
from utils import  get_dimensions


def face_array(df_initial):
    faces = []
    pixels = df_initial['pixels'].tolist()
    width, height = get_dimensions(df_initial)

    for sequence in pixels:
        face = [int(pixel) for pixel in sequence.split()]
        face = np.asarray(face).reshape(width, height)
        faces.append(face.astype(np.uint8))
    faces = np.asarray(faces)
    return faces

def get_data(emotion_type_dict, base_dir):
    df_initial = pd.read_csv(base_dir)


    df_initial['EmotionType'] = df_initial['emotion'].map(emotion_type_dict.get)


    df_train = df_initial[df_initial['Usage']=='Training']
    df_test = df_initial[df_initial['Usage']=='PublicTest']
    df_val = df_initial[df_initial['Usage']=='PrivateTest']

    return df_train, df_test, df_val, df_initial


def balance_data(df_train):

    data_count = df_train['emotion'].value_counts()
    max_data = max(data_count)
    data_aug_rate = list()
    for i in range(len(data_count)):
        data_aug_rate.append(int(round(max_data/data_count[i]))-1)

    # Copy fewer class to balance the number of 7 classes
    for i in range(7):
        if data_aug_rate[i]:
            df_train=df_train.append([df_train.loc[df_train['emotion'] == i,:]]*(data_aug_rate[i]), ignore_index=True)
    df_train['emotion'].value_counts()
    return df_train