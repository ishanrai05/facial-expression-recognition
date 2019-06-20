import numpy as np
import math

def get_dimensions(df_initial):
    width = height = int(math.sqrt(len(df_initial['pixels'].iloc[0].split())))
    return width, height

def arr_to_img(arr, width, height):
    return np.asarray([int(pixel) for pixel in arr.split()]).reshape(width, height).astype(np.uint8)

def compute_img_mean_std(img_arr):
    # computing the mean and std of on the whole dataset
    
    imgs = [np.asarray(i.flatten()) for i in img_arr]
    imgs = np.asarray(imgs)
    means = np.mean(imgs)
    stdevs = np.std(imgs)

    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))
    return means,stdevs