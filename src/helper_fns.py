import os
from matplotlib.image import imread
import numpy as np
import cv2
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
def prepare_train_test_mnist(data_path):
    total_data = []
    total_label = []
    for label in range(10):
        f = os.listdir(os.path.join(data_path,str(label)))
        for file_name in f:
            a = imread(os.path.join(data_path,str(label),file_name))
            a = a.reshape(784,)
            total_data.append(a)
            total_label.append(label)
    return np.asarray(total_data),np.asarray(total_label)

def prepare_train_test_cat_dog(data_path):
    total_data = []
    total_label = []
    for label in (["cat","dog"]):
        f = os.listdir(os.path.join(data_path,str(label)))
        for file_name in f:
            a = imread(os.path.join(data_path,str(label),file_name))
            a = rgb2gray(a)
            a = cv2.resize(a, (100, 100), interpolation=cv2.INTER_AREA)
            a = a.reshape(a.shape[0] * a.shape[1], )

            total_data.append(a)
            if label == "cat":
                total_label.append(0)
            else:
                total_label.append(1)
    return np.asarray(total_data),np.asarray(total_label)