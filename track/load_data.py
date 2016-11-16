import cv2
import scipy.io as sio
import numpy as np

videoName = "../generate/myVideo.avi"
labels = "../generate/locations.mat"

def load():
    return convert_video_to_numpy_array(videoName), read_mat(labels)


def convert_video_to_numpy_array(name):
    vidcap = cv2.VideoCapture(name)
    success, image = vidcap.read()
    success = True

    images = image[None, :]
    while success:
        success, image = vidcap.read()
        if success: images = np.concatenate((images, image[None, :]))
    return images

def read_mat(file_name):
    return np.rollaxis(sio.loadmat(file_name)['locations'], 3, 0)
