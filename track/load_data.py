import cv2
import scipy.io as sio
import numpy as np

videoName = "../generate/myVideo.avi"
locations = "../generate/locations.mat"

def load_single_frame(new_shape):
    labels = read_mat(locations, new_shape)
    images = convert_video_to_numpy_array(videoName, new_shape)
    return labels, images

def load_multi_frame(frame_size, new_shape):
    labels = read_mat(locations, new_shape)
    images = convert_video_to_numpy_array(videoName, new_shape)

    multiframe_labels = np.zeros([labels.shape[0] - frame_size, frame_size, labels.shape[1], labels.shape[2], labels.shape[3]])
    multiframe_images = np.zeros([images.shape[0] - frame_size, frame_size, images.shape[1], images.shape[2], images.shape[3]])
    for index in range(0, labels.shape[0] - frame_size):
        multiframe_labels[index, :, :, :] = labels[index:index + frame_size, :, :, :]
        multiframe_images[index, :, :, :] = images[index:index + frame_size, :, :, :]
    return multiframe_labels, multiframe_images


def convert_video_to_numpy_array(name, new_shape):
    vidcap = cv2.VideoCapture(name)
    success, image = vidcap.read()
    image = np.resize(image, new_shape)
    success = True

    images = image[None, :]
    while success:
        success, image = vidcap.read()
        image = np.resize(image, new_shape)
        if success: images = np.concatenate((images, image[None, :]))
    return images

def read_mat(file_name, new_shape):
    maps = np.rollaxis(sio.loadmat(file_name)['locations'], 3, 0)
    return np.resize(maps, [maps.shape[0], new_shape[0], new_shape[1], new_shape[2]])
