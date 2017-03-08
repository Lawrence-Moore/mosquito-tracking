import imageio
from PIL import Image
import cv2
import numpy as np
import h5py
import time

videoName = "../data/myVideo.avi"
locations = "../data/locations.mat"

def load_single_frame(start, num_images):
    all_images = []
    all_labels = []
    all_pixel_locations = []
    for i in range(start, start + num_images):
        i += 1
        video_name = "../data/video%d.avi" % i
        mat_name = "../data/locations%d.mat" % i
        pixel_location_name = "../data/pixel_locations%d.mat" % i
        all_labels.append(read_heatmap(mat_name))
        all_images.append(convert_video_to_numpy_array(video_name))
        all_pixel_locations.append(read_locations(pixel_location_name))

        current_time = time.localtime()
        print(i, time.strftime('%a, %d %b %Y %H:%M:%S GMT', current_time))
    return all_images, all_labels, all_pixel_locations

def load_multi_frame(frame_size):
    labels = read_mat(locations)
    images = convert_video_to_numpy_array(videoName)

    multiframe_labels = np.zeros([labels.shape[0] - frame_size, frame_size, labels.shape[1], labels.shape[2], labels.shape[3]])
    multiframe_images = np.zeros([images.shape[0] - frame_size, frame_size, images.shape[1], images.shape[2], images.shape[3]])
    for index in range(0, labels.shape[0] - frame_size):
        multiframe_labels[index, :, :, :] = labels[index:index + frame_size, :, :, :]
        multiframe_images[index, :, :, :] = images[index:index + frame_size, :, :, :]
    return multiframe_images, multiframe_labels


def convert_video_to_numpy_array(name):
    vid = cv2.VideoCapture(name)
    ret, image = vid.read()
    # image = np.array(Image.fromarray(image).resize(new_shape[0:2]))

    images = image[None, :]
    while ret:
        # image = np.array(Image.fromarray(image).resize(new_shape[0:2]))
        ret, frame = vid.read()
        if ret:
            images = np.concatenate((images, frame[None, :]))
    return images

def read_heatmap(file_name):
    # maps = np.array(h5py.File(file_name)['locations'])
    # maps = np.rollaxis(np.rollaxis(maps, 1, 0), 0, 4)
    maps = np.array(h5py.File(file_name)['locations'])
    maps = np.swapaxes(maps,1, 3)
    # resized_maps = np.zeros((maps.shape[0], new_shape[0], new_shape[1], 1))
    # for i in range(maps.shape[0]):
    #     map = maps[i, :, :, 0]
    #     resized_maps[i, :, :, 0] = np.array(Image.fromarray(map).resize(new_shape[0:2]))
    return maps

def read_locations(file_name):
    locations = np.array(h5py.File(file_name)['all_positions'])
    return np.rollaxis(locations, 2, 1)

def resize_images(images, new_shape):
    resized_images = []
    for image in images:
        resized_images.append(np.array(Image.fromarray(image).resize(new_shape[0:2])))
    return resized_images
