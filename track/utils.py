import tensorflow as tf
import numpy as np
import scipy.misc as misc
import os, sys
from six.moves import urllib
import tarfile
import zipfile
import scipy.io
import random
from load_data import load_single_frame

FLAGS = tf.app.flags.FLAGS

def get_model_data(dir_path, model_url):
    maybe_download_and_extract(dir_path, model_url)
    filename = model_url.split("/")[-1]
    filepath = os.path.join(dir_path, filename)
    if not os.path.exists(filepath):
        raise IOError("VGG Model not found!")
    data = scipy.io.loadmat(filepath)
    return data


def maybe_download_and_extract(dir_path, url_name, is_tarfile=False, is_zipfile=False):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    filename = url_name.split('/')[-1]
    filepath = os.path.join(dir_path, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write(
                '\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(url_name, filepath, reporthook=_progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
        if is_tarfile:
            tarfile.open(filepath, 'r:gz').extractall(dir_path)
        elif is_zipfile:
            with zipfile.ZipFile(filepath) as zf:
                zip_dir = zf.namelist()[0]
                zf.extractall(dir_path)

# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


def save_image(image, save_dir, name, mean=None):
    """
    Save image by unprocessing if mean given else just save
    :param mean:
    :param image:
    :param save_dir:
    :param name:
    :return:
    """
    if mean:
        image = unprocess_image(image, mean)
    misc.imsave(os.path.join(save_dir, name + ".png"), image)

def get_crop(location, crop_size, img_size):
    wiggle_room = 10

    def get_point_within_range(point, image_max_size):
        max_point = min(point, image_max_size - crop_size)
        min_point = max(point - crop_size, 0)
        # if point + crop_size - wiggle_room >= image_max_size:
        #     min_point = point - crop_size + 1
        # else:
        #     min_point = max(random.randint(point + wiggle_room - crop_size, point - wiggle_room), 0)
        if max_point < min_point:
            print(point, image_max_size)
            max_point = min_point
        return random.randint(min_point, max_point)
    # if (location[0] >= img_size[0] or location[1] >= img_size[1]):
    #     print(location, img_size)
    return get_point_within_range(location[0], img_size[0]), get_point_within_range(location[1], img_size[1])


def get_single_frame_samples(num_images_range):
    sample_images = np.zeros((FLAGS.batch_size, FLAGS.TRAIN_IMAGE_SIZE, FLAGS.TRAIN_IMAGE_SIZE, 3))
    sample_labels = np.zeros((FLAGS.batch_size, FLAGS.TRAIN_IMAGE_SIZE, FLAGS.TRAIN_IMAGE_SIZE, 2))
    sample_skeeters = np.zeros((FLAGS.batch_size, 1, 2))
    # choose the examples
    img_index = random.randint(num_images_range[0], num_images_range[1])

    source_image, source_label, source_locations = load_single_frame(img_index , 1)

    # choose the background
    # source_image = all_images[0]
    # source_label = all_labels[0]
    # source_locations = all_pixel_locations[0]

    for i in range(FLAGS.batch_size):

        # choose the specific image
        random_index = random.randint(0, source_image.shape[0] - 1)
        sample_image = source_image[random_index, :, :, :]
        sample_label = source_label[random_index, :, :, 0]
        sample_locations = source_locations[random_index, :, :]

        # crop to include the 'skeeter
        mosquito = sample_locations[random.randint(0, sample_locations.shape[0] - 1), :]
        (x_min, y_min) = get_crop(mosquito, FLAGS.TRAIN_IMAGE_SIZE, sample_image.shape[0:2])

        # randomly choose an area
        sample_images[i, :, :, :] = sample_image[x_min: x_min + FLAGS.TRAIN_IMAGE_SIZE, y_min: y_min + FLAGS.TRAIN_IMAGE_SIZE, :]
        label_crop = sample_label[x_min: x_min + FLAGS.TRAIN_IMAGE_SIZE, y_min: y_min + FLAGS.TRAIN_IMAGE_SIZE]
        sample_labels[i, :, :, 0][label_crop == 0] = 1
        sample_labels[i, :, :, 1][label_crop == 1] = 1
        sample_skeeters[i, :, :] = mosquito - np.array([x_min, y_min])

    return sample_images, sample_labels, sample_skeeters

def get_single_frame_test_images(num_per_image, num_images_range, img_size=512):
    num_images = num_images_range[1] - num_images_range[0]
    sample_images = np.zeros((num_per_image * num_images, img_size, img_size, 3))
    sample_pixel_locations = []
    # choose the examples
    for i, j in zip(range(num_images_range[0], num_images_range[1]), range(num_images)):

        # choose the background
        images, labels, pixel_locations = load_single_frame(i , 1)
        random_indices = np.random.choice(images.shape[0], num_per_image)
        sample_images[j * num_per_image: num_per_image * (j + 1), :, :, :] = images[random_indices, :, :, :]
        for index in random_indices:
            sample_pixel_locations.append(pixel_locations[index, :, :])

    return sample_images, sample_pixel_locations

def get_multi_frame_test_images(num_per_image, num_images_range, frame_depth, img_size=512):
    num_images = num_images_range[1] - num_images_range[0]
    sample_images = np.zeros((num_per_image * num_images, frame_depth, img_size, img_size, 3))
    sample_pixel_locations = []
    # choose the examples
    for i, j in zip(range(num_images_range[0], num_images_range[1]), range(num_images)):

        # choose the background
        images, labels, pixel_locations = load_single_frame(i , 1)
        random_indices = np.random.choice(images.shape[0] - frame_depth, num_per_image)
        sample_images[j * num_per_image: num_per_image * (j + 1), :, :, :] = images[random_indices, :, :, :]
        for index in random_indices:
            sample_pixel_locations.append(pixel_locations[index, :, :])

    return sample_images, sample_pixel_locations


def get_multi_frame_samples(num_images_range, frame_depth):
    sample_images = np.zeros((FLAGS.batch_size, frame_depth, FLAGS.TRAIN_IMAGE_SIZE, FLAGS.TRAIN_IMAGE_SIZE, 3))
    sample_labels = np.zeros((FLAGS.batch_size, FLAGS.TRAIN_IMAGE_SIZE, FLAGS.TRAIN_IMAGE_SIZE, 2))

    img_index = random.randint(num_images_range[0], num_images_range[1])

    all_images, all_labels, all_pixel_locations = load_single_frame(img_index , 1)

    # choose the background
    source_image = all_images[0]
    source_label = all_labels[0]
    source_locations = all_pixel_locations[0]

    # choose the examples
    for i in range(FLAGS.batch_size):

        # choose the specific image that's at least frame_depth back from the end
        random_index = random.randint(0, source_image.shape[0] - 1 - frame_depth)
        for frame_i in range(frame_depth):
            sample_image = source_image[random_index + frame_i, :, :, :]
            sample_locations = source_locations[random_index + frame_i, :, :]

            # crop to include the 'skeeter
            mosquito = sample_locations[random.randint(0, sample_locations.shape[0] - 1), :]
            (x_min, y_min) = get_crop(mosquito, FLAGS.TRAIN_IMAGE_SIZE, sample_image.shape[0:2])
            sample_images[i, frame_i, :, :, :] = sample_image[x_min: x_min + FLAGS.TRAIN_IMAGE_SIZE, y_min: y_min + FLAGS.TRAIN_IMAGE_SIZE, :]

        # get the last frame
        sample_label = source_label[random_index + frame_depth, :, :, 0]
        label_crop = sample_label[x_min: x_min + FLAGS.TRAIN_IMAGE_SIZE, y_min: y_min + FLAGS.TRAIN_IMAGE_SIZE]
        sample_labels[i, :, :, 0][label_crop == 0] = 1
        sample_labels[i, :, :, 1][label_crop == 1] = 1

    return sample_images, sample_labels


def get_variable(weights, name):
    init = tf.constant_initializer(weights, dtype=tf.float32)
    var = tf.get_variable(name=name, initializer=init,  shape=weights.shape)
    return var


def weight_variable(shape, stddev=0.02, name=None):
    # print(shape)
    initial = tf.truncated_normal(shape, stddev=stddev)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)


def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)


def get_tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)


def conv2d_basic(x, W, bias):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
    return tf.nn.bias_add(conv, bias)


def conv2d_strided(x, W, b):
    conv = tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)


def conv2d_transpose_strided(x, W, b, output_shape=None, stride = 2):
    # print x.get_shape()
    # print W.get_shape()
    if output_shape is None:
        output_shape = x.get_shape().as_list()
        output_shape[1] *= 2
        output_shape[2] *= 2
        output_shape[3] = W.get_shape().as_list()[2]
    # print output_shape
    conv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)


def leaky_relu(x, alpha=0.0, name=""):
    return tf.maximum(alpha * x, x, name)


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def local_response_norm(x):
    return tf.nn.lrn(x, depth_radius=5, bias=2, alpha=1e-4, beta=0.75)


def batch_norm(x, n_out, phase_train, scope='bn', decay=0.9, eps=1e-5):
    """
    Code taken from http://stackoverflow.com/a/34634291/2267819
    """
    with tf.variable_scope(scope):
        beta = tf.get_variable(name='beta', shape=[n_out], initializer=tf.constant_initializer(0.0)
                               , trainable=True)
        gamma = tf.get_variable(name='gamma', shape=[n_out], initializer=tf.random_normal_initializer(1.0, 0.02),
                                trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=decay)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)
    return normed


def process_image(image, mean_pixel):
    return image - mean_pixel


def unprocess_image(image, mean_pixel):
    return image + mean_pixel


def bottleneck_unit(x, out_chan1, out_chan2, down_stride=False, up_stride=False, name=None):
    """
    Modified implementation from github ry?!
    """

    def conv_transpose(tensor, out_channel, shape, strides, name=None):
        out_shape = tensor.get_shape().as_list()
        in_channel = out_shape[-1]
        kernel = weight_variable([shape, shape, out_channel, in_channel], name=name)
        shape[-1] = out_channel
        return tf.nn.conv2d_transpose(x, kernel, output_shape=out_shape, strides=[1, strides, strides, 1],
                                      padding='SAME', name='conv_transpose')

    def conv(tensor, out_chans, shape, strides, name=None):
        in_channel = tensor.get_shape().as_list()[-1]
        kernel = weight_variable([shape, shape, in_channel, out_chans], name=name)
        return tf.nn.conv2d(x, kernel, strides=[1, strides, strides, 1], padding='SAME', name='conv')

    def bn(tensor, name=None):
        """
        :param tensor: 4D tensor input
        :param name: name of the operation
        :return: local response normalized tensor - not using batch normalization :(
        """
        return tf.nn.lrn(tensor, depth_radius=5, bias=2, alpha=1e-4, beta=0.75, name=name)

    in_chans = x.get_shape().as_list()[3]

    if down_stride or up_stride:
        first_stride = 2
    else:
        first_stride = 1

    with tf.variable_scope('res%s' % name):
        if in_chans == out_chan2:
            b1 = x
        else:
            with tf.variable_scope('branch1'):
                if up_stride:
                    b1 = conv_transpose(x, out_chans=out_chan2, shape=1, strides=first_stride,
                                        name='res%s_branch1' % name)
                else:
                    b1 = conv(x, out_chans=out_chan2, shape=1, strides=first_stride, name='res%s_branch1' % name)
                b1 = bn(b1, 'bn%s_branch1' % name, 'scale%s_branch1' % name)

        with tf.variable_scope('branch2a'):
            if up_stride:
                b2 = conv_transpose(x, out_chans=out_chan1, shape=1, strides=first_stride, name='res%s_branch2a' % name)
            else:
                b2 = conv(x, out_chans=out_chan1, shape=1, strides=first_stride, name='res%s_branch2a' % name)
            b2 = bn(b2, 'bn%s_branch2a' % name, 'scale%s_branch2a' % name)
            b2 = tf.nn.relu(b2, name='relu')

        with tf.variable_scope('branch2b'):
            b2 = conv(b2, out_chans=out_chan1, shape=3, strides=1, name='res%s_branch2b' % name)
            b2 = bn(b2, 'bn%s_branch2b' % name, 'scale%s_branch2b' % name)
            b2 = tf.nn.relu(b2, name='relu')

        with tf.variable_scope('branch2c'):
            b2 = conv(b2, out_chans=out_chan2, shape=1, strides=1, name='res%s_branch2c' % name)
            b2 = bn(b2, 'bn%s_branch2c' % name, 'scale%s_branch2c' % name)

        x = b1 + b2
        return tf.nn.relu(x, name='relu')


def add_to_regularization_and_summary(var):
    if var is not None:
        tf.summary.histogram(var.op.name, var)
        tf.add_to_collection("reg_loss", tf.nn.l2_loss(var))


def add_activation_summary(var):
    if var is not None:
        tf.summary.histogram(var.op.name + "/activation", var)
        tf.summary.scalar(var.op.name + "/sparsity", tf.nn.zero_fraction(var))


def add_gradient_summary(grad, var):
    if grad is not None:
        tf.summary.histogram(var.op.name + "/gradient", grad)