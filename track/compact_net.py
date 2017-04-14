import tensorflow as tf
import numpy as np
from utils import batch_norm_conv, avg_pool_2x2, max_pool_2x2, weight_variable, conv2d_transpose_strided, bias_variable, conv2d_basic

FLAGS = tf.app.flags.FLAGS

def compact_base(image, phase):
    net = {}
    net['layer1'] = batch_norm_conv('conv1', image, [3, 3], num_output_layers=32, phase=phase)
    net['layer2'] = batch_norm_conv('conv2', net['layer1'], [3, 3], num_output_layers=32, phase=phase)
    net['layer2_p'] = avg_pool_2x2(net['layer2'])
    net['layer3'] = batch_norm_conv('conv3', net['layer2_p'], [3, 3], num_output_layers=32, phase=phase)
    net['layer4'] = batch_norm_conv('conv4', net['layer3'], [3, 3], num_output_layers=32, phase=phase)
    net['layer4_p'] = avg_pool_2x2(net['layer4'])
    net['layer5'] = batch_norm_conv('conv5', net['layer4_p'], [3, 3], num_output_layers=32, phase=phase)
    net['layer6'] = batch_norm_conv('conv6', net['layer5'], [3, 3], num_output_layers=32, phase=phase)
    net['layer6_p'] = avg_pool_2x2(net['layer6'])

    return net


def single_frame_inference(image, keep_prob, train=False):
    # set phase
    with tf.variable_scope("inference"):
        print("Single Frame Inference")
        net = compact_base(image, train)

        W7 = weight_variable([8, 8, 32, 512], name="W7")
        b7 = bias_variable([512], name="b7")
        # conv = tf.nn.conv2d(net['layer6_p'], W7, strides=[1, 1, 1, 1], padding="VALID")
        # conv7 = tf.nn.bias_add(conv, b7)
        conv7 = conv2d_basic(net['layer6_p'], W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")

        print(relu7.get_shape(), net['layer6_p'])
        W8 = weight_variable([1, 1, 512, 512], name="W8")
        b8 = bias_variable([512], name="b8")
        conv8 = conv2d_basic(relu7, W8, b8)
        relu8 = tf.nn.relu(conv8, name="relu8")

        W9 = weight_variable([1, 1, 512, FLAGS.NUM_OF_CLASSES], name="W9")
        b9 = bias_variable([FLAGS.NUM_OF_CLASSES], name="b9")
        conv9 = conv2d_basic(relu8, W9, b9)

        deconv_shape1 = net['layer4_p'].get_shape()
        W_t1 = weight_variable([4, 4, deconv_shape1[3].value, FLAGS.NUM_OF_CLASSES], name="W_t1")
        b_t1 = bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = conv2d_transpose_strided(conv9, W_t1, b_t1, output_shape=tf.shape(net['layer4_p']))
        fuse_1 = tf.add(conv_t1, net['layer4_p'], name="fuse_1")

        deconv_shape2 = net['layer2_p'].get_shape()
        W_t2 = weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(net['layer2_p']))
        fuse_2 = tf.add(conv_t2, net['layer2_p'], name="fuse_2")

        shape = tf.shape(image)
        deconv_shape3 = tf.pack([shape[0], shape[1], shape[2], FLAGS.NUM_OF_CLASSES])
        W_t3 = weight_variable([16, 16, FLAGS.NUM_OF_CLASSES, deconv_shape2[3].value], name="W_t3")
        b_t3 = bias_variable([FLAGS.NUM_OF_CLASSES], name="b_t3")
        conv_t3 = conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3)

        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

    return tf.expand_dims(annotation_pred, dim=3), conv_t3

def multi_frame_inference(image, frame_depth, train=False):
    frames = tf.split(1, frame_depth, image)
    frame_intermediates = []

    with tf.variable_scope("bottleneck"):
        image = tf.squeeze(frames[0], squeeze_dims=1)
        net = compact_base(image, train)
        frame_intermediates.append(net['layer6_p'])
    for i in range(1, frame_depth):
        with tf.variable_scope("bottleneck", reuse=True):
            # print("image", image.get_shape())
            image = tf.squeeze(frames[i], squeeze_dims=1)
            net = compact_base(image, train)
            frame_intermediates.append(net['layer6_p'])

    print("why, ", net['layer1'])

    conv_final_layer = tf.concat(3, frame_intermediates)

    with tf.variable_scope("inference"):

        W7 = weight_variable([8, 8, 32 * frame_depth, 32], name="W7")
        b7 = bias_variable([32], name="b7")
        conv7 = conv2d_basic(conv_final_layer, W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")

        W8 = weight_variable([1, 1, 32, 32], name="W8")
        b8 = bias_variable([32], name="b8")
        conv8 = conv2d_basic(relu7, W8, b8)
        relu8 = tf.nn.relu(conv8, name="relu8")

        W9 = weight_variable([1, 1, 32, FLAGS.NUM_OF_CLASSES], name="W9")
        b9 = bias_variable([FLAGS.NUM_OF_CLASSES], name="b9")
        conv9 = conv2d_basic(relu8, W9, b9)

        deconv_shape1 = net['layer4_p'].get_shape()
        W_t1 = weight_variable([4, 4, deconv_shape1[3].value, FLAGS.NUM_OF_CLASSES], name="W_t1")
        b_t1 = bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = conv2d_transpose_strided(conv9, W_t1, b_t1, output_shape=tf.shape(net['layer4_p']))
        fuse_1 = tf.add(conv_t1, net['layer4_p'], name="fuse_1")

        deconv_shape2 = net['layer2_p'].get_shape()
        W_t2 = weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(net['layer2_p']))
        fuse_2 = tf.add(conv_t2, net['layer2_p'], name="fuse_2")

        shape = tf.shape(image)
        deconv_shape3 = tf.pack([shape[0], shape[1], shape[2], FLAGS.NUM_OF_CLASSES])
        W_t3 = weight_variable([16, 16, FLAGS.NUM_OF_CLASSES, deconv_shape2[3].value], name="W_t3")
        b_t3 = bias_variable([FLAGS.NUM_OF_CLASSES], name="b_t3")
        conv_t3 = conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3)

        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

    return tf.expand_dims(annotation_pred, dim=3), conv_t3
