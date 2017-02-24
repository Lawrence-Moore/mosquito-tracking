from datetime import datetime
import os.path
import time

import numpy as np
import random
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import load_data
import vgg_single_frame, vgg_multi_frame
import multi_frame_network
from PIL import Image
import argparse

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('max_steps', 10000,
                            """Number of batches to run.""")

tf.app.flags.DEFINE_string('train_dir', 'tmp/train_logs',
                           """Directory where to write event logs """
                           """and checkpoint.""")


def inference(num_images, net_type):
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        new_shape = (FLAGS.TRAIN_IMAGE_SIZE, FLAGS.TRAIN_IMAGE_SIZE, 3)
        images_np, labels_np, all_pixel_locations = load_data.load_single_frame(num_images, new_shape)
        if net_type == 'single':
            # Get images and labels for the skeets
            image = tf.placeholder(tf.float32, shape=[None, new_shape[0], new_shape[1], new_shape[2]])
            label = tf.placeholder(tf.float32, shape=[None, new_shape[0], new_shape[1], 2])

            # Build a Graph that computes the logits predictions from the
            # inference model.
            keep_probability = tf.placeholder(tf.float32, name="keep_probability")
            pred_annotation, logits = vgg_single_frame.inference(image, keep_probability)

            tf.summary.image("input_image", image)
            tf.summary.image("ground_truth", tf.mul(tf.cast(tf.split(3, 2, label)[1], tf.uint8), 255))
            tf.summary.image("pred_annotation", tf.mul(tf.cast(pred_annotation, tf.uint8), 255))

            # Calculate loss.
            loss = vgg_single_frame.loss(logits, label, 500)

            # Build a Graph that trains the model with one batch of examples and
            # updates the model parameters.
            train_op = vgg_single_frame.train(loss, global_step)
        elif net_type == 'multi':
            # Get images and labels for the skeets
            frame_depth = 10
            image = tf.placeholder(tf.float32, shape=[None, frame_depth, new_shape[0], new_shape[1], new_shape[2]])
            label = tf.placeholder(tf.float32, shape=[None, new_shape[0], new_shape[1], 2])

            # Build a Graph that computes the logits predictions from the
            # inference model.
            keep_probability = tf.placeholder(tf.float32, name="keep_probability")
            pred_annotation, logits = vgg_multi_frame.inference(image, keep_probability, frame_depth)

            tf.summary.image("input_image", tf.squeeze(tf.split(1, frame_depth, image)[frame_depth - 1], squeeze_dims=1))
            tf.summary.image("ground_truth", tf.mul(tf.cast(tf.split(3, 2, label)[1], tf.uint8), 255))
            tf.summary.image("pred_annotation", tf.mul(tf.cast(pred_annotation, tf.uint8), 255))

            # Calculate loss.
            loss = vgg_multi_frame.loss(logits, label, 500)

            # Build a Graph that trains the model with one batch of examples and
            # updates the model parameters.
            train_op = vgg_multi_frame.train(loss, global_step)
        return train_op, loss


def train(num_images, net_type, model_name):
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        new_shape = (FLAGS.TRAIN_IMAGE_SIZE, FLAGS.TRAIN_IMAGE_SIZE, 3)
        images_np, labels_np, all_pixel_locations = load_data.load_single_frame(num_images, new_shape)
        if net_type == 'single':
            # Get images and labels for the skeets
            image = tf.placeholder(tf.float32, shape=[None, new_shape[0], new_shape[1], new_shape[2]])
            label = tf.placeholder(tf.float32, shape=[None, new_shape[0], new_shape[1], 2])

            # Build a Graph that computes the logits predictions from the
            # inference model.
            keep_probability = tf.placeholder(tf.float32, name="keep_probability")
            pred_annotation, logits = vgg_single_frame.inference(image, keep_probability)

            tf.summary.image("input_image", image)
            tf.summary.image("ground_truth", tf.mul(tf.cast(tf.split(3, 2, label)[1], tf.uint8), 255))
            tf.summary.image("pred_annotation", tf.mul(tf.cast(pred_annotation, tf.uint8), 255))

            # Calculate loss.
            loss = vgg_single_frame.loss(logits, label, 500)

            # Build a Graph that trains the model with one batch of examples and
            # updates the model parameters.
            train_op = vgg_single_frame.train(loss, global_step)
        elif net_type == 'multi':
            # Get images and labels for the skeets
            frame_depth = 10
            image = tf.placeholder(tf.float32, shape=[None, frame_depth, new_shape[0], new_shape[1], new_shape[2]])
            label = tf.placeholder(tf.float32, shape=[None, new_shape[0], new_shape[1], 2])

            # Build a Graph that computes the logits predictions from the
            # inference model.
            keep_probability = tf.placeholder(tf.float32, name="keep_probability")
            pred_annotation, logits = vgg_multi_frame.inference(image, keep_probability, frame_depth)

            tf.summary.image("input_image", tf.squeeze(tf.split(1, frame_depth, image)[frame_depth - 1], squeeze_dims=1))
            tf.summary.image("ground_truth", tf.mul(tf.cast(tf.split(3, 2, label)[1], tf.uint8), 255))
            tf.summary.image("pred_annotation", tf.mul(tf.cast(pred_annotation, tf.uint8), 255))

            # Calculate loss.
            loss = vgg_multi_frame.loss(logits, label, 500)

            # Build a Graph that trains the model with one batch of examples and
            # updates the model parameters.
            train_op = vgg_multi_frame.train(loss, global_step)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Build an initialization operation to run below.
        print("initializing all variables")
        init = tf.global_variables_initializer()

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        sess.run(init)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

        print("create the summar writter and begin training")

        for step in xrange(FLAGS.max_steps):
            start_time = time.time()

            if net_type == 'single':
                sample_images, sample_labels = get_single_frame_samples(num_images, images_np, labels_np, all_pixel_locations)
                _, loss_value = sess.run([train_op, loss], feed_dict={image: sample_images, label: sample_labels, keep_probability: 0.85})
            elif net_type == "multi":
                sample_images, sample_labels = get_multi_frame_samples(num_images, images_np, labels_np, all_pixel_locations, frame_depth)
                _, loss_value = sess.run([train_op, loss], feed_dict={image: sample_images, label: sample_labels, keep_probability: 0.85})
            duration = time.time() - start_time

            # print("got here")
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                summary_str = sess.run(summary_op, feed_dict={image: sample_images, label: sample_labels, keep_probability: 0.85})
                summary_writer.add_summary(summary_str, step)

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                     examples_per_sec, sec_per_batch))

                if step % 100 == 0:
                    summary_str = sess.run(summary_op, feed_dict={image: sample_images, label: sample_labels, keep_probability: 0.85})
                    summary_writer.add_summary(summary_str, step)

                # Save the model checkpoint periodically.
                if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                    checkpoint_path = os.path.join(FLAGS.train_dir, '%s_step%d.ckpt' % (model_name, step))
                    saver.save(sess, checkpoint_path, global_step=step)



def get_crop(location, crop_size, img_size):
    wiggle_room = 10

    def get_point_within_range(point, image_max_size):
        max_point = min(point, image_max_size - crop_size)
        min_point = max(point - crop_size, 0)
        # if point + crop_size - wiggle_room >= image_max_size:
        #     min_point = point - crop_size + 1
        # else:
        #     min_point = max(random.randint(point + wiggle_room - crop_size, point - wiggle_room), 0)
        return random.randint(min_point, max_point)
    return get_point_within_range(location[0], img_size[0]), get_point_within_range(location[1], img_size[1])


def get_single_frame_samples(num_images, images_np, labels_np, all_pixel_locations):
    sample_images = np.zeros((FLAGS.batch_size, FLAGS.TRAIN_IMAGE_SIZE, FLAGS.TRAIN_IMAGE_SIZE, 3))
    sample_labels = np.zeros((FLAGS.batch_size, FLAGS.TRAIN_IMAGE_SIZE, FLAGS.TRAIN_IMAGE_SIZE, 2))
    
    # choose the examples
    for i in range(FLAGS.batch_size):
        img_index = random.randint(0, num_images - 1)

        # choose the background
        source_image = images_np[img_index]
        source_label = labels_np[img_index]
        source_locations = all_pixel_locations[img_index]

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

    return sample_images, sample_labels


def get_multi_frame_samples(num_images, images_np, labels_np, all_pixel_locations, frame_depth):
    sample_images = np.zeros((FLAGS.batch_size, frame_depth, FLAGS.TRAIN_IMAGE_SIZE, FLAGS.TRAIN_IMAGE_SIZE, 3))
    sample_labels = np.zeros((FLAGS.batch_size, FLAGS.TRAIN_IMAGE_SIZE, FLAGS.TRAIN_IMAGE_SIZE, 2))
    
    # choose the examples
    for i in range(FLAGS.batch_size):
        img_index = random.randint(0, num_images - 1)

        # choose the background
        source_image = images_np[img_index]
        source_label = labels_np[img_index]
        source_locations = all_pixel_locations[img_index]

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

def main(argv=None):
    parser = argparse.ArgumentParser(description='Takes in the number of background images and architecture style')
    parser.add_argument('--num_images', help='number of backgroudn images')
    parser.add_argument('--arch', help='either single frame or multi frame')
    parser.add_argument('--model_name', help='name of the model')
    args = parser.parse_args()
    if args.arch and args.num_images and args.model_name:
        if args.arch != "single" and args.arch != "multi":
            print("Please pass in either single or multi as the architecture type")
        elif int(args.num_images) > 50:
            print("We only got fifty images fam")
        else:
            train(int(args.num_images), args.arch, args.model_name)
    else:
        print("boy, you missing either the number of background images, architecture type, or name of the model to be saved with. Fix ya yourself")

if __name__ == '__main__':
    tf.app.run()
