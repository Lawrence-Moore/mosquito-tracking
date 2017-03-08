from datetime import datetime
import os.path
import time

import numpy as np
import random
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import sys
import load_data
import vgg_single_frame, vgg_multi_frame
import multi_frame_network
from PIL import Image
import argparse
import utils

FLAGS = tf.app.flags.FLAGS


def build_net(net_type, image_shape, train=False):
    global_step = tf.Variable(0, trainable=False)
    if net_type == 'single':
        # Get images and labels for the skeets
        image = tf.placeholder(tf.float32, shape=[None, image_shape[0], image_shape[1], image_shape[2]])
        label = tf.placeholder(tf.float32, shape=[None, image_shape[0], image_shape[1], 2])

        # Build a Graph that computes the logits predictions from the
        # inference model.
        keep_probability = tf.placeholder(tf.float32, name="keep_probability")
        pred_annotation, logits = vgg_single_frame.inference(image, keep_probability, train=train)

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
        image = tf.placeholder(tf.float32, shape=[None, FLAGS.frame_depth, image_shape[0], image_shape[1], image_shape[2]])
        label = tf.placeholder(tf.float32, shape=[None, image_shape[0], image_shape[1], 2])

        # Build a Graph that computes the logits predictions from the
        # inference model.
        keep_probability = tf.placeholder(tf.float32, name="keep_probability")
        pred_annotation, logits = vgg_multi_frame.inference(image, keep_probability, FLAGS.frame_depth)

        tf.summary.image("input_image", tf.squeeze(tf.split(1, FLAGS.frame_depth, image)[FLAGS.frame_depth - 1], squeeze_dims=1))
        tf.summary.image("ground_truth", tf.mul(tf.cast(tf.split(3, 2, label)[1], tf.uint8), 255))
        tf.summary.image("pred_annotation", tf.mul(tf.cast(pred_annotation, tf.uint8), 255))

        # Calculate loss.
        loss = vgg_multi_frame.loss(logits, label, 500)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = vgg_multi_frame.train(loss, global_step)
    return image, label, keep_probability, logits, train_op, loss, pred_annotation


def train(num_image_range, net_type, model_name):
    # Create a saver.
    with tf.Graph().as_default():
        image_shape = (FLAGS.TRAIN_IMAGE_SIZE, FLAGS.TRAIN_IMAGE_SIZE, 3)
        image, label, keep_probability, logits, train_op, loss, pred_annotation = build_net(net_type, image_shape, train=True)
        saver = tf.train.Saver(tf.global_variables())

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Build an initialization operation to run below.
        print("initializing all variables")
        init = tf.global_variables_initializer()

        # Start running operations on the Graph.
        sess = tf.Session()
        sess.run(init)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)

        print("create the summar writter and begin training")

        for step in xrange(FLAGS.max_steps):
            start_time = time.time()

            if net_type == 'single':
                sample_images, sample_labels, _ = utils.get_single_frame_samples(num_image_range)
                _, loss_value = sess.run([train_op, loss], feed_dict={image: sample_images, label: sample_labels, keep_probability: 0.85})
            elif net_type == "multi":
                sample_images, sample_labels = utils.get_multi_frame_samples(num_image_range, FLAGS.frame_depth)
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
                if step == 10 or step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                    checkpoint_path = os.path.join(FLAGS.logs_dir, '%s.ckpt' % model_name)
                    saver.save(sess, checkpoint_path)


def evaluate(net_type, model_name, num_images_range):
    with tf.Graph().as_default():

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for background_image, background_locations in zip(test_images, real_locations):
                # tf.reset_default_graph()
            # sample_images, sample_labels, sample_skeeters = utils.get_single_frame_samples(test_images, test_labels,
            #                                                                                real_locations)
            # image, label, keep_probability, logits, train_op, loss, pred_annotation = build_net(net_type,
            #                                                                                     sample_images[0].shape)
                random_index = random.randint(0, background_image.shape[0] - 1)
                sample_image = background_image[random_index, :, :, :]
                sample_locations = background_locations[random_index, :, :]

                image, label, keep_probability, logits, train_op, loss, pred_annotation = build_net(net_type, sample_image.shape)
                saver = tf.train.Saver(tf.global_variables())

                checkpoint_path = os.path.join(FLAGS.logs_dir, '%s.ckpt' % model_name)
                saver.restore(sess, checkpoint_path)

                # try something from each image
                sample_images = sample_image[np.newaxis]
                # # sample_label = sample_label[np.newaxis]
                # print("herh", sample_image.shape)

                predictions, logits = sess.run([pred_annotation, logits], feed_dict={image: sample_images, keep_probability: 0.85})
                print(predictions[0].shape, logits[0].shape)
                boxes = generate_object_detections(predictions, perc_thresh=0.6)
                print(boxes)
                closest_distances = closest_mosquito(boxes[0], sample_locations)
                print(closest_distances)
            # boxes = utils.non_max_suppression_fast(boxes, 0.5)
            # print(boxes, real_locations[0].shape, len(real_locations))
            # for i in range(len(boxes)):
            #     print(boxes[i], sample_skeeters[i, :, :])


def generate_object_detections(predictions, perc_thresh=0.6):
    all_boxes = []

    for index in range(predictions.shape[0]):
        prediction = predictions[index, :, :, :]
        boxes = np.zeros([1, 4])
        did_any_meet_thresh = True
        width, height = prediction.shape[0], prediction.shape[1]
        window_size = 1
        while window_size < min(prediction.shape[0], prediction.shape[1]) and did_any_meet_thresh:
            did_any_meet_thresh = False
            for i in range(0, width - window_size):
                for j in range(0, height - window_size):
                    box = prediction[i: i + window_size, j: j + window_size]
                    # print(box.shape, window_size ** 2, np.sum(box))
                    if (np.sum(box) / (window_size ** 2)) >= perc_thresh:
                        did_any_meet_thresh = True
                        boxes = np.vstack((boxes, [i, j, i + window_size, j + window_size]))
            window_size += 1

        boxes_after_suppression = utils.non_max_suppression_fast(boxes[1:, :], 0.25)
        all_boxes.append(boxes_after_suppression)

    return all_boxes


def closest_mosquito(boxes, real_location):
    closest_dist = []
    for i in range(boxes.shape[0]):
        box = boxes[i, :]
        center = int(box[0] + (box[2] - box[0]) / 2), int(box[1] + (box[3] - box[1]) /2)
        min_dist = sys.maxsize
        for j in range(real_location.shape[0]):
            skeeter = real_location[j, :]
            dist = ((skeeter[0] - center[0])**2 + (skeeter[1] - center[1])**2) **(1/2)
            if dist < min_dist:
                min_dist = dist
        closest_dist.append(min_dist)
    return closest_dist

def main(argv=None):
    parser = argparse.ArgumentParser(description='Takes in the number of background images and architecture style')
    parser.add_argument('--train_num_images', help='number of backgroudn images')
    parser.add_argument('--test_num_images', help='number of backgroudn images')
    parser.add_argument('--arch', help='either single frame or multi frame')
    parser.add_argument('--model_name', help='name of the model')
    parser.add_argument('--train', help='boolean of whether we are training')
    args = parser.parse_args()
    if args.arch and args.model_name:
        if args.arch != "single" and args.arch != "multi":
            print("Please pass in either single or multi as the architecture type")
        elif args.train_num_images and int(args.train_num_images) > 70:
            print("We only got so many training images fam")
        else:
            # load images
            train_num_images, test_num_images = int(args.train_num_images), int(args.test_num_images)
            train_num_images_range = (0, train_num_images - 1)
            test_num_images_range = (70, test_num_images + 70 - 1)
            # train_images, train_labels, train_pixel_locations = load_data.load_single_frame(0, train_num_images)
            # test_images, test_labels, test_pixel_locations = load_data.load_single_frame(70, test_num_images)
            
            print("Images loaded")
            # build the net
            if args.train == "True":
                train(train_num_images_range, args.arch, args.model_name)
            # print(test_pixel_locations[0].shape, "wuttttt")
            evaluate(args.arch, args.model_name, test_num_images_range)
    else:
        print("boy, you missing either the number of background images, architecture type, or name of the model to be saved with. Fix ya yourself")

if __name__ == '__main__':
    tf.app.run()
