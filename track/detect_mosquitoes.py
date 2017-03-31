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
import matplotlib.pyplot as plt

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
        tf.summary.scalar("yo", tf.constant(5))
        summary_op = tf.summary.merge_all()

        # Build an initialization operation to run below.
        print("initializing all variables")
        init = tf.global_variables_initializer()

        # Start running operations on the Graph.
        sess = tf.Session()
        sess.run(init)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        directory = os.path.join(FLAGS.logs_dir, '%s/' % model_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        checkpoint_path = os.path.join(FLAGS.logs_dir, '%s/%s.ckpt' % (model_name, model_name))
        summary_writer = tf.summary.FileWriter(directory, sess.graph)

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

                # Save the model checkpoint periodically.
                if step == 10 or step % 100 == 0 or (step + 1) == FLAGS.max_steps:
                    print("Saving session")
                    saver.save(sess, checkpoint_path)


def evaluate(net_type, model_name, num_images_range):
    # FLAGS.batch_size = 10
    with tf.Graph().as_default():

        with tf.Session() as sess:
            sample_images, sample_locations = utils.get_single_frame_test_images(1, num_images_range)
            image, label, keep_probability, logits, train_op, loss, pred_annotation = build_net(net_type,
                                                                                                sample_images[0].shape)
            # random_index = random.randint(0, background_image.shape[0] - 1)
            # sample_image = background_image[random_index, :, :, :]
            # sample_locations = background_locations[random_index, :, :]
            #
            # image, label, keep_probability, logits, train_op, loss, pred_annotation = build_net(net_type, sample_image.shape)
            saver = tf.train.Saver(tf.global_variables())

            checkpoint_path = os.path.join(FLAGS.logs_dir, '%s.ckpt' % model_name)
            saver.restore(sess, checkpoint_path)

            # try something from each image
            # sample_images = sample_image[np.newaxis]
            # # sample_label = sample_label[np.newaxis]
            # print("herh", sample_image.shape)

            predictions, logits = sess.run([pred_annotation, logits], feed_dict={image: sample_images, keep_probability: 0.85})
            np.save("predictions", predictions)
            np.save("logits", logits)
            np.save("sample_locations", sample_locations)

            all_boxes = generate_object_detections(predictions, perc_thresh=0.6)
            print(len(all_boxes), all_boxes[0].shape)
            sorted_boxes, sorted_confidences = sort_detections_by_confidence(all_boxes, logits)
            list_all_true_positives = find_true_positives(sorted_boxes, sample_locations, dist_cutoff=10)

            all_true_positives = sort_and_combine_true_positives(sorted_confidences, list_all_true_positives)
            all_false_positives = np.ones(all_true_positives.shape())
            all_false_positives[all_true_positives > 0] = 0

            plot_precision_recall(all_false_positives, all_true_positives)
            # boxes = utils.non_max_suppression_fast(boxes, 0.5)
            # print(boxes, real_locations[0].shape, len(real_locations))
            # for i in range(len(boxes)):
            #     print(boxes[i], sample_skeeters[i, :, :])

def plot_precision_recall(all_false_positives, all_true_positives):
    plt.xlabel('Recall')
    plt.ylablel('Precision')
    plt.title('Precision Recall Curve')
    plt.axis([0, 1, 0, 1])
    p_axis, r_axis = np.cumsum(all_true_positives), np.cumsum(all_false_positives)
    plt.plot(p_axis, r_axis, 'k-')
    plt.show

def sort_and_combine_true_positives(sorted_confidences, list_all_true_positives):
    all_confidences, all_results = np.concatenate(sorted_confidences), np.concatenate(list_all_true_positives)

    sorted_order = all_confidences.argsort()[::-1]
    return all_results[sorted_order]

def generate_object_detections(predictions, perc_thresh=0.6):
    all_boxes = []

    for index in range(predictions.shape[0]):
        # go through the different pixel by pixel mosquito predictions generated by the net
        prediction = predictions[index, :, :, :]
        boxes = np.zeros([1, 4])

        did_any_meet_thresh = True
        width, height = prediction.shape[0], prediction.shape[1]
        window_size = 1
        # increase window size; stop
        stop = False
        while window_size < min(width, height) and not stop:
            did_any_meet_thresh = False
            for i in range(0, width - window_size):
                for j in range(0, height - window_size):
                    box = prediction[i: i + window_size, j: j + window_size]
                    # print(box.shape, window_size ** 2, np.sum(box))
                    if (np.sum(box) / (window_size ** 2)) >= perc_thresh:
                        did_any_meet_thresh = True
                        boxes = np.vstack((boxes, [i, j, i + window_size, j + window_size]))
                    elif did_any_meet_thresh:
                        stop = True
            window_size += 1

        boxes_after_suppression = utils.non_max_suppression_fast(boxes[1:, :], 0.25)
        all_boxes.append(boxes_after_suppression)

    return all_boxes


def find_true_positives(all_boxes, real_locations, dist_cutoff):
    all_true_positives = []
    # iterate through each sample
    for k, image_boxes in enumerate(all_boxes):
        real_location = real_locations[k, :, :]
        box_true_positives = np.zeros(image_boxes.shape[0])
        # iterate through each box in a sample
        for i in range(image_boxes.shape[0]):
            box = image_boxes[i, :]
            center = int(box[0] + (box[2] - box[0]) / 2), int(box[1] + (box[3] - box[1]) /2)
            min_dist = sys.maxsize
            for j in range(real_location.shape[0]):
                skeeter = real_location[j, :]
                dist = ((skeeter[0] - center[0])**2 + (skeeter[1] - center[1])**2) **(1/2)
                if dist < min_dist:
                    min_dist = dist
            print(min_dist)
            box_true_positives[i] = min_dist < dist_cutoff
        all_true_positives.append(box_true_positives)
    return all_true_positives

def sort_detections_by_confidence(all_boxes, logits):
    sorted_confidences = []
    sorted_boxes = []
    for img_index, boxes in enumerate(all_boxes):
        print(boxes.shape)
        confidences = np.ones(boxes.shape[0])
        for i in range(boxes.shape[0]):
            box = boxes[i, :]
            confidences[i] = np.mean(logits[img_index, box[0]: box[2], box[1]: box[3], 1])
        sorted_order = confidences.argsort()[::-1]
        sorted_boxes.append(boxes[sorted_order, :])
        sorted_confidences.append(confidences[sorted_order])

    return sorted_boxes, sorted_confidences


def inference(num_images, net_type):
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        new_shape = (FLAGS.TRAIN_IMAGE_SIZE, FLAGS.TRAIN_IMAGE_SIZE, 3)
        # images_np, labels_np, all_pixel_locations = load_data.load_single_frame(num_images, new_shape)
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
        return train_op, loss


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
        # print(source_image.shape, random_index)
        sample_image = source_image[random_index, :, :, :]
        sample_label = source_label[random_index, :, :, 0]
        sample_locations = source_locations[random_index, :, :]

        # crop to include the 'skeeter
        mosquito = sample_locations[random.randint(0, sample_locations.shape[0] - 1), :]
        (x_min, y_min) = get_crop(mosquito, FLAGS.TRAIN_IMAGE_SIZE, sample_image.shape[0:2])

        # randomly choose an area
        sample_images[i, :, :, :] = sample_image[x_min: x_min + FLAGS.TRAIN_IMAGE_SIZE,
                                    y_min: y_min + FLAGS.TRAIN_IMAGE_SIZE, :]
        label_crop = sample_label[x_min: x_min + FLAGS.TRAIN_IMAGE_SIZE, y_min: y_min + FLAGS.TRAIN_IMAGE_SIZE]
        sample_labels[i, :, :, 0][label_crop == 0] = 1
        sample_labels[i, :, :, 1][label_crop == 1] = 1

    return sample_images, sample_labels


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

            test_index_start = 8
            test_num_images_range = (test_index_start, test_index_start + test_num_images)
            # train_images, train_labels, train_pixel_locations = load_data.load_single_frame(0, train_num_images)
            # test_images, test_labels, test_pixel_locations = load_data.load_single_frame(70, test_num_images)
            
            print("Images loaded")
            # build the net
            if args.train == "True":
                train(train_num_images_range, args.arch, args.model_name)
                # old_train(1, args.arch, args.model_name)
            # print(test_pixel_locations[0].shape, "wuttttt")
            else:
                evaluate(args.arch, args.model_name, test_num_images_range)
    else:
        print("boy, you missing either the number of background images, architecture type, or name of the model to be saved with. Fix ya yourself")

if __name__ == '__main__':
    tf.app.run()

