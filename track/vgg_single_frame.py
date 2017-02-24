from __future__ import print_function
import tensorflow as tf
import numpy as np

import utils
import datetime
from six.moves import xrange
from vgg_base import vgg_net

FLAGS = tf.app.flags.FLAGS

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
MOVING_AVERAGE_DECAY = 0.9999
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = FLAGS.learning_rate

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

def inference(image, keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)

    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))

    weights = np.squeeze(model_data['layers'])

    # accounts for the mean being subtracted from the image
    processed_image = utils.process_image(image, mean_pixel)

    with tf.variable_scope("inference"):
        image_net = vgg_net(weights, processed_image)
        conv_final_layer = image_net["conv5_3"]

        pool5 = utils.max_pool_2x2(conv_final_layer)

        print("pool5 shape", pool5.get_shape())
        W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
        b6 = utils.bias_variable([4096], name="b6")
        conv6 = utils.conv2d_basic(pool5, W6, b6)
        relu6 = tf.nn.relu(conv6, name="relu6")

        print("reul6 shape", relu6.get_shape())

        if FLAGS.debug:
            utils.add_activation_summary(relu6)
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")

        if FLAGS.debug:
            utils.add_activation_summary(relu7)
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

        W8 = utils.weight_variable([1, 1, 4096, FLAGS.NUM_OF_CLASSES], name="W8")
        b8 = utils.bias_variable([FLAGS.NUM_OF_CLASSES], name="b8")
        conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
        # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")

        # now to upscale to actual image size
        deconv_shape1 = image_net["pool4"].get_shape()
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, FLAGS.NUM_OF_CLASSES], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

        deconv_shape2 = image_net["pool3"].get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

        shape = tf.shape(image)
        deconv_shape3 = tf.pack([shape[0], shape[1], shape[2], FLAGS.NUM_OF_CLASSES])
        W_t3 = utils.weight_variable([16, 16, FLAGS.NUM_OF_CLASSES, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([FLAGS.NUM_OF_CLASSES], name="b_t3")
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")
        print("ayy", conv_t3)

    return tf.expand_dims(annotation_pred, dim=3), conv_t3


def train(total_loss, global_step):
    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.AdamOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op

def _add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

      # Attach a scalar summary to all individual losses and the total loss; do the
      # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name +' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def loss(logits, labels, weight=100):
    return tf.reduce_mean((tf.nn.weighted_cross_entropy_with_logits(logits, labels, pos_weight=weight, name="entropy")))


# def main(argv=None):
#     keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
#     image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
#     annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")

#     pred_annotation, logits = inference(image, keep_probability)
#     tf.summary.image("input_image", image, max_outputs=2)
#     tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=2)
#     tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=2)
#     loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits,
#                                                                           tf.squeeze(annotation, squeeze_dims=[3]),
#                                                                           name="entropy")))
#     tf.summary.scalar("entropy", loss)

#     trainable_var = tf.trainable_variables()
#     if FLAGS.debug:
#         for var in trainable_var:
#             utils.add_to_regularization_and_summary(var)
#     train_op = train(loss, trainable_var)

#     print("Setting up summary op...")
#     summary_op = tf.summary.merge_all()

#     print("Setting up image reader...")
#     train_records, valid_records = scene_parsing.read_dataset(FLAGS.data_dir)
#     print(len(train_records))
#     print(len(valid_records))

#     print("Setting up dataset reader")
#     image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
#     if FLAGS.mode == 'train':
#         train_dataset_reader = dataset.BatchDatset(train_records, image_options)
#     validation_dataset_reader = dataset.BatchDatset(valid_records, image_options)

#     sess = tf.Session()

#     print("Setting up Saver...")
#     saver = tf.train.Saver()
#     summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)

#     sess.run(tf.initialize_all_variables())
#     ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
#     if ckpt and ckpt.model_checkpoint_path:
#         saver.restore(sess, ckpt.model_checkpoint_path)
#         print("Model restored...")

#     if FLAGS.mode == "train":
#         for itr in xrange(MAX_ITERATION):
#             train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)
#             feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.85}

#             sess.run(train_op, feed_dict=feed_dict)

#             if itr % 10 == 0:
#                 train_loss, summary_str = sess.run([loss, summary_op], feed_dict=feed_dict)
#                 print("Step: %d, Train_loss:%g" % (itr, train_loss))
#                 summary_writer.add_summary(summary_str, itr)

#             if itr % 500 == 0:
#                 valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.batch_size)
#                 valid_loss = sess.run(loss, feed_dict={image: valid_images, annotation: valid_annotations,
#                                                        keep_probability: 1.0})
#                 print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))
#                 saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)

#     elif FLAGS.mode == "visualize":
#         valid_images, valid_annotations = validation_dataset_reader.get_random_batch(FLAGS.batch_size)
#         pred = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations,
#                                                     keep_probability: 1.0})
#         valid_annotations = np.squeeze(valid_annotations, axis=3)
#         pred = np.squeeze(pred, axis=3)

#         for itr in range(FLAGS.batch_size):
#             utils.save_image(valid_images[itr].astype(np.uint8), FLAGS.logs_dir, name="inp_" + str(5+itr))
#             utils.save_image(valid_annotations[itr].astype(np.uint8), FLAGS.logs_dir, name="gt_" + str(5+itr))
#             utils.save_image(pred[itr].astype(np.uint8), FLAGS.logs_dir, name="pred_" + str(5+itr))
#             print("Saved image: %d" % itr)


# if __name__ == "__main__":
#     tf.app.run()
