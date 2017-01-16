import math

import tensorflow as tf
import logging
from math import ceil
import numpy as np
import sys
import os
from tensorflow.python.ops import rnn, rnn_cell


FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

MOVING_AVERAGE_DECAY = 0.9999
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.
display_step = 10
learning_rate = 0.001
training_iters = 100000


class SimpleNet:

    def inference(self, images, train=False, debug=False):

        self.conv1 = self._conv_layer(images, "conv1", [5, 5, 3, 32], [1, 2, 2, 1])
        self.pool1 = self._max_pool(self.conv1, 'pool1', 5, 2)

        self.conv2 = self._conv_layer(self.pool1, "conv2", [5, 5, 32, 32], [1, 2, 2, 1])
        self.pool2 = self._max_pool(self.conv2, 'pool2', 5, 2)

        self.conv3 = self._conv_layer(self.pool2, "conv3", [5, 5, 32, 32])
        self.pool3 = self._max_pool(self.conv3, 'pool3', 5, 1)

        self.conv4 = self._conv_layer(self.pool3, "conv4", [5, 5, 32, 32])
        self.pool4 = self._max_pool(self.conv4, 'pool4', 5, 1)

        self.conv5 = self._conv_layer(self.pool4, "conv5",  [17, 17, 32, 32])
        self.pool5 = self._max_pool(self.conv5, 'pool5', 17, 1)


        self.fc1 = self._conv_layer(self.pool4, "fc1", [1, 1, 32, 64])
        self.fc2 = self._conv_layer(self.fc1, "fc2", [1, 1, 64, 64])

        self.heatmap = self.deconv(self.fc2, "deconv")

        return tf.cast(self.heatmap, tf.float32)

    def _max_pool(self, bottom, name, ksize, stride):
        return tf.nn.max_pool(bottom, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding='SAME', name=name)

    def _conv_layer(self, bottom, name, shape, strides=[1, 1, 1, 1]):
        with tf.variable_scope(name) as scope:

            # get the weights and biases
            kernel = tf.get_variable(name + "_weights",
                                 shape=shape,
                                 dtype=tf.float32, 
                                 initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32))

            biases = tf.get_variable(name + "_biases",
                                 shape=shape[3],
                                 dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.0))

            conv = tf.nn.conv2d(bottom, kernel, strides, padding='SAME')

            bias = tf.nn.bias_add(conv, biases)

            relu = tf.nn.relu(bias)
            # Add summary to Tensorboard
            self._activation_summary(relu)
            return relu


    def deconv(self, input, name, stride=16, ksize=4):
        with tf.variable_scope(name):

            kernel = tf.get_variable(name + "_weights",
                                     shape=[ksize, ksize, 3, 64],
                                     dtype=tf.float32, 
                                     initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32))
            output_shape = input.get_shape().as_list()
            output_shape[1] *= stride
            output_shape[2] *= stride
            output_shape[3] = kernel.get_shape().as_list()[2]

            output_shape = tf.pack([tf.shape(input)[0], output_shape[1], output_shape[2], output_shape[3]])
            return tf.nn.conv2d_transpose(input, kernel, output_shape, [1, stride, stride, 1], padding='VALID')


    def _variable_with_weight_decay(self, shape, stddev, wd):
        """Helper to create an initialized Variable with weight decay.

        Note that the Variable is initialized with a truncated normal
        distribution.
        A weight decay is added only if one is specified.

        Args:
          name: name of the variable
          shape: list of ints
          stddev: standard deviation of a truncated Gaussian
          wd: add L2Loss weight decay multiplied by this float. If None, weight
              decay is not added for this Variable.

        Returns:
          Variable Tensor
        """

        initializer = tf.truncated_normal_initializer(stddev=stddev)
        var = tf.get_variable('weights', shape=shape,
                              initializer=initializer)

        if wd and (not tf.get_variable_scope().reuse):
            weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def _bias_variable(self, shape, constant=0.0):
        initializer = tf.constant_initializer(constant)
        return tf.get_variable(name='biases', shape=shape,
                               initializer=initializer)


    def loss(self, logits, labels):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels)
        loss = tf.reduce_mean(cross_entropy)
        return loss

    def _add_loss_summaries(self, total_loss):
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        losses = tf.get_collection('losses')
        loss_averages_op = loss_averages.apply(losses + [total_loss])

          # Attach a scalar summary to all individual losses and the total loss; do the
          # same for the averaged version of the losses.
        for l in losses + [total_loss]:
            # Name each loss as '(raw)' and name the moving average version of the loss
            # as the original loss name.
            tf.scalar_summary(l.op.name +' (raw)', l)
            tf.scalar_summary(l.op.name, loss_averages.average(l))

        return loss_averages_op


    def train(self, total_loss, global_step):
        # Variables that affect learning rate.
        num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
        decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                      global_step,
                                      decay_steps,
                                      LEARNING_RATE_DECAY_FACTOR,
                                      staircase=True)
        tf.scalar_summary('learning_rate', lr)

        # Generate moving averages of all losses and associated summaries.
        loss_averages_op = self._add_loss_summaries(total_loss)

        # Compute gradients.
        with tf.control_dependencies([loss_averages_op]):
            opt = tf.train.AdamOptimizer(lr)
            grads = opt.compute_gradients(total_loss)

        # Apply gradients.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            tf.histogram_summary(var.op.name, var)

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                tf.histogram_summary(var.op.name + '/gradients', grad)

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
            train_op = tf.no_op(name='train')

        return train_op


    def _activation_summary(self, x):
        """Helper to create summaries for activations.

        Creates a summary that provides a histogram of activations.
        Creates a summary that measure the sparsity of activations.

        Args:
          x: Tensor
        Returns:
          nothing
        """
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        tensor_name = x.op.name
        # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
        tf.histogram_summary(tensor_name + '/activations', x)
        tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

