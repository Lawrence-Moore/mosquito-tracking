import math

import tensorflow as tf

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


def inference(images):
    """.
    Args:
    images: Images placeholder, from inputs().
    hidden1_units: Size of the first hidden layer.
    hidden2_units: Size of the second hidden layer.
    Returns:
    softmax_linear: Output tensor with the computed logits.
    """
    # 32 5x5 filters
    conv1 = conv(images, "conv1", [5, 5, 3, 32], [1, 2, 2, 1])
    pool1 = max_pool(conv1, "pool1", 5)

    conv2 = conv(pool1, "conv2", [5, 5, 32, 32], [1, 2, 2, 1])
    pool2 = max_pool(conv2, "pool2", 5)

    conv3 = conv(pool2, "conv3", [5, 5, 32, 32])
    pool3 = max_pool(conv3, "pool3", 5)

    conv4 = conv(pool3, "conv4", [5, 5, 32, 32])
    pool4 = max_pool(conv4, "pool4", 5)

    conv5 = conv(conv4, "conv5", [17, 17, 32, 32])
    pool5 = max_pool(conv5, "pool5", 17)

    print "BRRRUH", conv4.get_shape()

    fc1 = conv(pool5, "fconv1", [1, 1, 32, 64])
    fc2 = conv(fc1, "fconv2", [1, 1, 64, 64])

    print "sluro", fc2.get_shape()

    # orig_shape=images.get_shape()
    heatmaps = deconv(fc2, "strided_conv", filter_shape=[16, 16, 64, 1], num_filters=64, stride=2)

    print "maps", heatmaps.get_shape()

    return heatmaps 


def conv(input, name, shape, strides=[1, 1, 1, 1]):
    with tf.variable_scope(name):

        # get the weights and biases
        kernel = tf.get_variable(name + "_weights",
                                 shape=shape,
                                 dtype=tf.float32, 
                                 initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32))

        biases = tf.get_variable(name + "_biases",
                                 shape=shape[3],
                                 dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.0))

        # convolve
        conv = tf.nn.conv2d(input, kernel, strides, padding='SAME')
        bias = tf.nn.bias_add(conv, biases)

        # relu
        relu = tf.nn.relu(bias)
        return relu


def deconv(input, name, filter_shape, num_filters, stride=1):
    with tf.variable_scope(name):

        static_input_shape = input.get_shape().as_list()
        dyn_input_shape = tf.shape(input)

        batch_size = dyn_input_shape[0]

        print "batch_size", "static_input_shape", batch_size, static_input_shape

        kernel = tf.get_variable(name + "_weights",
                            shape=[filter_shape[0], filter_shape[1], filter_shape[2], static_input_shape[3]],
                            initializer=tf.truncated_normal_initializer(stddev=5e-2))

        out_h = dyn_input_shape[1] * stride
        out_w = dyn_input_shape[2] * stride

        out_shape = tf.pack([batch_size, out_h, out_w, 1])

        print "out shape", out_shape.get_shape()

        # get the weights and biases
        # kernel = tf.get_variable(name + "_weights",
        #                          shape=filter_shape,
        #                          dtype=tf.float32, 
        #                          initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32))


        # convolve
        conv = tf.nn.conv2d_transpose(input, kernel, out_shape, [1, stride, stride, 1], padding='SAME')

        b = tf.get_variable(
            name + '_b', [num_filters],
            initializer=tf.constant_initializer(0.0))

        # relu
        return conv + b


def max_pool(bottom, name, size):
    return tf.nn.max_pool(bottom, ksize=[1, size, size, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def loss(logits, labels):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels)
    loss = tf.reduce_mean(cross_entropy)
    return loss

def _add_loss_summaries(total_loss):
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
    tf.scalar_summary('learning_rate', lr)

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

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
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
  dtype = tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var