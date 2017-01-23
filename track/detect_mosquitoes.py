from datetime import datetime
import os.path
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import load_data
import simple_feed_forward
import multi_frame_network

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")

tf.app.flags.DEFINE_string('train_dir', '/tmp/train_logs',
                           """Directory where to write event logs """
                           """and checkpoint.""")

def train():
    """Eval CIFAR-10 for a number of steps."""
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        net_type = 'simple'
        new_shape = (1008, 1008, 3)
        if net_type == 'simple':
            # Get images and labels for the skeets
            images_np, labels_np = load_data.load_single_frame(new_shape)

            print("images_np", images_np.shape, labels_np.shape)

            # for testing purposes:
            # images_np = labels_np

            images = tf.placeholder(tf.float32, shape=[None, images_np.shape[1], images_np.shape[2], images_np.shape[3]])
            labels = tf.placeholder(tf.float32, shape=[None, images_np.shape[1], images_np.shape[2], images_np.shape[3]])

            # Build a Graph that computes the logits predictions from the
            # inference model.
            net = simple_feed_forward.SimpleNet()
            logits = net.inference(images)

            # Calculate loss.
            loss = net.loss(logits, labels)

            # Build a Graph that trains the model with one batch of examples and
            # updates the model parameters.
            train_op = net.train(loss, global_step)
        elif net_type == 'multi-frame':
            # Get images and labels for the skeets
            frame_depth = 10
            images_np, labels_np = load_data.load_multi_frame(frame_depth, new_shape)
            images = tf.placeholder(tf.float32, shape=[None, images_np.shape[1], images_np.shape[2], images_np.shape[3], images_np.shape[4]])
            labels = tf.placeholder(tf.float32, shape=[None, images_np.shape[1], images_np.shape[2], images_np.shape[3], images_np.shape[4]])

            # Build a Graph that computes the logits predictions from the
            # inference model.
            net = multi_frame_network.MultiframeNet()
            print("shapes", images.get_shape(), tf.shape(images))
            logits = net.inference(images, frame_depth)

            # Calculate loss.
            loss = net.loss(logits, labels)

            # Build a Graph that trains the model with one batch of examples and
            # updates the model parameters.
            train_op = net.train(loss, global_step)

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
            _, loss_value = sess.run([train_op, loss], feed_dict={images: labels_np, labels: labels_np})
            duration = time.time() - start_time

            # print("got here")
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                     examples_per_sec, sec_per_batch))

                if step % 100 == 0:
                    summary_str = sess.run(summary_op, feed_dict={images: images_np, labels: labels_np})
                    summary_writer.add_summary(summary_str, step)

                # Save the model checkpoint periodically.
                if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                    checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):
    train()

if __name__ == '__main__':
    tf.app.run()
