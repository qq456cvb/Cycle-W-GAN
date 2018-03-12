import tensorflow as tf
import tensorflow.contrib.slim as slim


def identity_block(input, channel_in, channel_out, kernel_size=3, stride=1, is_training=True):
    a = tf.nn.relu(input)
    a = slim.batch_norm(inputs=a, is_training=is_training)
    a = slim.conv2d(inputs=a, num_outputs=channel_in, kernel_size=1, stride=stride, activation_fn=None)

    b = tf.nn.relu(a)
    b = slim.batch_norm(inputs=b, is_training=is_training)
    b = slim.conv2d(inputs=b, num_outputs=channel_in, kernel_size=kernel_size, stride=1, activation_fn=None)

    c = tf.nn.relu(b)
    c = slim.batch_norm(inputs=c, is_training=is_training)
    c = slim.conv2d(inputs=c, num_outputs=channel_out, kernel_size=1, stride=1, activation_fn=None)
    return c + input


def down_sample_block(input, channel_in, channel_out, kernel_size=3, stride=2, is_training=True):
    a1 = tf.nn.relu(input)
    a1 = slim.batch_norm(inputs=a1, is_training=is_training)
    a1 = slim.conv2d(inputs=a1, num_outputs=channel_in, kernel_size=1, stride=stride, activation_fn=None)

    b1 = tf.nn.relu(a1)
    b1 = slim.batch_norm(inputs=b1, is_training=is_training)
    b1 = slim.conv2d(inputs=b1, num_outputs=channel_in, kernel_size=kernel_size, stride=1, activation_fn=None)

    c1 = tf.nn.relu(b1)
    c1 = slim.batch_norm(inputs=c1, is_training=is_training)
    c1 = slim.conv2d(inputs=c1, num_outputs=channel_out, kernel_size=1, stride=1, activation_fn=None)

    a2 = tf.nn.relu(input)
    a2 = slim.batch_norm(inputs=a2, is_training=is_training)
    a2 = slim.conv2d(inputs=a2, num_outputs=channel_out, kernel_size=1, stride=stride, activation_fn=None)

    return c1 + a2


def up_sample_block(input, channel_in, channel_out, kernel_size=2, stride=2, is_training=True):
    a1 = tf.nn.relu(input)
    a1 = slim.batch_norm(inputs=a1, is_training=is_training)
    a1 = slim.conv2d(inputs=a1, num_outputs=channel_in, kernel_size=1, stride=1, activation_fn=None)

    b1 = tf.nn.relu(a1)
    b1 = slim.batch_norm(inputs=b1, is_training=is_training)
    b1 = slim.conv2d_transpose(inputs=b1, num_outputs=channel_in, kernel_size=kernel_size, stride=stride, activation_fn=None)

    c1 = tf.nn.relu(b1)
    c1 = slim.batch_norm(inputs=c1, is_training=is_training)
    c1 = slim.conv2d(inputs=c1, num_outputs=channel_out, kernel_size=1, stride=1, activation_fn=None)

    a2 = tf.nn.relu(input)
    a2 = slim.batch_norm(inputs=a2, is_training=is_training)
    a2 = slim.conv2d_transpose(inputs=a2, num_outputs=channel_out, kernel_size=kernel_size, stride=stride, activation_fn=None)

    return c1 + a2
