import tensorflow as tf
import tensorflow.contrib.slim as slim
from res_utils import identity_block, down_sample_block


def forward(input, scope, is_training, feature_channels=int(32)):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        x = slim.conv2d(input, feature_channels, 7, 2, activation_fn=None)

        feature_channels //= 4
        x = identity_block(x, feature_channels, feature_channels * 4, is_training=is_training)
        feature_channels *= 2
        x = down_sample_block(x, feature_channels, feature_channels * 4, is_training=is_training)

        x = identity_block(x, feature_channels, feature_channels * 4, is_training=is_training)
        feature_channels *= 2
        x = down_sample_block(x, feature_channels, feature_channels * 4, is_training=is_training)

        x = identity_block(x, feature_channels, feature_channels * 4, is_training=is_training)
        feature_channels *= 2
        x = down_sample_block(x, feature_channels, feature_channels * 4, is_training=is_training)

        x = identity_block(x, feature_channels, feature_channels * 4, is_training=is_training)
        feature_channels *= 2
        x = down_sample_block(x, feature_channels, feature_channels * 4, is_training=is_training)

        x = tf.reduce_mean(x, [1, 2])
        x = slim.fully_connected(x, 256)
        val = slim.fully_connected(x, 1, activation_fn=None)
    return val
