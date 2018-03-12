import numpy as np
import tensorflow as tf
import generator
import GAN_f
import GAN_g


def model_fn(img_A, img_B, training=True, train_f=True):
    cycle_loss_weight = 0.5
    optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-3)
    w_dist_A = tf.reduce_mean(GAN_f.forward(img_A, 'fA', training)
                              - GAN_f.forward(GAN_g.forward(img_B, 'gB', training), 'fA', training))
    w_dist_B = tf.reduce_mean(GAN_f.forward(img_B, 'fB', training)
                              - GAN_f.forward(GAN_g.forward(img_A, 'gA', training), 'fB', training))
    cycle_loss = tf.reduce_mean(tf.abs(GAN_g.forward(GAN_g.forward(img_A, 'gA', training), 'gB', training) - img_A))\
                 + tf.reduce_mean(tf.abs(GAN_g.forward(GAN_g.forward(img_B, 'gB', training), 'gA', training) - img_B))
    if training:
        if train_f:
            # train f to maximize wasserstein distance
            loss = - w_dist_A - w_dist_B
            return optimizer.minimize(loss,
                               var_list=tf.trainable_variables(scope='fA')+tf.trainable_variables(scope='fB')), loss
        else:
            # train g to minimize wasserstein distance
            loss = w_dist_A + w_dist_B + cycle_loss_weight * cycle_loss
            return optimizer.minimize(loss,
                               var_list=tf.trainable_variables(scope='gA')+tf.trainable_variables(scope='gB')), loss


def train_input_fn():
    gen = generator.data_generator
    dataset = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32),
                                             (
                                                 tf.TensorShape([None, None, 3]),
                                                 tf.TensorShape([None, None, 3])
                                             ))
    dataset = dataset.batch(16)
    # dataset = dataset.prefetch(1)
    next_batch = dataset.make_one_shot_iterator().get_next()
    return next_batch[0], next_batch[1]


if __name__ == '__main__':
    inputs = train_input_fn()
    train_f_op, loss_f = model_fn(inputs[0], inputs[1], True, True)
    train_g_op, loss_g = model_fn(inputs[0], inputs[1], True, False)

    n_critic = 5
    num_steps = 100
    with tf.train.MonitoredSession() as sess:
        for i in range(num_steps):
            print("step %d" % i)
            if i % (n_critic + 1) == 0:
                sess.run(train_g_op)
            else:
                sess.run(train_f_op)
