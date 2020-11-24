import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt


def get_generator(noise_img, output_dim=3, is_train=True, alpha=0.01):
    """
    @Author: Nelson Zhao
    --------------------
    :param noise_img: 噪声信号，tensor类型
    :param output_dim: 生成图片的depth
    :param is_train: 是否为训练状态，该参数主要用于作为batch_normalization方法中的参数使用
    :param alpha: Leaky ReLU系数
    """

    with tf.variable_scope("generator", reuse=(not is_train)):
        # 100 x 1 to 8 * 8 * 12
        # 全连接层
        layer1 = tf.layers.dense(noise_img, 8 * 8 * 12, activation=tf.nn.relu)
        layer1 = tf.reshape(layer1, [-1, 8, 8, 12])

        # 4 x 4 x 512 to 8 x 8 x 256
        layer2 = tf.layers.conv2d_transpose(layer1, 384, 5, strides=2, padding='same', activation=tf.nn.relu)
        layer2 = tf.layers.batch_normalization(layer2, training=is_train)

        # 8 x 8 256 to 16 x 16 x 128
        layer3 = tf.layers.conv2d_transpose(layer2, 256, 5, strides=2, padding='same', activation=tf.nn.relu)
        layer3 = tf.layers.batch_normalization(layer3, training=is_train)
        # layer3 = tf.maximum(alpha * layer3, layer3)
        # layer3 = tf.nn.dropout(layer3, keep_prob=0.8)

        layer4 = tf.layers.conv2d_transpose(layer3, 192, 5, strides=2, padding='same', activation=tf.nn.relu)
        layer4 = tf.layers.batch_normalization(layer4, training=is_train)

        # 16 x 16 x 128 to 32 x 32 x 3
        logits = tf.layers.conv2d_transpose(layer4, output_dim, 5, strides=2, padding='same')
        # MNIST原始数据集的像素范围在0-1，这里的生成图片范围为(-1,1)
        # 因此在训练时，记住要把MNIST像素范围进行resize
        outputs = tf.tanh(logits)

        return outputs

def show_generator_output(sess, n_images, inputs_noise, inputs_real, output_dim, y):
    """
    @param sess: TensorFlow session
    @param n_images: 展示图片的数量
    @param inputs_noise: 噪声图片
    @param output_dim: 图片的depth（或者叫channel）
    @param image_mode: 图像模式：RGB或者灰度
    """
    cmap = 'Greys_r'
    noise_shape = inputs_noise.get_shape().as_list()[-1]
    # 生成噪声图片
    noise = np.random.uniform(-1, 1, size=[n_images, noise_shape])
    noise = noise.astype(np.float32)
    images = np.zeros((1, 128, 128, 3))
    # samples = sess.run(get_generator(inputs_noise, output_dim, True), feed_dict={inputs_noise: noise})
    samples = sess.run(y, feed_dict={inputs_real: images, inputs_noise: noise})
    return samples


def load_model():
    with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())
        # import_meta_graph填的名字meta文件的名字
        saver = tf.train.import_meta_graph('./model/model.ckpt-201.meta')
        # 检查checkpoint，所以只填到checkpoint所在的路径下即可，不需要填checkpoint
        saver.restore(sess, "./model/model.ckpt-201")
        inputs_noise = tf.placeholder(tf.float32, [None, 100], name='inputs_noise')
        inputs_real = tf.placeholder(tf.float32, [None, 128, 128, 3], name='inputs_real')
        train_vars = tf.trainable_variables()
        a = [n.name for n in tf.get_default_graph().as_graph_def().node]
        g = tf.get_default_graph()
        m = g.get_tensor_by_name('generator/Tanh:0')
        out_img = show_generator_output(sess, 2, inputs_noise, inputs_real, 3, m)
        print(train_vars)



if __name__ == '__main__':
    load_model()