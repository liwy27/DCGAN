import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt


def collect_data(file_path):
    data = []
    for root, folder, file in os.walk(file_path):
        for i in file:
            img_name = "%s/%s" % (root, i)
            print(img_name)
            image = np.array(Image.open(img_name).resize((128, 128))).astype(np.float32) / 127.5 - 1
            data.append(image)
    return np.array(data)


def get_inputs(noise_dim, image_height, image_width, image_depth):
    """
    @Author: Nelson Zhao
    --------------------
    :param noise_dim: 噪声图片的size
    :param image_height: 真实图像的height
    :param image_width: 真实图像的width
    :param image_depth: 真实图像的depth
    """
    inputs_real = tf.placeholder(tf.float32, [None, image_height, image_width, image_depth], name='inputs_real')
    inputs_noise = tf.placeholder(tf.float32, [None, noise_dim], name='inputs_noise')

    return inputs_real, inputs_noise


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


def get_discriminator(inputs_img, reuse=False, alpha=0.2):
    """
    @Author: Nelson Zhao
    --------------------
    @param inputs_img: 输入图片，tensor类型
    @param alpha: Leaky ReLU系数
    """

    with tf.variable_scope("discriminator", reuse=reuse):
        # 128 x 128 x 3 to 16 x 16 x 128
        # 第一层不加入BN
        layer1 = tf.layers.conv2d(inputs_img, 16, 3, strides=2, padding='same')
        layer1 = tf.maximum(alpha * layer1, layer1)
        layer1 = tf.nn.dropout(layer1, keep_prob=0.5)

        # 16 x 16 x 128 to 8 x 8 x 256
        layer2 = tf.layers.conv2d(layer1, 32, 3, strides=1, padding='same')
        layer2 = tf.layers.batch_normalization(layer2, training=True)
        layer2 = tf.maximum(alpha * layer2, layer2)
        layer2 = tf.nn.dropout(layer2, keep_prob=0.5)

        # 8 x 8 x 256 to 4 x 4 x 512
        layer3 = tf.layers.conv2d(layer2, 64, 3, strides=2, padding='same')
        layer3 = tf.layers.batch_normalization(layer3, training=True)
        layer3 = tf.maximum(alpha * layer3, layer3)
        layer3 = tf.nn.dropout(layer3, keep_prob=0.8)

        layer4 = tf.layers.conv2d(layer3, 128, 3, strides=1, padding='same')
        layer4 = tf.layers.batch_normalization(layer4, training=True)
        layer4 = tf.maximum(alpha * layer4, layer4)
        layer4 = tf.nn.dropout(layer4, keep_prob=0.5)

        layer5 = tf.layers.conv2d(layer4, 256, 3, strides=2, padding='same')
        layer5 = tf.layers.batch_normalization(layer5, training=True)
        layer5 = tf.maximum(alpha * layer5, layer5)
        layer5 = tf.nn.dropout(layer5, keep_prob=0.5)

        layer6 = tf.layers.conv2d(layer5, 512, 3, strides=1, padding='same')
        layer6 = tf.layers.batch_normalization(layer6, training=True)
        layer6 = tf.maximum(alpha * layer6, layer6)
        layer6 = tf.nn.dropout(layer6, keep_prob=0.5)

        # 4 x 4 x 512 to 4*4*512 x 1
        flatten = tf.reshape(layer6, (-1, 16 * 16 * 512))
        logits = tf.layers.dense(flatten, 1)
        outputs = tf.sigmoid(logits)

        return logits, outputs


def get_loss(inputs_real, inputs_noise, image_depth, smooth=0.1):
    """
    @Author: Nelson Zhao
    --------------------
    @param inputs_real: 输入图片，tensor类型
    @param inputs_noise: 噪声图片，tensor类型
    @param image_depth: 图片的depth（或者叫channel）
    @param smooth: label smoothing的参数
    """

    g_outputs = get_generator(inputs_noise, image_depth, is_train=True)
    d_logits_real, d_outputs_real = get_discriminator(inputs_real)
    d_logits_fake, d_outputs_fake = get_discriminator(g_outputs, reuse=True)

    # 计算Loss
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                    labels=tf.ones_like(d_outputs_fake) * (1 - smooth)))

    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                                         labels=tf.ones_like(d_outputs_real) * (
                                                                                     1 - smooth)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                         labels=tf.zeros_like(d_outputs_fake)))
    d_loss = tf.add(d_loss_real, d_loss_fake)

    return g_loss, d_loss


def get_optimizer(g_loss, d_loss, beta1=0.5, learning_rate=0.0002):
    """
    @Author: Nelson Zhao
    --------------------
    @param g_loss: Generator的Loss
    @param d_loss: Discriminator的Loss
    @learning_rate: 学习率
    """

    train_vars = tf.trainable_variables()

    g_vars = [var for var in train_vars if var.name.startswith("generator")]
    d_vars = [var for var in train_vars if var.name.startswith("discriminator")]

    # Optimizer
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        g_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)
        d_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)

    return g_opt, d_opt


def plot_images(samples, n_samples):
    samples = (samples + 1) / 2

    fig, axes = plt.subplots(nrows=1, ncols=n_samples, sharex=True, sharey=True, figsize=(128, 128))
    for img, ax in zip(samples, axes):
        ax.imshow(img.reshape((128, 128, 3)), cmap='Greys_r')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    fig.tight_layout(pad=0)
    plt.savefig('./data.png')
    plt.close()


def show_generator_output(sess, n_images, inputs_noise, output_dim):
    """
    @Author: Nelson Zhao
    --------------------
    @param sess: TensorFlow session
    @param n_images: 展示图片的数量
    @param inputs_noise: 噪声图片
    @param output_dim: 图片的depth（或者叫channel）
    @param image_mode: 图像模式：RGB或者灰度
    """
    cmap = 'Greys_r'
    noise_shape = inputs_noise.get_shape().as_list()[-1]
    # 生成噪声图片
    examples_noise = np.random.uniform(-1, 1, size=[n_images, noise_shape])

    samples = sess.run(get_generator(inputs_noise, output_dim, False),
                       feed_dict={inputs_noise: examples_noise})
    return samples


# 定义参数
batch_size = 100
noise_size = 100
epochs = 50000
n_samples = 2
learning_rate = 0.0002
beta1 = 0.5
FILEPATH = "../mobilenetv3/data/train/dog"


def train(noise_size, data_shape, batch_size, n_samples):
    """
    @Author: Nelson Zhao
    --------------------
    @param noise_size: 噪声size
    @param data_shape: 真实图像shape
    @batch_size:
    @n_samples: 显示示例图片数量
    """

    # 存储loss
    losses = []
    steps = 0

    inputs_real, inputs_noise = get_inputs(noise_size, data_shape[1], data_shape[2], data_shape[3])
    g_loss, d_loss = get_loss(inputs_real, inputs_noise, data_shape[-1])
    g_train_opt, d_train_opt = get_optimizer(g_loss, d_loss, beta1, learning_rate)
    images = collect_data(FILEPATH)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 迭代epoch
        for e in range(epochs):
            for batch_i in range(images.shape[0] // batch_size):
                steps += 1
                batch_images = images[batch_i * batch_size: (batch_i + 1) * batch_size]

                # scale to -1, 1
                # batch_images = batch_images * 2 - 1

                # noise
                batch_noise = np.random.uniform(-1, 1, size=(batch_size, noise_size))

                # run optimizer
                _ = sess.run(g_train_opt, feed_dict={inputs_real: batch_images,
                                                     inputs_noise: batch_noise})
                _ = sess.run(d_train_opt, feed_dict={inputs_real: batch_images,
                                                     inputs_noise: batch_noise})

                if steps % 100 == 0:
                    train_loss_d = d_loss.eval({inputs_real: batch_images,
                                                inputs_noise: batch_noise})
                    train_loss_g = g_loss.eval({inputs_real: batch_images,
                                                inputs_noise: batch_noise})
                    losses.append((train_loss_d, train_loss_g))
                    # 显示图片
                    samples = show_generator_output(sess, n_samples, inputs_noise, data_shape[-1])
                    plot_images(samples, n_samples)
                    print("Epoch {}/{}....".format(e + 1, epochs),
                          "Discriminator Loss: {:.4f}....".format(train_loss_d),
                          "Generator Loss: {:.4f}....".format(train_loss_g))


if __name__ == '__main__':
    # collect_data(FILEPATH)
    with tf.Graph().as_default():
        train(noise_size, [-1, 128, 128, 3], batch_size, n_samples)


