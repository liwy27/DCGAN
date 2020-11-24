import tensorflow as tf
from main import get_generator, plot_images, show_generator_output


# 定义参数
noise_size = 100
n_samples = 2
beta1 = 0.5
learning_rate = 0.0001

def predict(noise_size, n_samples):
    """
    @param noise_size: 噪声size
    @param data_shape: 真实图像shape
    @batch_size:
    @n_samples: 显示示例图片数量
    """
    inputs_noise = tf.placeholder(tf.float32, [None, noise_size], name='inputs_noise')
    g_outputs = get_generator(inputs_noise, 3, is_train=True)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "./model/model.ckpt-201")
        samples = show_generator_output(sess, n_samples, inputs_noise, 3)
        plot_images(samples, n_samples)

if __name__ == '__main__':
    # collect_data(FILEPATH)
    with tf.Graph().as_default():
        predict(noise_size, n_samples)


