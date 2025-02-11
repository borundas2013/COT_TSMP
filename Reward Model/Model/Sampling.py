
import tensorflow as tf
from tensorflow.keras import layers
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var, z_mean_1, z_log_var_1 = inputs
        batch = tf.shape(z_log_var)[0]
        dim = tf.shape(z_log_var)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon, z_mean_1 + tf.exp(0.5 * z_log_var_1) * epsilon