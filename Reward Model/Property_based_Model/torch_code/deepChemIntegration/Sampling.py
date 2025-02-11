import tensorflow as tf
from tensorflow.keras import layers

class Sampling(layers.Layer):
    def __init__(self, temperature=1.0, **kwargs):
        super(Sampling, self).__init__(**kwargs)
        self.temperature = temperature

    def call(self, inputs):
        """
        Sample points from the latent space for both molecules.
        
        Args:
            inputs: List containing [z_mean, z_log_var, z_mean_1, z_log_var_1]
                z_mean, z_mean_1: Mean vectors for both molecules
                z_log_var, z_log_var_1: Log variance vectors for both molecules
        
        Returns:
            Two sampled points from the latent distributions
        """
        z_mean, z_log_var, z_mean_1, z_log_var_1 = inputs
        
        # Get batch size and latent dimension
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        
        # Sample random points from normal distribution
        epsilon = tf.random.normal(shape=(batch, dim)) 
        epsilon_1 = tf.random.normal(shape=(batch, dim)) 
        
        # Reparameterization trick
        sampled_z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        sampled_z_1 = z_mean_1 + tf.exp(0.5 * z_log_var_1) * epsilon_1
        
        return sampled_z, sampled_z_1
