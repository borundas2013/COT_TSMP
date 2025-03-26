import tensorflow as tf

from datetime import datetime
class PretrainedDecoder(tf.keras.layers.Layer):
        def __init__(self, max_length,pretrained_decoder_gru,pretrained_decoder_dense, **kwargs):
            super().__init__(**kwargs)
            self.pretrained_decoder_gru = pretrained_decoder_gru
            self.pretrained_decoder_dense = pretrained_decoder_dense
            self.max_length = max_length
            self.context_projection = tf.keras.layers.Dense(512) #64
            self.attention = tf.keras.layers.MultiHeadAttention(
                num_heads=8, 
                key_dim=64 # 64
            )
            self.layer_norm1 = tf.keras.layers.LayerNormalization()
            self.layer_norm2 = tf.keras.layers.LayerNormalization()

        
        def call(self, inputs, training=None):
            encoder_output, group_aware, relationship = inputs
            batch_size = tf.shape(encoder_output)[0]
            
            # Initial hidden state
            hidden_state = tf.zeros([batch_size, self.pretrained_decoder_gru.units])
            
            # Expand dimensions for time steps
            hidden_state = tf.expand_dims(hidden_state, axis=1)  # [batch, 1, units]
            
                # Process with attention
            attended_context = self.attention(
                query=encoder_output,
                key=encoder_output,
                value=encoder_output
            )
            combined = self.layer_norm1(encoder_output + attended_context)
        
        # Combine with group and relationship info
            context = tf.keras.layers.Concatenate(axis=-1)([
                combined,
                group_aware,
                relationship
               ])
        
        # Project and normalize
            projected_context = self.context_projection(context)  # [batch, time, 128]
            projected_context = self.layer_norm2(projected_context)
        
        # Process with GRU - maintaining time dimension
            gru_output = self.pretrained_decoder_gru(
                projected_context,
                initial_state=hidden_state[:, 0, :]  # Remove time dimension for initial state
            )
        
        # Get logits
            logits = self.pretrained_decoder_dense(gru_output)  # [batch, time, vocab_size]
        
            if training:
                logits /= 0.8  # Temperature scaling during training
                
            return logits
    
        def get_config(self):
            config = super().get_config()
            config.update({
                "max_length": self.max_length
            })
            return config  
        
        @classmethod
        def from_config(cls, config):
            return cls(**config)
        
class PretrainedEncoder(tf.keras.layers.Layer):
    def __init__(self, max_length, pretrained_encoder, **kwargs):
        super().__init__(**kwargs)
        self.pretrained_encoder = pretrained_encoder
        self.max_length = max_length
        # Add padding to match input dimensions
        self.pad_layer = tf.keras.layers.Dense(136) #72

    def call(self, inputs):
        # Pad inputs to match expected dimensions
        padded_inputs = self.pad_layer(inputs)
        # Pass through pretrained encoder
        return self.pretrained_encoder(padded_inputs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.pretrained_encoder.units)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "max_length": self.max_length
        })
        return config  
        
    @classmethod
    def from_config(cls, config):
        return cls(**config)
     
