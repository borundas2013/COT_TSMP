import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Concatenate, Dropout,Add
from tensorflow.keras.models import Model
import numpy as np

# ================================
# Noise Scheduler Class
# ================================
class NoiseScheduler:
    def __init__(self, num_steps):
        self.num_steps = num_steps
        self.betas = np.linspace(0.01, 0.2, num_steps)

    def add_noise(self, x, t):
        """
        Add noise to input tensor
        Args:
            x: input tensor of shape (batch_size, sequence_length)
            t: timestep (integer tensor)
        """
        # Convert t to proper index and ensure it's a tensor
        t = tf.cast(t, tf.int32)

        # Get beta value for timestep t and ensure it's float32
        beta = tf.gather(self.betas, t)
        beta = tf.cast(beta, tf.float32)

        # ✅ Reshape beta to (batch_size, 1) to align with input shape
        if tf.rank(x) == 2:
            beta = tf.reshape(beta, [-1, 1])  # For 2D input
        elif tf.rank(x) == 3:
            beta = tf.reshape(beta, [-1, 1, 1])  # For 3D input

        # ✅ Generate noise with same shape and dtype as input
        noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=1.0, dtype=x.dtype)

        # ✅ Apply noise (maintain shape)
        noisy_x = tf.sqrt(1.0 - beta) * tf.cast(x, tf.float32) + tf.sqrt(beta) * noise
        
        return noisy_x, noise

    # def add_noise(self, x, t):
    #     beta = self.betas[t]
    #     noise = tf.random.normal(shape=tf.shape(x))
    #     noisy_x = tf.sqrt(1 - beta) * x + tf.sqrt(beta) * noise
    #     return noisy_x, noise

# ================================
# Diffusion Model Class
# ================================
class DiffusionModel(tf.keras.Model):
    def __init__(self, pretrained_model,num_steps=1000,vocab_size=None,embedding_dim=None,loss_fn=None):
        super(DiffusionModel, self).__init__()
        self.pretrained_model = pretrained_model
        self.num_steps = num_steps
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.loss_fn = loss_fn
        self.pretrained_embedding = pretrained_model.get_layer("embedding")
        self.pretrained_encoder = pretrained_model.get_layer("gru")
        self.pretrained_decoder_gru = pretrained_model.get_layer("gru_2")
        self.pretrained_dense = pretrained_model.get_layer("dense")

        self.scheduler = NoiseScheduler(num_steps)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, inputs, t, training=False):
        monomer1_input, monomer2_input, group_input, decoder_input1, decoder_input2 = inputs

        # ========= Step 1: Embed Inputs =========
        monomer1_embed = self.pretrained_embedding(monomer1_input)
        monomer2_embed = self.pretrained_embedding(monomer2_input)

        if training:
            monomer1_embed, noise1 = self.scheduler.add_noise(monomer1_embed, t)
            monomer2_embed, noise2 = self.scheduler.add_noise(monomer2_embed, t)

    

        monomer1_dense_embedding = Dense(136, activation="relu", name="monomer1_dense_embedding")(monomer1_embed)
        monomer2_dense_embedding = Dense(136, activation="relu", name="monomer2_dense_embedding")(monomer2_embed)

        # Use pretrained encoder directly
        encoder_output1 = self.pretrained_encoder(monomer1_dense_embedding)  # Shape: (batch, max_length, 512)
        encoder_output2 = self.pretrained_encoder(monomer2_dense_embedding)  # Shape: (batch, max_length, 512)

        # Extract last state manually if return_state=False
        encoder_state1 = encoder_output1[:, -1, :]  # Shape: (batch, 512)
        encoder_state2 = encoder_output2[:, -1, :]  # Shape: (batch, 512)

        # Project encoder state to match decoder size (128)
        encoder_state1_projected = Dense(128, activation="relu", name="encoder_projection1")(encoder_state1)
        encoder_state2_projected = Dense(128, activation="relu", name="encoder_projection2")(encoder_state2)

        # Project group input to size 128
        group_projected = Dense(128, activation="relu", name="group_projection")(group_input)  # Shape: (batch, 128)

        # Combine encoder state + group state using addition
        modified_state1 = Add(name="modified_state1")([encoder_state1_projected, group_projected])
        modified_state2 = Add(name="modified_state2")([encoder_state2_projected, group_projected])

        modified_state1 = Dense(512, activation="relu", name="state_projection1")(modified_state1)
        modified_state2 = Dense(512, activation="relu", name="state_projection2")(modified_state2)

        # Decoder embeddings
        decoder_embedded1 = self.pretrained_embedding(decoder_input1)  # Shape: (batch, max_length, 128)
        decoder_embedded2 = self.pretrained_embedding(decoder_input2)  # Shape: (batch, max_length, 128)

        decoder_dense_embedded1 = Dense(136, activation="relu", name="decoder_dense_embedding1")(decoder_embedded1)
        decoder_dense_embedded2 = Dense(136, activation="relu", name="decoder_dense_embedding2")(decoder_embedded2)

        # Project to correct size (128) before passing to GRU
        decoder_input1_combined = Dense(128, activation="relu", name="decoder_projection1")(decoder_dense_embedded1)
        decoder_input2_combined = Dense(128, activation="relu", name="decoder_projection2")(decoder_dense_embedded2)

       
        decoder_output1 = self.pretrained_decoder_gru(decoder_input1_combined, initial_state=modified_state1)
        decoder_output2 = self.pretrained_decoder_gru(decoder_input2_combined, initial_state=modified_state2)
       

        # Add attention layer here
        attention_layer1 = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=128)
        attention_layer2 = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=128)
        
        # Apply attention between decoder output and encoder output
        context_vector1 = attention_layer1(query=decoder_output1, value=encoder_output1,key=encoder_output1)  # Using encoder_output1 from monomer1
        context_vector2 = attention_layer2(query=decoder_output2, value=encoder_output2,key=encoder_output2)  # Using encoder_output2 from monomer2

        # Combine context vectors with decoder outputs
        decoder_output1 = tf.keras.layers.Concatenate()([decoder_output1, context_vector1])
        decoder_output2 = tf.keras.layers.Concatenate()([decoder_output2, context_vector2])

        decoder_output1 = tf.keras.layers.Dense(512, activation="relu", name="decoder_output1")(decoder_output1)
        decoder_output2 = tf.keras.layers.Dense(512, activation="relu", name="decoder_output2")(decoder_output2)

        # Add dropout
        decoder_output1 = Dropout(0.1, name='dropout1')(decoder_output1)
        decoder_output2 = Dropout(0.1, name='dropout2')(decoder_output2)

        # Create two separate final dense layers with same weights as pretrained_dense
        final_dense1 = Dense(units=self.pretrained_dense.units, activation=self.pretrained_dense.activation, name='final_dense1')
        final_dense2 = Dense(units=self.pretrained_dense.units, activation=self.pretrained_dense.activation, name='final_dense2')

        final_dense1.build(decoder_output1.shape)
        final_dense2.build(decoder_output2.shape)
        final_dense1.set_weights(self.pretrained_dense.get_weights())
        final_dense2.set_weights(self.pretrained_dense.get_weights())

        # Final outputs
        output1 = final_dense1(decoder_output1)
        output2 = final_dense2(decoder_output2)

        return output1, output2
    
    def train_step(self, data):
        """
        Modified train_step to handle the correct data format
        """
        if isinstance(data, tuple):
            inputs, targets = data
        else:
            inputs = data
            targets = None

        # Unpack inputs
        monomer1_input = tf.cast(inputs['monomer1_input'], tf.float32)
        monomer2_input = tf.cast(inputs['monomer2_input'], tf.float32)
        group_input = tf.cast(inputs['group_input'], tf.float32)
        decoder_input1 = tf.cast(inputs['decoder_input1'], tf.float32)
        decoder_input2 = tf.cast(inputs['decoder_input2'], tf.float32)
        original_monomer1 = tf.cast(inputs['original_monomer1'], tf.float32)
        original_monomer2 = tf.cast(inputs['original_monomer2'], tf.float32)

        # Sample random timesteps
        batch_size = tf.shape(monomer1_input)[0]
        t = tf.random.uniform(shape=(batch_size,), minval=0, maxval=self.num_steps, dtype=tf.int32)

        with tf.GradientTape() as tape:
            # Add noise to original inputs
            noisy_monomer1, noise1 = self.scheduler.add_noise(monomer1_input, t)
            noisy_monomer2, noise2 = self.scheduler.add_noise(monomer2_input, t)

            # Get model predictions
            print("Monomer 1: ", monomer1_input.shape)
            print("Monomer 2: ", monomer2_input.shape)
            print("Noisy Monomer 1: ", noisy_monomer1.shape)
            print("Noisy Monomer 2: ", noisy_monomer2.shape)
            print("Group Input: ", group_input.shape)
            print("Decoder Input 1: ", decoder_input1.shape)
            print("Decoder Input 2: ", decoder_input2.shape)
            pred_monomer1, pred_monomer2 = self([
                noisy_monomer1, 
                noisy_monomer2, 
                group_input, 
                decoder_input1, 
                decoder_input2
            ], t, training=True)

            # # Calculate losses
            # loss1 = tf.reduce_mean(tf.square(pred_monomer1 - original_monomer1))
            # loss2 = tf.reduce_mean(tf.square(pred_monomer2 - original_monomer2))
            # total_loss = loss1 + loss2
            total_loss = self.loss_fn(original_monomer1, pred_monomer1) + self.loss_fn(original_monomer2, pred_monomer2)

        # Apply gradients
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Update metrics
        self.loss_tracker.update_state(total_loss)   
        print("Total Loss: ", total_loss)

        return {
            'loss': self.loss_tracker.result(),
        }
    
    def sample(self, monomer1_input, monomer2_input, group_input, decoder_input1, decoder_input2):
        x = [monomer1_input, monomer2_input, group_input, decoder_input1, decoder_input2]

        # Start from noise and gradually denoise over diffusion steps
        for t in reversed(range(self.scheduler.num_steps)):
            pred_monomer1, pred_monomer2 = self(x, t, training=False)

            beta = self.scheduler.betas[t]

            # Denoising step
            monomer1_input = (monomer1_input - tf.sqrt(beta) * pred_monomer1) / tf.sqrt(1 - beta)
            monomer2_input = (monomer2_input - tf.sqrt(beta) * pred_monomer2) / tf.sqrt(1 - beta)

        return monomer1_input, monomer2_input
    
def build_diffusion_model(pretrained_model,num_steps=1000,vocab_size=None,embedding_dim=None,loss_fn=None):
    model = DiffusionModel(
        pretrained_model=pretrained_model,
        num_steps=num_steps,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        loss_fn=loss_fn
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.MeanSquaredError()
    )
    return model
        
        
        

