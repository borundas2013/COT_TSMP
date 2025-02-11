import tensorflow as tf
import numpy as np
import pandas as pd
from rdkit import Chem
from Data_Process_with_prevocab import *
from LoadPreTrainedModel import *
from rdkit import DataStructs
from rdkit.Chem import AllChem
import os
import json
from datetime import datetime

@keras.saving.register_keras_serializable()
class GroupAwareLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.group_dense = None
        self.combine_dense = None
    
    def build(self, input_shape):
        features_shape, group_vec_shape = input_shape
        
        # Initialize layers in build
        self.group_dense = tf.keras.layers.Dense(
            128, 
            activation='relu',
            name='group_dense'
        )
        self.combine_dense = tf.keras.layers.Dense(
            256, 
            activation='relu',
            name='combine_dense'
        )
        
        super(GroupAwareLayer, self).build(input_shape)
    
    def call(self, inputs):
        features, group_vec = inputs
        
        # Process group vector
        group_dense = self.group_dense(group_vec)
        
        # Expand dimensions and tile
        group_dense_expanded = tf.expand_dims(group_dense, axis=1)
        group_dense_tiled = tf.tile(
            group_dense_expanded,
            [1, tf.shape(features)[1], 1]
        )
        
        # Combine features
        combined = tf.keras.layers.Concatenate(axis=-1)([features, group_dense_tiled])
        return self.combine_dense(combined)
    
    def get_config(self):
        config = super().get_config()
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

@keras.saving.register_keras_serializable()
class CombinedLoss(tf.keras.losses.Loss):
    def __init__(self, desired_groups, weights=[1.0, 0.5, 0.5], **kwargs):
        super().__init__(**kwargs)
        self.desired_groups = desired_groups
        self.weights = weights
        
    def call(self, y_true, y_pred):
        # Reconstruction loss
        recon_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        
        # RDKit validity loss
        def check_rdkit_valid(pred_logits):
            pred_tokens = tf.argmax(pred_logits, axis=-1)
            tokenizer = PreTrainedTokenizerFast.from_pretrained("code/vocab/smiles_tokenizer")
            try:
                smiles = tokenizer.decode(pred_tokens.numpy(), skip_special_tokens=True)
                mol = Chem.MolFromSmiles(smiles)
                return tf.constant(1.0) if mol is not None else tf.constant(0.0)
            except:
                return tf.constant(0.0)
        
        validity_scores = tf.map_fn(
            check_rdkit_valid,
            y_pred,
            fn_output_signature=tf.float32
        )
        valid_loss = 1.0 - tf.reduce_mean(validity_scores)
        
        # Group presence loss
        def check_groups(pred_logits):
            pred_tokens = tf.argmax(pred_logits, axis=-1)
            tokenizer = PreTrainedTokenizerFast.from_pretrained("code/vocab/smiles_tokenizer")
            try:
                smiles = tokenizer.decode(pred_tokens.numpy(), skip_special_tokens=True)
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return tf.constant(0.0)
                
                group_patterns = {
                    "C=C": "C=C",
                    "NC": "[NX2]=[CX3]",
                    "C1OC1": "[OX2]1[CX3][CX3]1",
                    "CCS": "CCS",
                    "C=C(C=O)": "C=C(C=O)"
                }
                
                group_scores = []
                for group in self.desired_groups:
                    if group in group_patterns:
                        pattern = group_patterns[group]
                        has_group = mol.HasSubstructMatch(Chem.MolFromSmarts(pattern))
                        group_scores.append(float(has_group))
                
                return tf.constant(sum(group_scores) / len(self.desired_groups))
            except:
                return tf.constant(0.0)
        
        group_scores = tf.map_fn(
            check_groups,
            y_pred,
            fn_output_signature=tf.float32
        )
        group_loss = 1.0 - tf.reduce_mean(group_scores)
        
        # Combine losses
        total_loss = (self.weights[0] * recon_loss + 
                     self.weights[1] * valid_loss + 
                     self.weights[2] * group_loss)
        
        return total_loss
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "desired_groups": self.desired_groups,
            "weights": self.weights
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

# @keras.saving.register_keras_serializable()
# class ValidRateMetric(tf.keras.metrics.Metric):
#     def __init__(self, name='valid_rate', **kwargs):
#         super(ValidRateMetric, self).__init__(name=name, **kwargs)
#         self.valid_count = self.add_weight(name='valid', initializer='zeros')
#         self.total_count = self.add_weight(name='total', initializer='zeros')
#         self.tokenizer = PreTrainedTokenizerFast.from_pretrained("code/vocab/smiles_tokenizer")
    
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         # Convert logits to token indices
#         pred_tokens = tf.argmax(y_pred, axis=-1)
        
#         def check_validity(tokens):
#             try:
#                 # Convert tokens to SMILES string
#                 smiles = self.tokenizer.decode(tokens.numpy(), skip_special_tokens=True)
#                 smiles = smiles.replace(" ", "")  # Remove any spaces
                
#                 # Check if SMILES is valid using RDKit
#                 mol = Chem.MolFromSmiles(smiles)
#                 return tf.constant(1.0) if mol is not None else tf.constant(0.0)
#             except:
#                 return tf.constant(0.0)
        
#         # Use tf.py_function to wrap the RDKit operations
#         valid_scores = tf.map_fn(
#             lambda x: tf.py_function(check_validity, [x], tf.float32),
#             pred_tokens,
#             fn_output_signature=tf.float32
#         )
        
#         batch_size = tf.cast(tf.shape(y_pred)[0], tf.float32)
#         self.valid_count.assign_add(tf.reduce_sum(valid_scores))
#         self.total_count.assign_add(batch_size)
    
#     def result(self):
#         return self.valid_count / self.total_count
    
#     def reset_state(self):
#         self.valid_count.assign(0.)
#         self.total_count.assign(0.)
    
#     def get_config(self):
#         base_config = super().get_config()
#         return base_config
group_patterns = {
                    "C=C": "C=C",
                    "NC": "[NX2]=[CX3]",
                    "C1OC1": "[OX2]1[CX3][CX3]1",
                    "CCS": "CCS",
                    "C=C(C=O)": "C=C(C=O)"
                }

@keras.saving.register_keras_serializable()
class DetailedSMILESMetrics(tf.keras.metrics.Metric):
    def __init__(self, name='detailed_metrics', **kwargs):
        super().__init__(name=name, **kwargs)
        # SMILES reconstruction metrics
        self.recon_correct = self.add_weight(name='recon_correct', initializer='zeros')
        self.total_tokens = self.add_weight(name='total_tokens', initializer='zeros')
        
        # Group metrics
        self.group_correct = self.add_weight(name='group_correct', initializer='zeros')
        self.total_groups = self.add_weight(name='total_groups', initializer='zeros')
        
        # Validity metrics
        self.valid_count = self.add_weight(name='valid_count', initializer='zeros')
        self.total_sequences = self.add_weight(name='total_sequences', initializer='zeros')
        
        # Loss components
        self.recon_loss = self.add_weight(name='recon_loss', initializer='zeros')
        self.group_loss = self.add_weight(name='group_loss', initializer='zeros')
        self.valid_loss = self.add_weight(name='valid_loss', initializer='zeros')
        self.total_batches = self.add_weight(name='total_batches', initializer='zeros')
        
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained("code/vocab/smiles_tokenizer")
    
    def generate_full_smiles(self, logits):
        """Generate complete SMILES from logits"""
        # Get most likely tokens at each position
        pred_tokens = tf.argmax(logits, axis=-1)
        
        # Convert to SMILES strings
        def tokens_to_smiles(tokens):
            try:
                smiles = self.tokenizer.decode(tokens.numpy(), skip_special_tokens=True)
                return smiles.replace(" ", "")
            except:
                return ""
        
        # Map over batch dimension
        smiles_list = tf.map_fn(
            lambda x: tf.py_function(tokens_to_smiles, [x], tf.string),
            pred_tokens,
            fn_output_signature=tf.string
        )
        
        return smiles_list, pred_tokens
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # First generate complete SMILES
        generated_smiles, pred_tokens = self.generate_full_smiles(y_pred)
        true_tokens = tf.argmax(y_true, axis=-1)
        
        # 1. Reconstruction metrics and loss
        correct_tokens = tf.cast(tf.equal(pred_tokens, true_tokens), tf.float32)
        self.recon_correct.assign_add(tf.reduce_sum(correct_tokens))
        self.total_tokens.assign_add(tf.cast(tf.size(true_tokens), tf.float32))
        recon_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        self.recon_loss.assign_add(tf.reduce_mean(recon_loss))
        
        # 2. Validity check for complete SMILES
        def check_validity(smiles):
            try:
                mol = Chem.MolFromSmiles(smiles.numpy().decode())
                return tf.constant(1.0) if mol is not None else tf.constant(0.0)
            except:
                return tf.constant(0.0)
        
        valid_scores = tf.map_fn(
            lambda x: tf.py_function(check_validity, [x], tf.float32),
            generated_smiles,
            fn_output_signature=tf.float32
        )
        self.valid_count.assign_add(tf.reduce_sum(valid_scores))
        self.total_sequences.assign_add(tf.cast(tf.shape(generated_smiles)[0], tf.float32))
        valid_loss = 1.0 - tf.reduce_mean(valid_scores)
        self.valid_loss.assign_add(valid_loss)
        
        # 3. Group check for complete SMILES
        def check_groups(smiles):
            try:
                mol = Chem.MolFromSmiles(smiles.numpy().decode())
                if mol is None:
                    return tf.constant(0.0)
                
                group_patterns = {
                    "C=C": "C=C",
                    "NC": "[NX2]=[CX3]",
                    "C1OC1": "[OX2]1[CX3][CX3]1",
                    "CCS": "CCS",
                    "C=C(C=O)": "C=C(C=O)"
                }
                
                correct_groups = 0
                for pattern in group_patterns.values():
                    if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                        correct_groups += 1
                return tf.constant(float(correct_groups))
            except:
                return tf.constant(0.0)
        
        group_scores = tf.map_fn(
            lambda x: tf.py_function(check_groups, [x], tf.float32),
            generated_smiles,
            fn_output_signature=tf.float32
        )
        self.group_correct.assign_add(tf.reduce_sum(group_scores))
        self.total_groups.assign_add(tf.cast(tf.shape(generated_smiles)[0] * len(group_patterns), tf.float32))
        group_loss = 1.0 - tf.reduce_mean(group_scores)
        self.group_loss.assign_add(group_loss)
        
        # Update batch counter
        self.total_batches.assign_add(1.0)
    
    def result(self):
        return {
            # Accuracies
            'recon_accuracy': self.recon_correct / self.total_tokens,
            'valid_rate': self.valid_count / self.total_sequences,
            'group_accuracy': self.group_correct / self.total_groups,
            # Losses
            'recon_loss': self.recon_loss / self.total_batches,
            'valid_loss': self.valid_loss / self.total_batches,
            'group_loss': self.group_loss / self.total_batches
        }
    
    def reset_state(self):
        self.recon_correct.assign(0.)
        self.total_tokens.assign(0.)
        self.group_correct.assign(0.)
        self.total_groups.assign(0.)
        self.valid_count.assign(0.)
        self.total_sequences.assign(0.)
        self.recon_loss.assign(0.)
        self.group_loss.assign(0.)
        self.valid_loss.assign(0.)
        self.total_batches.assign(0.)
    
    def get_config(self):
        config = super().get_config()
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

# @keras.saving.register_keras_serializable()
# class SMILESMetrics(tf.keras.metrics.Metric):
#     def __init__(self, name='smiles_metrics', **kwargs):
#         super().__init__(name=name, **kwargs)
#         # SMILES reconstruction accuracy
#         self.smiles_correct = self.add_weight(name='smiles_correct', initializer='zeros')
#         self.total_tokens = self.add_weight(name='total_tokens', initializer='zeros')
        
#         # Group accuracy
#         self.group_correct = self.add_weight(name='group_correct', initializer='zeros')
#         self.total_groups = self.add_weight(name='total_groups', initializer='zeros')
        
#         # Validity
#         self.valid_count = self.add_weight(name='valid_count', initializer='zeros')
#         self.total_sequences = self.add_weight(name='total_sequences', initializer='zeros')
        
#         self.tokenizer = PreTrainedTokenizerFast.from_pretrained("code/vocab/smiles_tokenizer")
    
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         # 1. SMILES Reconstruction Accuracy
#         pred_tokens = tf.argmax(y_pred, axis=-1)
#         true_tokens = tf.argmax(y_true, axis=-1)
#         correct_tokens = tf.cast(tf.equal(pred_tokens, true_tokens), tf.float32)
#         self.smiles_correct.assign_add(tf.reduce_sum(correct_tokens))
#         self.total_tokens.assign_add(tf.cast(tf.size(true_tokens), tf.float32))
        
#         # 2. Validity Check
#         def check_validity(tokens):
#             try:
#                 smiles = self.tokenizer.decode(tokens.numpy(), skip_special_tokens=True)
#                 mol = Chem.MolFromSmiles(smiles)
#                 return tf.constant(1.0) if mol is not None else tf.constant(0.0)
#             except:
#                 return tf.constant(0.0)
        
#         valid_scores = tf.map_fn(
#             lambda x: tf.py_function(check_validity, [x], tf.float32),
#             pred_tokens,
#             fn_output_signature=tf.float32
#         )
#         self.valid_count.assign_add(tf.reduce_sum(valid_scores))
#         self.total_sequences.assign_add(tf.cast(tf.shape(pred_tokens)[0], tf.float32))
        
#         # 3. Group Accuracy
#         def check_groups(tokens):
#             try:
#                 smiles = self.tokenizer.decode(tokens.numpy(), skip_special_tokens=True)
#                 mol = Chem.MolFromSmiles(smiles)
#                 if mol is None:
#                     return tf.constant(0.0)
                
                
                
#                 correct_groups = 0
#                 for pattern in group_patterns.values():
#                     if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
#                         correct_groups += 1
#                 return tf.constant(float(correct_groups))
#             except:
#                 return tf.constant(0.0)
        
#         group_scores = tf.map_fn(
#             lambda x: tf.py_function(check_groups, [x], tf.float32),
#             pred_tokens,
#             fn_output_signature=tf.float32
#         )
#         self.group_correct.assign_add(tf.reduce_sum(group_scores))
#         self.total_groups.assign_add(tf.cast(tf.shape(pred_tokens)[0] * len(group_patterns), tf.float32))
    
#     def result(self):
#         # Return dictionary of metrics
#         return {
#             'smiles_accuracy': self.smiles_correct / self.total_tokens,
#             'validity_rate': self.valid_count / self.total_sequences,
#             'group_accuracy': self.group_correct / self.total_groups
#         }
    
#     def reset_state(self):
#         # Reset each weight individually
#         self.smiles_correct.assign(0.)
#         self.total_tokens.assign(0.)
#         self.group_correct.assign(0.)
#         self.total_groups.assign(0.)
#         self.valid_count.assign(0.)
#         self.total_sequences.assign(0.)
class PretrainedDecoder(tf.keras.layers.Layer):
        def __init__(self, max_length,pretrained_decoder_gru,pretrained_decoder_dense, **kwargs):
            super().__init__(**kwargs)
            self.pretrained_decoder_gru = pretrained_decoder_gru
            self.pretrained_decoder_dense = pretrained_decoder_dense
            self.max_length = max_length
            self.context_projection = tf.keras.layers.Dense(128)
            self.attention = tf.keras.layers.MultiHeadAttention(
                num_heads=8, 
                key_dim=32
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
        self.pad_layer = tf.keras.layers.Dense(136)

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
     
     
def create_group_relationship_model(pretrained_model, max_length, vocab_size, desired_groups):
    """
    Create model with explicit group relationship handling
    """
    # Input layers - model expects 3 inputs in specific order
    inputs = [
        tf.keras.layers.Input(shape=(max_length,), name='monomer1_input'),
        tf.keras.layers.Input(shape=(max_length,), name='monomer2_input'),
        tf.keras.layers.Input(shape=(len(Constants.GROUP_VOCAB),), name='group_input')
    ]
    
    monomer1_input, monomer2_input, group_input = inputs
    
    # Extract pretrained features
    pretrained_embedding = pretrained_model.get_layer('embedding')
    # pretrained_encoder = [layer for layer in pretrained_model.layers 
    #                      if 'encoder' in layer.name]
    pretrained_encoder = pretrained_model.get_layer('gru')
    print(pretrained_encoder)
    pretrained_decoder_gru = pretrained_model.get_layer('gru_1')
    pretrained_decoder_dense = pretrained_model.get_layer('dense')
    
    # Freeze pretrained layers
    pretrained_embedding.trainable = False
    # for layer in pretrained_encoder:
    #     layer.trainable = False
    pretrained_decoder_gru.trainable = False
    pretrained_decoder_dense.trainable = False
    pretrained_encoder.trainable = False
    
    # Apply pretrained embedding
    monomer1_emb = pretrained_embedding(monomer1_input)
    monomer2_emb = pretrained_embedding(monomer2_input)
    
    # Get chemical features from pretrained model
    def get_chemical_features(x):
        # for encoder_layer in pretrained_encoder:
        #     x = encoder_layer(x)
        return PretrainedEncoder(max_length=max_length,pretrained_encoder=pretrained_encoder)(x)
    
    monomer1_features = get_chemical_features(monomer1_emb)
    monomer2_features = get_chemical_features(monomer2_emb)
    
    # Create group-aware features
    def create_group_aware_features(features, group_vec):
        """Combine features with group information"""
        group_aware_layer = GroupAwareLayer()
        return group_aware_layer([features, group_vec])
    
    # Apply group-aware features
    group_aware1 = create_group_aware_features(monomer1_features, group_input)
    group_aware2 = create_group_aware_features(monomer2_features, group_input)
    
    # Relationship modeling
    def create_relationship_block():
        return tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(0.1)
        ])
    
    relationship_block = create_relationship_block()
    
    # Process pairs with attention to their relationship
    combined_features = tf.keras.layers.Concatenate()(
        [group_aware1, group_aware2]
    )
    relationship_output = relationship_block(combined_features)
    
    
    
    # Create decoders using pretrained components
    # In create_group_relationship_model function:
    decoder1 = PretrainedDecoder(max_length=max_length,
                                 pretrained_decoder_gru=pretrained_decoder_gru,
                                 pretrained_decoder_dense=pretrained_decoder_dense, name='decoder1')(
        [monomer1_features, group_aware1, relationship_output]
    )
    decoder2 = PretrainedDecoder(max_length=max_length,
                                 pretrained_decoder_gru=pretrained_decoder_gru,
                                 pretrained_decoder_dense=pretrained_decoder_dense, name='decoder2')(
        [monomer2_features, group_aware2, relationship_output]
    )
    
    # Create model
    model = tf.keras.Model(
        inputs=inputs,
        outputs=[decoder1, decoder2]
    )
    
    # Compile with loss and metrics
    model.compile(
        optimizer='adam',
        loss={
            'decoder1': CombinedLoss(desired_groups=desired_groups),
            'decoder2': CombinedLoss(desired_groups=desired_groups)
        },
        metrics={
            'decoder1': [DetailedSMILESMetrics(name='monomer1_metrics')],
            'decoder2': [DetailedSMILESMetrics(name='monomer2_metrics')]
        }
    )
    
    return model

# def create_group_relationship_model(pretrained_model, max_length, vocab_size, desired_groups):
#     """
#     Create model with explicit group relationship handling
#     """
#     # Input layers - model expects 3 inputs in specific order
#     inputs = [
#         tf.keras.layers.Input(shape=(max_length,), name='monomer1_input'),
#         tf.keras.layers.Input(shape=(max_length,), name='monomer2_input'),
#         tf.keras.layers.Input(shape=(len(Constants.GROUP_VOCAB),), name='group_input')
#     ]
    
#     monomer1_input, monomer2_input, group_input = inputs
    
#     # Extract pretrained features
#     pretrained_embedding = pretrained_model.get_layer('embedding')
#     pretrained_encoder = [layer for layer in pretrained_model.layers 
#                          if 'encoder' in layer.name and 'attention' in layer.name]
    
#     # Freeze pretrained layers
#     pretrained_embedding.trainable = False
#     for layer in pretrained_encoder:
#         layer.trainable = False
    
#     # Apply pretrained embedding
#     monomer1_emb = pretrained_embedding(monomer1_input)
#     monomer2_emb = pretrained_embedding(monomer2_input)
    
#     # Get chemical features from pretrained model
#     def get_chemical_features(x):
#         for encoder_layer in pretrained_encoder:
#             x = encoder_layer(x)
#         return x
    
#     monomer1_features = get_chemical_features(monomer1_emb)
#     monomer2_features = get_chemical_features(monomer2_emb)
    
#     # Create group-aware features
#     def create_group_aware_features(features, group_vec):
#         """Combine features with group information"""
        
       
    
#         # Create and apply the layer
#         group_aware_layer = GroupAwareLayer()
#         return group_aware_layer([features, group_vec])
    
#     # Apply group-aware features
#     group_aware1 = create_group_aware_features(monomer1_features, group_input)
#     group_aware2 = create_group_aware_features(monomer2_features, group_input)
    
#     # Relationship modeling
#     def create_relationship_block():
#         return tf.keras.Sequential([
#             tf.keras.layers.Dense(256, activation='relu'),
#             tf.keras.layers.LayerNormalization(),
#             tf.keras.layers.Dropout(0.1)
#         ])
    
#     relationship_block = create_relationship_block()
    
#     # Process pairs with attention to their relationship
#     combined_features = tf.keras.layers.Concatenate()(
#         [group_aware1, group_aware2]
#     )
#     relationship_output = relationship_block(combined_features)
    
#     # Final prediction layers
#     def create_monomer_decoder(features, name):
#         """Create decoder for monomer generation"""
#         def create_decoder_block(relationship_output, group_aware_features):
#             # Use Keras Concatenate layer instead of tf.concat
#             combined = tf.keras.layers.Concatenate(axis=-1)([relationship_output, group_aware_features])
#             x = tf.keras.layers.Dense(512, activation='relu')(combined)
#             x = tf.keras.layers.Dropout(0.2)(x)
#             return tf.keras.layers.Dense(vocab_size, activation='softmax', name=name)(x)
        
#         return create_decoder_block(*features)
    
#     # Output layers considering relationship
#     output1 = create_monomer_decoder(
#         [relationship_output, group_aware1],
#         'monomer1_output'
#     )
#     output2 = create_monomer_decoder(
#         [relationship_output, group_aware2],
#         'monomer2_output'
#     )
    
#     # Create model with explicit input order
#     model = tf.keras.Model(
#         inputs=inputs,  # Pass as list
#         outputs=[output1, output2]
#     )
    
#     # Create combined loss with weights and desired groups
#     loss_fn = CombinedLoss(desired_groups=desired_groups,weights=[1.0, 0.5, 0.5])
    
#     model.compile(
#         optimizer='adam',
#         loss={
#             'monomer1_output': loss_fn,
#             'monomer2_output': loss_fn
#         },
#         metrics={
#             'monomer1_output': [ValidRateMetric()],
#             'monomer2_output': [ValidRateMetric()]
#         }
#     )
    
#     return model

# Training with relationship awareness


def process_dual_monomer_data(excel_path, smiles_col='SMILES'):

    try:
        # Read Excel file
        df = pd.read_excel(excel_path)
        
        # Verify column exists
        if smiles_col not in df.columns:
            raise ValueError(f"Required column {smiles_col} not found in Excel file")
        
        # Extract SMILES pairs and remove any NaN values
        smiles_pairs = df[smiles_col].dropna().tolist()
        
        # Initialize lists for valid monomers
        valid_monomer1 = []
        valid_monomer2 = []
        
        # Process each SMILES pair
        for pair in smiles_pairs:
            try:
                # Split the SMILES string into two monomers
                split_pair = pair.split(',')
                if len(split_pair) >= 2:
                    m1, m2 = split_pair[0], split_pair[1]
                    m1, m2 = m1.strip(), m2.strip()
                else:
                    print(f"Skipping malformed pair: {pair} (missing comma or wrong format)")
                    continue
                
                # Verify both SMILES are valid
                if Chem.MolFromSmiles(str(m1)) and Chem.MolFromSmiles(str(m2)):
                    valid_monomer1.append(str(m1))
                    valid_monomer2.append(str(m2))
                else:
                    print(f"Skipping invalid SMILES pair: {m1}, {m2}")
            except ValueError:
                print(f"Skipping malformed pair: {pair} (missing comma or wrong format)")
                continue
       
        return valid_monomer1, valid_monomer2
    
    except Exception as e:
        print(f"Error processing Excel file: {str(e)}")
        raise
def count_functional_groups(smiles, smarts_pattern):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return len(mol.GetSubstructMatches(Chem.MolFromSmarts(smarts_pattern)))
def encode_functional_groups(monomer1_list, monomer2_list):
    # SMARTS patterns for different functional groups
    epoxy_smarts = "[OX2]1[CX3][CX3]1"    # Epoxy group
    imine_smarts = "[NX2]=[CX3]"          # Imine group
    vinyl_smarts = "C=C"                  # Vinyl group
    thiol_smarts = "CCS"                  # Thiol group
    acryl_smarts = "C=C(C=O)"             # Acrylic group
    
    all_groups = []
    
    for m1, m2 in zip(monomer1_list, monomer2_list):
        found_groups_m1 = []
        found_groups_m2 = []
        
        # Check for each group in monomer 1
        if count_functional_groups(m1, epoxy_smarts) >= 2:
            found_groups_m1.append("C1OC1")
        if count_functional_groups(m1, imine_smarts) >= 2:
            found_groups_m1.append("NC")
        if count_functional_groups(m1, vinyl_smarts) >= 2:
            found_groups_m1.append("C=C")
        if count_functional_groups(m1, thiol_smarts) >= 2:
            found_groups_m1.append("CCS")
        if count_functional_groups(m1, acryl_smarts) >= 2:
            found_groups_m1.append("C=C(C=O)")
        
        # Check for each group in monomer 2
        if count_functional_groups(m2, epoxy_smarts) >= 2:
            found_groups_m2.append("C1OC1")
        if count_functional_groups(m2, imine_smarts) >= 2:
            found_groups_m2.append("NC")
        if count_functional_groups(m2, vinyl_smarts) >= 2:
            found_groups_m2.append("C=C")
        if count_functional_groups(m2, thiol_smarts) >= 2:
            found_groups_m2.append("CCS")
        if count_functional_groups(m2, acryl_smarts) >= 2:
            found_groups_m2.append("C=C(C=O)")
        
        # Combine groups from both monomers
        combined_groups = found_groups_m1 + found_groups_m2
        if not combined_groups:
            combined_groups.append('No group')
        
        all_groups.append(combined_groups)
    
    # Encode groups using the vocabulary
    encoded_groups = [encode_groups(groups, Constants.GROUP_VOCAB) for groups in all_groups]
    
    return encoded_groups

def prepare_training_data(max_length, vocab):
    monomer1_list, monomer2_list = process_dual_monomer_data('Small_Data/smiles.xlsx')
    group_features = encode_functional_groups(monomer1_list, monomer2_list)
    tokens1 = tokenize_smiles(monomer1_list)
    tokens2 = tokenize_smiles(monomer2_list)
    
    # Add 1 to max_length to match model's expected shape
    padded_tokens1 = pad_token(tokens1, max_length + 1, vocab)
    padded_tokens2 = pad_token(tokens2, max_length + 1, vocab)
    
    # Convert to numpy arrays
    padded_tokens1 = np.array(padded_tokens1)
    padded_tokens2 = np.array(padded_tokens2)
    group_features = np.array(group_features)
    
    # Ensure group_features has the correct shape (batch_size, num_groups)
    if len(group_features.shape) > 2:
        group_features = group_features.reshape(group_features.shape[0], -1)
    
    # Print shapes for debugging
    print("Input shapes:")
    print(f"monomer1_input shape: {padded_tokens1[:, :-1].shape}")
    print(f"monomer2_input shape: {padded_tokens2[:, :-1].shape}")
    print(f"group_input shape: {group_features.shape}")
    
    # Create target data (shifted by one position)
    target1 = tf.keras.utils.to_categorical(padded_tokens1[:, 1:], num_classes=len(vocab))
    target2 = tf.keras.utils.to_categorical(padded_tokens2[:, 1:], num_classes=len(vocab))
    
    print("Target shapes:")
    print(f"target1 shape: {target1.shape}")
    print(f"target2 shape: {target2.shape}")
    
    # Return properly formatted dictionaries
    inputs = {
        'monomer1_input': padded_tokens1[:, :-1],
        'monomer2_input': padded_tokens2[:, :-1],
        'group_input': group_features
    }
    
    outputs = {
        'monomer1_output': target1,
        'monomer2_output': target2
    }
    
    return inputs, outputs

class ValidRateCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"\nEpoch {epoch + 1} Valid Rates:")
        print(f"Decoder1 Valid Rate: {logs.get('decoder1_valid_rate', 0.0):.4f}")
        print(f"Decoder2 Valid Rate: {logs.get('decoder2_valid_rate', 0.0):.4f}")

def train_with_relationships(model, train_data, epochs=1):
    X, y = train_data
    outputs = {
        'decoder1': y['monomer1_output'],
        'decoder2': y['monomer2_output']
    }
    
    history = model.fit(
        [X['monomer1_input'], X['monomer2_input'], X['group_input']],
        outputs,
        epochs=epochs,
        batch_size=32,
        validation_split=0.2,
        callbacks=[ValidRateCallback()]
    )
    return history

def validate_predictions(model, vocab, max_length, num_samples=5):
    """
    Validate model predictions by comparing input and generated SMILES
    """
    # Get validation data
    monomer1_list, monomer2_list = process_dual_monomer_data('Small_Data/smiles.xlsx')
    group_features = encode_functional_groups(monomer1_list, monomer2_list)
    
    # Ensure group_features has correct shape (batch_size, num_groups)
    group_features = np.array(group_features)
    if len(group_features.shape) > 2:
        group_features = group_features.reshape(group_features.shape[0], -1)
    
    # Prepare input data
    tokens1 = tokenize_smiles(monomer1_list)
    tokens2 = tokenize_smiles(monomer2_list)
    padded_tokens1 = pad_token(tokens1, max_length + 1, vocab)
    padded_tokens2 = pad_token(tokens2, max_length + 1, vocab)
    
    # Convert to numpy arrays and ensure all have same first dimension
    padded_tokens1 = np.array(padded_tokens1)[:, :-1]
    padded_tokens2 = np.array(padded_tokens2)[:, :-1]
    
    # Print shapes for debugging
    print("Validation data shapes:")
    print(f"monomer1 shape: {padded_tokens1.shape}")
    print(f"monomer2 shape: {padded_tokens2.shape}")
    print(f"group shape: {group_features.shape}")
    
    # Get predictions
    predictions = model.predict([
        padded_tokens1,
        padded_tokens2,
        group_features
    ])
    
    # Convert indices to SMILES
    idx_to_token = {idx: token for token, idx in vocab.items()}
    

    def decode_smiles(pred_sequence):
        pred_indices = np.argmax(pred_sequence, axis=-1)
        tokenizer = PreTrainedTokenizerFast.from_pretrained("code/vocab/smiles_tokenizer")
        decoded = tokenizer.decode(pred_indices, skip_special_tokens=True).replace(" ","")
        smiles = ''.join(decoded)
        return smiles
    
    def is_valid_smiles(smiles):
        """Check if SMILES string is valid"""
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None

    def calculate_similarity(smi1, smi2):
        """Calculate Tanimoto similarity between two SMILES strings"""
        # Only calculate if both SMILES are valid
        if not (is_valid_smiles(smi1) and is_valid_smiles(smi2)):
            return None
            
        try:
            mol1 = Chem.MolFromSmiles(smi1)
            mol2 = Chem.MolFromSmiles(smi2)
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2)
            return DataStructs.TanimotoSimilarity(fp1, fp2)
        except:
            return None

    # Print comparisons
    print("\nValidation Results:")
    print("=" * 80)
    
    for i in range(min(num_samples, len(monomer1_list))):
        print(f"\nSample {i+1}:")
        print("-" * 40)
        
        # Monomer 1
        input_smiles1 = monomer1_list[i]
        pred_smiles1 = decode_smiles(predictions[0][i])
        print("Monomer 1:")
        print(f"Input SMILES:     {input_smiles1}")
        print(f"Predicted SMILES: {pred_smiles1}")
        valid1_input = is_valid_smiles(input_smiles1)
        valid1_pred = is_valid_smiles(pred_smiles1)
        print(f"Input Valid: {valid1_input}, Prediction Valid: {valid1_pred}")
        
        # Monomer 2
        input_smiles2 = monomer2_list[i]
        pred_smiles2 = decode_smiles(predictions[1][i])
        print("\nMonomer 2:")
        print(f"Input SMILES:     {input_smiles2}")
        print(f"Predicted SMILES: {pred_smiles2}")
        valid2_input = is_valid_smiles(input_smiles2)
        valid2_pred = is_valid_smiles(pred_smiles2)
        print(f"Input Valid: {valid2_input}, Prediction Valid: {valid2_pred}")
        
        # Calculate similarity only if both SMILES are valid
        print("\nSimilarity Scores:")
        if valid1_input and valid1_pred:
            sim1 = calculate_similarity(input_smiles1, pred_smiles1)
            print(f"Monomer 1 Similarity: {sim1:.3f}")
        else:
            print("Monomer 1 Similarity: Not calculated (invalid SMILES)")
            
        if valid2_input and valid2_pred:
            sim2 = calculate_similarity(input_smiles2, pred_smiles2)
            print(f"Monomer 2 Similarity: {sim2:.3f}")
        else:
            print("Monomer 2 Similarity: Not calculated (invalid SMILES)")
            
        print("=" * 80)



def save_model(model, model_params, save_dir="saved_models_new_two_monomer"):
    """Save the model weights and parameters"""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save weights with correct extension
    weights_path = os.path.join(save_dir, f"weights_{timestamp}.weights.h5")
    model.save_weights(weights_path)
    
    # Save parameters
    params_path = os.path.join(save_dir, f"params_{timestamp}.json")
    with open(params_path, 'w') as f:
        json.dump(model_params, f)
    
    return weights_path, params_path

def load_model(weights_path, params_path, pretrained_model):
    """Load saved weights into a new model instance"""
    try:
        # Load parameters
        with open(params_path, 'r') as f:
            params = json.load(f)
        
        # Create new model instance
        new_model = create_group_relationship_model(
            pretrained_model=pretrained_model,
            max_length=params['max_length'],
            vocab_size=params['vocab_size'],
            desired_groups=params['desired_groups']
        )
        
        # Load weights
        new_model.load_weights(weights_path)
        
        print(f"Weights loaded from: {weights_path}")
        print(f"Parameters loaded from: {params_path}")
        
        return new_model, params
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None

def get_latest_model(save_dir="saved_models"):
    """Find the most recently saved model and its parameters"""
    import os
    import glob
    
    # Find all model files and parameter files
    model_files = glob.glob(os.path.join(save_dir, "model_*.keras"))
    param_files = glob.glob(os.path.join(save_dir, "params_*.json"))
    
    if not model_files:
        print("No saved models found")
        return None, None
    
    # Get the most recent model
    latest_model = max(model_files, key=os.path.getctime)
    
    # Find matching parameters file
    model_timestamp = latest_model.split('_')[-1].replace('.keras', '')
    matching_params = [p for p in param_files if model_timestamp in p]
    latest_params = matching_params[0] if matching_params else None
    
    return latest_model, latest_params

def create_weighted_losses():
    """Create weighted loss functions with RDKit MOL validation and group checking"""
    
    def reconstruction_loss(y_true, y_pred):
        """Standard reconstruction loss"""
        return tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    
    def rdkit_validity_loss(y_pred):
        """Check if predicted SMILES can create valid MOL file"""
        def check_rdkit_valid(pred_logits):
            pred_tokens = tf.argmax(pred_logits, axis=-1)
            tokenizer = PreTrainedTokenizerFast.from_pretrained("code/vocab/smiles_tokenizer")
            try:
                smiles = tokenizer.decode(pred_tokens.numpy(), skip_special_tokens=True)
                mol = Chem.MolFromSmiles(smiles)
                return tf.constant(1.0) if mol is not None else tf.constant(0.0)
            except:
                return tf.constant(0.0)
            
        validity_scores = tf.map_fn(
            check_rdkit_valid,
            y_pred,
            fn_output_signature=tf.float32
        )
        
        return 1.0 - tf.reduce_mean(validity_scores)
    
    def group_presence_loss(y_pred, desired_groups):
        """Check for presence of desired functional groups using RDKit"""
        def check_groups(pred_logits):
            pred_tokens = tf.argmax(pred_logits, axis=-1)
            tokenizer = PreTrainedTokenizerFast.from_pretrained("code/vocab/smiles_tokenizer")
            try:
                smiles = tokenizer.decode(pred_tokens.numpy(), skip_special_tokens=True)
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return tf.constant(0.0)
                
                # Check for each desired group
                group_patterns = {
                    "C=C": "C=C",              # Vinyl group
                    "NC": "[NX2]=[CX3]",       # Imine group
                    "C1OC1": "[OX2]1[CX3][CX3]1",  # Epoxy group
                    "CCS": "CCS",              # Thiol group
                    "C=C(C=O)": "C=C(C=O)"     # Acrylic group
                }
                
                group_count = 0
                for group in desired_groups:
                    if group in group_patterns:
                        pattern = group_patterns[group]
                        if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                            group_count += 1
                
                return tf.constant(sum(group_scores) / len(desired_groups))
            except:
                return tf.constant(0.0)
            
        group_scores = tf.map_fn(
            check_groups,
            y_pred,
            fn_output_signature=tf.float32
        )
        
        return 1.0 - tf.reduce_mean(group_scores)
    
    return reconstruction_loss, rdkit_validity_loss, group_presence_loss

def combined_loss(desired_groups, weights=[1.0, 0.5, 0.5]):
    """Combine all three losses with weights"""
    recon_loss, valid_loss, group_loss = create_weighted_losses()
    
    def loss_function(y_true, y_pred):
        # Calculate losses
        r_loss = recon_loss(y_true, y_pred)
        v_loss = valid_loss(y_pred)
        g_loss = group_loss(y_pred, desired_groups)
        
        # Print for monitoring
        tf.print("\nLosses:", {
            "reconstruction": r_loss,
            "rdkit_validity": v_loss,
            "group_presence": g_loss
        })
        
        return weights[0] * r_loss + weights[1] * v_loss + weights[2] * g_loss
    
    return loss_function

def generate_monomer_pair_with_temperature(model, input_smiles, desired_groups, vocab, max_length, temperature=1.0, num_samples=5):
    # Prepare input SMILES
    tokens = tokenize_smiles([input_smiles])
    padded_tokens = pad_token(tokens, max_length, vocab)
    input_seq = np.array(padded_tokens)  # Shape: (1, max_length)
    
    # Prepare group features
    group_features = encode_groups(desired_groups, Constants.GROUP_VOCAB)
    group_features = np.array([group_features])
    
    def sample_with_temperature(predictions, temperature):
        """Sample from predictions with temperature scaling"""
        if temperature == 0:
            return np.argmax(predictions)
        predictions = np.asarray(predictions).astype('float64')
        predictions = np.log(predictions + 1e-7) / temperature
        exp_preds = np.exp(predictions)
        predictions = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, predictions, 1)
        return np.argmax(probas)
    
    def generate_monomer(decoder_index):
        """Generate single monomer sequence"""
        # Initialize decoder sequence with proper shape
        decoder_seq = np.zeros((1, max_length))
        decoder_seq[0, 0] = vocab['<start>']
        
        generated_tokens = []
        for i in range(max_length-1):
            # Ensure all inputs have correct shapes
            output = model.predict([
                input_seq,  # Shape: (1, max_length)
                decoder_seq,  # Shape: (1, max_length)
                group_features  # Shape: (1, num_groups)
            ], verbose=0)
            
            next_token_probs = output[decoder_index][0, i]
            next_token = sample_with_temperature(next_token_probs, temperature)
            
            generated_tokens.append(next_token)
            if next_token == vocab['<end>']:
                break
            if i < max_length - 2:
                decoder_seq[0, i + 1] = next_token
        
        return generated_tokens
    
    def check_groups(smiles, desired_groups):
        """Check presence of desired groups"""
        if smiles == "":
            return [], desired_groups   
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [], desired_groups
        group_patterns = {
            "C=C": "C=C",              # Vinyl group
            "NC": "[NX2]=[CX3]",       # Imine group
            "C1OC1": "[OX2]1[CX3][CX3]1",  # Epoxy group
            "CCS": "CCS",              # Thiol group
        "C=C(C=O)": "C=C(C=O)"     # Acrylic group
    }
        
        present_groups = []
        not_present_groups = []
        for group in desired_groups:
            pattern = group_patterns.get(group)
            if pattern and mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                present_groups.append(group)
            else:
                not_present_groups.append(group)
        return present_groups, not_present_groups
    
    # Generate multiple pairs
    generated_pairs = []
    valid_pairs = []
    
    print(f"\nGenerating {num_samples} monomer pairs with temperature {temperature}:")
    print("=" * 80)
    print(f"Input SMILES: {input_smiles}")
    print(f"Desired Groups: {', '.join(desired_groups)}")
    print("-" * 80)
    
    for i in range(num_samples):
        # Generate monomers using each decoder
        tokens1 = generate_monomer(0)  # decoder1
        tokens2 = generate_monomer(1)  # decoder2
        
        # Convert to SMILES
        smiles1 = decode_smiles(tokens1)
        smiles2 = decode_smiles(tokens2)
        
        # Check validity
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        valid1 = mol1 is not None and smiles1 != ""
        valid2 = mol2 is not None and smiles2 != ""
        
        print(f"\nPair {i+1}:")
        print(f"Monomer 1: {smiles1}")
        print(f"Valid: {valid1}")
        if valid1:
            present1, missing1 = check_groups(smiles1, desired_groups)
            print(f"Present groups: {present1}")
            print(f"Missing groups: {missing1}")
        
        print(f"Monomer 2: {smiles2}")
        print(f"Valid: {valid2}")
        if valid2:
            present2, missing2 = check_groups(smiles2, desired_groups)
            print(f"Present groups: {present2}")
            print(f"Missing groups: {missing2}")
        
        generated_pairs.append((smiles1, smiles2))
        if valid1 and valid2:
            valid_pairs.append((smiles1, smiles2))
    
    return generated_pairs, valid_pairs
# Example usage:
if __name__ == "__main__":
    # Load pretrained model
    
    pretrained_model, smiles_vocab, model_params = load_and_retrain()
    
    # Prepare training data
    train_data = prepare_training_data(
        max_length=model_params['max_length'],
        vocab=smiles_vocab
    )
    
    # # Create new model
    desired_groups = ["C=C", "NC"]  # Example desired groups
    new_model = create_group_relationship_model(
        pretrained_model=pretrained_model,
        max_length=model_params['max_length'],
        vocab_size=len(smiles_vocab),
        desired_groups=desired_groups
    )
    
    # # Print model summary
    print("\nModel summary:")
    new_model.summary()
    
    # # Train model
    history = train_with_relationships(
        model=new_model,
        train_data=train_data,
        epochs=1
    )
    
    # # After training your model...
    # validate_predictions(
    #     model=new_model,
    #     vocab=smiles_vocab,
    #     max_length=model_params['max_length'],
    #     num_samples=1
    # )


    # Generate with different temperatures
    input_smiles = "CC=CC"
    desired_groups = ["C=C", "NC"]
    generated_data = []

        # Generate with different temperatures
    temperatures = [0.5]
    for temp in temperatures:
        generated, valid = generate_monomer_pair_with_temperature(
        model=new_model,
        input_smiles=input_smiles,
        desired_groups=desired_groups,
        vocab=smiles_vocab,
        max_length=model_params['max_length'],
        temperature=temp,
        num_samples=3
        )
    # Store only valid pairs
        generated_data.append((temp, valid))

    #save_generated_smiles(input_smiles, generated_data)
    
    print(f"\nValid pairs generated at temperature {temp}:")
    for pair in valid:
        print(f"Monomer 1: {pair[0]}")
        print(f"Monomer 2: {pair[1]}")
        print("-" * 40)
    
    # # Generate with different temperatures
    # temperatures = [0.5, 1.0, 1.5]
    # for temp in temperatures:
    #     generated, valid = generate_monomer_pair_with_temperature(
    #         model=new_model,
    #         input_smiles=input_smiles,
    #         desired_groups=desired_groups,
    #         vocab=smiles_vocab,
    #         max_length=model_params['max_length'],
    #         temperature=temp,
    #         num_samples=3
    #     )

    # Save model
    
    
    weights_path, params_path = save_model(
    model=new_model,
    model_params={
        'max_length': model_params['max_length'],
        'vocab_size': len(smiles_vocab),
        'desired_groups': desired_groups
    }
    )
    loaded_model, params = load_model(
        weights_path=weights_path,
        params_path=params_path,
        pretrained_model=pretrained_model  # Original pretrained model
    )
    

    
    # if loaded_model is not None:
    #     print("Model loaded successfully!")
    #     # Test with a simple prediction
    #     input_smiles = "CC=CC"
    #     desired_groups = ["C=C", "NC"]
        
    #     generated, valid = generate_monomer_pair_with_temperature(
    #         model=loaded_model,
    #         input_smiles=input_smiles,
    #         desired_groups=desired_groups,
    #         vocab=smiles_vocab,
    #         max_length=loaded_params['max_length'],
    #         temperature=1.0,
    #         num_samples=3
    #     )
