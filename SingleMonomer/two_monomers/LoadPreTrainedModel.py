import json
import tensorflow as tf
from Data_Process_with_prevocab import *
import random
import keras

def calculate_group_similarity(target_groups, generated_groups):
    """Calculate similarity between target and generated functional groups"""
    target_set = set(target_groups)
    generated_set = set(generated_groups)
    
    intersection = len(target_set.intersection(generated_set))
    union = len(target_set.union(generated_set))
    
    return intersection / max(union, 1)

@keras.saving.register_keras_serializable()
def custom_loss(y_true, y_pred, group_smarts_list, reverse_vocab):
    # Convert inputs to tensors and fix shapes
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)
    print(y_true.shape)
    print(y_pred.shape)
    
    if len(y_true.shape) == 3:
        y_true = tf.squeeze(y_true, axis=-1)
    
    y_true = tf.cast(y_true, tf.int32)
    
    # Calculate reconstruction loss
    reconstruction_loss = tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False)
    )

     # Helper function to decode predictions to SMILES
    def decode_predictions(pred_tensor):
        pred_indices = tf.argmax(pred_tensor, axis=-1)
        predicted_smiles=[]
        for tokens in pred_indices:
            smiles = decode_smiles(tokens)
            predicted_smiles.append(smiles)
       
        return predicted_smiles
    
    # Validity check function
    def check_validity(pred_tensor):
        smiles_list = decode_predictions(pred_tensor)
        validity_scores = []
        valid_smiles=[]
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None and smiles != "":
                    valid_smiles.append(smiles)
                    validity_scores.append(0.0)
                else:
                    validity_scores.append(1.0)
            except:
                validity_scores.append(1.0)
        return tf.constant(validity_scores, dtype=tf.float32), valid_smiles
    
    
    def check_groups(pred_tensor, group_smarts_list):
        validity_scores, smiles_list = check_validity(pred_tensor)
        group_scores = []
        if len(smiles_list) == 0:
            return tf.constant([1.0], dtype=tf.float32),validity_scores
        
        for smile in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smile)
                generated_groups = []
                for smarts in Constants.GROUP_VOCAB.keys():
                    if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts)):
                        generated_groups.append(smarts)
            
            # Calculate similarity score
                similarity = calculate_group_similarity(group_smarts_list, generated_groups)
                group_scores.append(1.0 - similarity)  # Convert to loss
            except:
                group_scores.append(1.0)
        return tf.constant(group_scores, dtype=tf.float32),validity_scores
    
    reconstruction_weight = 1.0
    validity_weight = 0.5
    group_weight = 0.5

    group_loss_score,validity_scores = check_groups(y_pred,Constants.GROUP_VOCAB.keys())

  
    validity_loss = tf.reduce_mean(validity_scores)
    group_loss = tf.reduce_mean(group_loss_score)
    
    print('\nReconstruction Loss: ',reconstruction_loss.numpy())
    print('Validity Loss: ',validity_loss.numpy())
    print('Group Loss: ',group_loss.numpy())
    total_loss = (reconstruction_weight * reconstruction_loss + 
                 validity_weight * validity_loss + 
                 group_weight * group_loss)
 
    
    # Return only reconstruction loss for now to ensure stable training
    return total_loss

@keras.saving.register_keras_serializable()
class CustomLoss(tf.keras.losses.Loss):
    def __init__(self, group_smarts_list, reverse_vocab, name='custom_loss', reduction='sum_over_batch_size'):
        super().__init__(name=name, reduction=reduction)
        self.group_smarts_list = group_smarts_list
        self.reverse_vocab = reverse_vocab
    
    def call(self, y_true, y_pred):
        return custom_loss(y_true, y_pred, self.group_smarts_list, self.reverse_vocab)
    
    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "group_smarts_list": list(self.group_smarts_list),  # Convert to list if it's not already
            "reverse_vocab": self.reverse_vocab
        }
    
    @classmethod
    def from_config(cls, config):
        return cls(**config) 
def load_and_retrain(save_dir="../saved_models_new"):
    # Load the vocabulary
    with open(f"{save_dir}/smiles_vocab.json", "r") as f:
        smiles_vocab = json.load(f)
    
    # Load model parameters
    with open(f"{save_dir}/model_params.json", "r") as f:
        model_params = json.load(f)
    
    # Load the model with updated path
    model = tf.keras.models.load_model(f"{save_dir}/model.keras")
    
    return model, smiles_vocab, model_params
