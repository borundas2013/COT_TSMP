import tensorflow as tf
from rdkit import Chem
from Data_Process_with_prevocab import *
from LoadPreTrainedModel import *
from datetime import datetime
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
print(tf.__version__)
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
# tf.config.run_functions_eagerly(True)
from Reward_Score import *
# import numpy as np

# @keras.saving.register_keras_serializable()
# class CombinedLoss(tf.keras.losses.Loss):
#     def __init__(self,decoder_name=None, **kwargs):
#         name = f"{decoder_name}_combined_loss" if decoder_name else "combined_loss"
#         super().__init__(name=name, **kwargs)
#         self.recon_weight = 1.0
#         self.valid_weight = 1.0
#         self.dl_weight = 0.8
#         self.reward_weight = 0.8
#         self.decoder_name = decoder_name
        
#     def call(self, y_true, y_pred):

#         recon_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False)
#         recon_loss = tf.reduce_mean(recon_loss)
#         recon_loss = tf.sigmoid(recon_loss)


#         def check_rdkit_valid(inputs):
#             y_pred, y_true = inputs
       
            
#             # Get predicted SMILES
#             pred_tokens = np.argmax(y_pred.numpy(), axis=-1)
#             true_tokens = np.argmax(y_true.numpy(), axis=-1)
            
           
            
#             try:
#                 # Decode both predicted and true SMILES
#                 pred_smiles = decode_smiles(pred_tokens)
#                 true_smiles = decode_smiles(true_tokens)

                
#                 # Get molecules and canonical SMILES
#                 pred_mol = Chem.MolFromSmiles(pred_smiles)
#                 true_mol = Chem.MolFromSmiles(true_smiles)
                
#                 # Save to file if valid
#                 if pred_mol is not None or true_mol is not None:
#                     valid_pairs = {
#                         "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
#                         "true": {
#                             "original_smiles": true_smiles,
#                             "canonical_smiles": Chem.MolToSmiles(true_mol, canonical=True) if true_mol else None,
#                             "valid": true_mol is not None
#                         },
#                         "predicted": {
#                             "original_smiles": pred_smiles,
#                             "canonical_smiles": Chem.MolToSmiles(pred_mol, canonical=True) if pred_mol else None,
#                             "valid": pred_mol is not None
#                         }
#                     }
                    
#                     # Append to JSON file
#                     output_file = Constants.VALID_PAIRS_FILE
#                     try:
#                         with open(output_file, 'r') as f:
#                             all_pairs = json.load(f)
#                     except (FileNotFoundError, json.JSONDecodeError):
#                         all_pairs = []
                    
#                     all_pairs.append(valid_pairs)
#                     with open(output_file, 'w') as f:
#                         json.dump(all_pairs, f, indent=4)
                
#                 return 1.0 if pred_mol is not None else 0.0
#             except:
#                 return 0.0
        
    

#         def calculate_tanimoto(smiles1, smiles2):
#             """Calculate Tanimoto similarity between two SMILES strings"""
#             try:
#                 mol1 = Chem.MolFromSmiles(smiles1)
#                 mol2 = Chem.MolFromSmiles(smiles2)
#                 if mol1 is None or mol2 is None:
#                     return 0.0
            
#                 fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
#                 fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
            
#                 return DataStructs.TanimotoSimilarity(fp1, fp2)
#             except:
#                 return 0.0
        
#         def diversity_loss(inputs):
#             y_pred, y_true = inputs
#             # Convert tensors to predicted tokens and true tokens first
#             pred_tokens = np.argmax(y_pred.numpy(), axis=-1)
#             true_tokens = np.argmax(y_true.numpy(), axis=-1)
#             try:
#                 # Decode tokens to SMILES
#                 pred_smiles = decode_smiles(pred_tokens)
#                 true_smiles = decode_smiles(true_tokens)
#                 mol1 = Chem.MolFromSmiles(pred_smiles)
#                 mol2 = Chem.MolFromSmiles(true_smiles)
#                 if mol1 is None or mol2 is None:
#                     return 0.0
                
#                 # Calculate Tanimoto similarity
#                 similarity = calculate_tanimoto(pred_smiles, true_smiles)
#                 return similarity
#             except:
#                 return 0.0


#         def rewarrd_calculate(inputs):
#             y_pred, y_true = inputs
#             pred_tokens = np.argmax(y_pred.numpy(), axis=-1)
#             true_tokens = np.argmax(y_true.numpy(), axis=-1)
#             pred_smiles = decode_smiles(pred_tokens)
#             true_smiles = decode_smiles(true_tokens)
#             chemical_groups = extract_group_smarts(true_smiles)
#             reward_score = get_reward_score(pred_smiles, chemical_groups)
#             return reward_score
        
#         valid_losses = []
#         dl_losses = []
#         rewards = []
#         for i in range(len(y_pred)):    
#             inputs = (y_pred[i], y_true[i])
#             valid_losses.append(check_rdkit_valid(inputs))
#             dl_losses.append(diversity_loss(inputs))
#             reward_score = rewarrd_calculate(inputs)
#             rewards.append(reward_score)
        
#         valid_loss =1-np.mean(valid_losses)
#         dl_loss = 1-np.mean(dl_losses)
#         reward = np.mean(rewards)
#         scaled_reward = reward * self.reward_weight

    
#         # Combine losses
#         total_loss = (self.recon_weight * recon_loss + 
#                      self.valid_weight * valid_loss + 
#                      self.dl_weight * dl_loss)
#         print("--------------------------------")
#         print("Recon Loss: ", recon_loss)
#         print("Valid Loss: ", valid_loss)
#         print("Diversity Loss: ", dl_loss)
#         print("Total Loss: ", total_loss)
#         print("Reward: ", reward)
#         print("Scaled Reward: ", scaled_reward)

#         final_loss = max(0.0,total_loss - scaled_reward)
  
#         print("Final Loss: ", final_loss)
#         return final_loss
    
#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "recon_loss": self.recon_loss,
#             "valid_loss": self.valid_loss,
#             "dl_loss": self.dl_loss,
#             'decoder_name': self.decoder_name
#         })
#         return config
    
#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)


import tensorflow as tf
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import json
from datetime import datetime
import numpy as np

@tf.keras.utils.register_keras_serializable()
class CombinedLoss(tf.keras.losses.Loss):
    def __init__(self, decoder_name=None, **kwargs):
        name = f"{decoder_name}_combined_loss" if decoder_name else "combined_loss"
        super().__init__(name=name, **kwargs)
        self.recon_weight = 1.0
        self.valid_weight = 1.0
        self.dl_weight = 0.8
        self.reward_weight = 0.8
        self.decoder_name = decoder_name

    def call(self, y_true, y_pred):
        #  Fix: Use TensorFlow loss function correctly
        print("y_true: ", y_true.shape)
        print("y_pred: ", y_pred.shape)

        #loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        #recon_loss = tf.reduce_mean(loss_fn(y_true, y_pred))
        recon_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False)
        recon_loss = tf.reduce_mean(recon_loss)

        def get_tokens_and_smiles(y_true, y_pred):
            pred_tokens = tf.argmax(y_pred, axis=-1)
            true_tokens = tf.argmax(y_true, axis=-1)#y_true#
            pred_smiles = decode_smiles(pred_tokens.numpy())
            true_smiles = decode_smiles(true_tokens.numpy().astype(int))
            return pred_tokens, true_tokens, pred_smiles, true_smiles

        #  Fix: Convert to TensorFlow tensors
        def check_valid(y_true, y_pred):
            """Check if predicted SMILES is valid using RDKit."""
            pred_tokens, true_tokens, pred_smiles, true_smiles = get_tokens_and_smiles(y_true, y_pred)
            print("--------------------------------")
            print("Pred Smiles: ", pred_smiles)
            print("True Smiles: ", true_smiles)
            print("--------------------------------")

            pred_mol = Chem.MolFromSmiles(pred_smiles)
            return tf.convert_to_tensor(1.0 if pred_mol is not None else 0.0, dtype=tf.float32)

        def calculate_tanimoto(smiles1, smiles2):
            """Compute Tanimoto similarity using RDKit fingerprints."""
            mol1 = Chem.MolFromSmiles(smiles1)
            mol2 = Chem.MolFromSmiles(smiles2)
            if mol1 is None or mol2 is None:
                return tf.convert_to_tensor(0.0, dtype=tf.float32)

            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)

            similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
            return tf.convert_to_tensor(similarity, dtype=tf.float32)

        def diversity_loss(y_true, y_pred):
            """Calculate diversity loss (1 - Tanimoto similarity)."""
            pred_tokens, true_tokens, pred_smiles, true_smiles = get_tokens_and_smiles(y_true, y_pred)

            similarity = calculate_tanimoto(pred_smiles, true_smiles)
            return tf.convert_to_tensor(1.0 - similarity, dtype=tf.float32)

        # def reward_calculate(y_true, y_pred):
        #     """Compute reward score based on chemical groups."""
        #     pred_tokens, true_tokens, pred_smiles, true_smiles = get_tokens_and_smiles(y_true, y_pred)

        #     # pred_smiles = decode_smiles(pred_tokens.numpy())
        #     # true_smiles = decode_smiles(true_tokens.numpy())

        #     chemical_groups = extract_group_smarts(true_smiles)
        #     reward_score = get_reward_score(pred_smiles, chemical_groups)

        #     return tf.convert_to_tensor(reward_score, dtype=tf.float32)

        #  Vectorized processing for batch-wise computation
        valid_losses = tf.map_fn(lambda x: check_valid(x[0], x[1]), (y_true, y_pred), dtype=tf.float32)
        dl_losses = tf.map_fn(lambda x: diversity_loss(x[0], x[1]), (y_true, y_pred), dtype=tf.float32)
        #rewards = tf.map_fn(lambda x: reward_calculate(x[0], x[1]), (y_true, y_pred), dtype=tf.float32)

        valid_loss = 1.0-tf.reduce_mean(valid_losses)
        dl_loss = 1.0-tf.reduce_mean(dl_losses)
        #reward = tf.reduce_mean(rewards)

        #scaled_reward = reward * self.reward_weight

        # Compute final loss
        total_loss = (
            self.recon_weight * recon_loss + 
            self.valid_weight * valid_loss + 
            self.dl_weight * dl_loss
        )

        final_loss = tf.maximum(0.0, total_loss)#tf.maximum(0.0, total_loss - scaled_reward)

        #  Print Debugging Info
        print("--------------------------------")
        print("Recon Loss:", recon_loss)
        print("Valid Loss:", valid_loss)
        print("Diversity Loss:", dl_loss)
        print("Total Loss:", total_loss)
        # print("Reward:", reward)
        # print("Scaled Reward:", scaled_reward)
        print("Final Loss:", final_loss)

        return final_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            "recon_weight": self.recon_weight,
            "valid_weight": self.valid_weight,
            "dl_weight": self.dl_weight,
            "reward_weight": self.reward_weight,
            "decoder_name": self.decoder_name
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    