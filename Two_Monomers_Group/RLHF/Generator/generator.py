# import os
# import sys

# # Get the absolute path of the current file's directory
# current_dir = os.path.dirname(os.path.abspath(__file__))

# # Add parent directory to Python path
# parent_dir = os.path.dirname(os.path.dirname(current_dir))
# sys.path.append(parent_dir)

# # Add RLHF directory to Python path
# rlhf_dir = os.path.dirname(current_dir)
# sys.path.append(rlhf_dir)

# # Add Generator directory to Python path 
# generator_dir = current_dir
# sys.path.append(generator_dir)

# import os
# import tensorflow as tf
# from Constants import *
# from Data_Process_with_prevocab import *
# from LoadPreTrainedModel import *
# from pretrained_weights import *
# from saveandload import *
# from dual_smile_process import *
# from validation_prediction import *
# from group_based_model import create_group_relationship_model
# from pathlib import Path
# from losswithReward import CombinedLoss
# from NewModelApp1 import *


# save_dir_abs = os.path.join("Two_Monomers_Group", "pretrained_model", "saved_models_rl_gpu_3")
# file_path = os.path.join('Two_Monomers_Group', 'Data', "smiles_orginal.xlsx")
# pretrained_model, smiles_vocab, model_params = load_and_retrain(save_dir=save_dir_abs)

# prediction_model = create_model(model_params['max_length'], len(smiles_vocab), pretrained_model)
# weights_path = os.path.join(root_dir, "RLHF","Generator","models","group_based_rl_model_n2", "weights_model.weights.h5")

# prediction_model.load_weights(weights_path)
# monomer1_list, monomer2_list = process_dual_monomer_data(file_path)
# group_combinations = [["C=C", "C=C(C=O)"], ["C=C", "CCS"], ["C1OC1", "NC"], ["C=C", "[OH]"]]


# def sample_with_temperature(predictions, temperature):
#         """Sample from predictions with temperature scaling"""
#         if temperature == 0:
#             return np.argmax(predictions)
#         predictions = np.asarray(predictions).astype('float64')
#         predictions = np.log(predictions + 1e-7) / temperature
#         exp_preds = np.exp(predictions)
#         predictions = exp_preds / np.sum(exp_preds)
#         probas = np.random.multinomial(1, predictions, 1)
#         return np.argmax(probas), predictions, probas

# def generate_monomer_pair_with_temperature(model, input_smiles, desired_groups, vocab, max_length, temperatures=[0.8], group_smarts1=None, group_smarts2=None, add_noise=False):
#     # Prepare input SMILES
#     tokens,tokens2  = tokenize_smiles([input_smiles[0]]),tokenize_smiles([input_smiles[1]])
#     padded_tokens = pad_token(tokens, max_length, vocab)
#     padded_tokens2 = pad_token(tokens2, max_length, vocab)
#     input_seq = np.array(padded_tokens)  # Shape: (1, max_length)
#     input_seq2 = np.array(padded_tokens2)  # Shape: (1, max_length)
    
#     # Prepare group features
#     group_features = encode_groups(desired_groups, Constants.GROUP_VOCAB)
#     group_features = np.array([group_features])
    
#     if add_noise:
#         noisy_tokens1 = add_swap_noise(input_seq, 0.1)
#         noisy_tokens2 = add_swap_noise(input_seq2, 0.1) 

#         noisy_tokens1 = add_gaussian_noise(noisy_tokens1, 0.1)
#         noisy_tokens2 = add_gaussian_noise(noisy_tokens2, 0.1)
#         input_seq =noisy_tokens1
#         input_seq2 =noisy_tokens2
    
   
    
#     def generate_monomer():
#         """Generate single monomer sequence"""
#         # Initialize decoder sequence with proper shape
        
#         decoder_seq = np.zeros((1, max_length))
#         decoder_seq[0, 0] = vocab['<start>']
#         decoder_seq2 = np.zeros((1, max_length))
#         decoder_seq2[0, 0] = vocab['<start>']
        

#         all_tokens = []
#         all_tokens_probs = []
#         for temperature in temperatures:
#             print(f"\nGenerating monomer pairs with temperature {temperature}:")
#             print("=" * 80)
#             print(f"Input SMILES: {input_smiles}")
#             print(f"Desired Groups: {', '.join(desired_groups)}")
#             print("-" * 80)
#             generated_tokens = []
#             generated_tokens2 = []
#             #it should be 245
#             for i in range(max_length):  
#                 output = model.predict({
#                     'monomer1_input': input_seq,
#                     'monomer2_input': input_seq2,
#                     'group_input': group_features,
#                     'decoder_input1': decoder_seq,
#                     'decoder_input2': decoder_seq2
#                     },verbose=0)
            
#                 next_token_probs = output[0][0, i]
#                 next_token, next_token_probs, next_token_probs_array = sample_with_temperature(next_token_probs, temperature)

#                 next_token_probs2 = output[1][0, i]
#                 next_token2, next_token_probs2, next_token_probs_array2 = sample_with_temperature(next_token_probs2, temperature)
            
#                 generated_tokens.append(next_token)
#                 generated_tokens2.append(next_token2)
#                 if next_token == vocab['<end>']:
#                     if next_token2 == vocab['<end>']:
#                         break
#                 if next_token2 == vocab['<end>']:
#                     if next_token == vocab['<end>']:
#                         break

#                 if i < max_length - 2:
#                     decoder_seq[0, i + 1] = next_token
#                     decoder_seq2[0, i + 1] = next_token
#             all_tokens.append([generated_tokens,generated_tokens2,temperature])
#             all_tokens_probs.append([next_token_probs_array,next_token_probs_array2])
#         return all_tokens, all_tokens_probs
    
#     tokens, tokens_probs = generate_monomer()
#     for i in range(len(tokens)):
#         smiles1= decode_smiles(tokens[i][0])
#         smiles2= decode_smiles(tokens[i][1])
#         print(smiles1, smiles2)
#     return tokens, tokens_probs


# for i in range(len(monomer1_list[:2])):
#     selected_groups = random.choice(group_combinations)
#     smiles1 = monomer1_list[i]
#     smiles2 = monomer2_list[i]
#     group1 = selected_groups[0]
#     group2 = selected_groups[1]
#     print(smiles1, smiles2, group1, group2)
#     tokens, tokens_probs = generate_monomer_pair_with_temperature(prediction_model, [smiles1, smiles2], [group1, group2], smiles_vocab, 245, temperatures=[0.2,0.4,0.8])

import os
import sys

# Get the absolute path of the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

# Add RLHF directory to Python path
rlhf_dir = os.path.dirname(current_dir)
sys.path.append(rlhf_dir)

# Add Generator directory to Python path 
generator_dir = current_dir
sys.path.append(generator_dir)
import tensorflow as tf
import numpy as np
from Data_Process_with_prevocab import *
from dual_smile_process import *
import random
from LoadPreTrainedModel import *
import os
from NewModelApp1 import *
import Two_Monomers_Group.RLHF.RLHFConstants as RLHFConstants
import Constants

class GeneratorModel(tf.keras.Model):
    def __init__(self, base_model, smiles_vocab, max_length=245):
        super(GeneratorModel, self).__init__()
        self.base_model = base_model
        self.vocab = smiles_vocab
        self.max_length = max_length
        self.prediction_model = create_model(max_length, len(smiles_vocab), base_model)
        self.group_combinations = [
            ["C=C", "C=C(C=O)"], 
            ["C=C", "CCS"], 
            ["C1OC1", "NC"], 
            ["C=C", "[OH]"],
            ["NC","C1OC1"],
            ["CCS","C=C"],
            ["[OH]","C=C"],
            ["C=C(C=O)","C=C"],

        ]

    def sample_with_temperature(self, predictions, temperature):
        """Sample from predictions with temperature scaling"""
        if temperature == 0:
            return np.argmax(predictions)
        predictions = np.asarray(predictions).astype('float64')
        predictions = np.log(predictions + 1e-7) / temperature
        exp_preds = np.exp(predictions)
        predictions = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, predictions, 1)
        return np.argmax(probas), predictions, probas
    
    def get_tokens_from_probabilities(self,probabilities, vocab, temperature=1.0):
        """Convert probability distributions to token indices"""
        # probabilities shape: [sequence_length, vocab_size]
        index_to_token = {idx: token for token, idx in vocab.items()}
       
        print(f"Probabilities shape: {probabilities.shape}")  # e.g., (5, 1000) for seq_len=5, vocab_size=1000
        
        if temperature == 0:
            # Use argmax for deterministic selection
            token_indices = tf.argmax(probabilities, axis=-1)  # shape: [sequence_length]
            print(f"Token indices shape: {token_indices.shape}")  # e.g., (5,)
        else:
            logits = tf.math.log(probabilities + 1e-10)
            scaled_logits = logits / temperature
            token_indices = tf.random.categorical(scaled_logits, num_samples=1)[:, 0]
        tokens=[]
        

        for i in range(len(token_indices)):
            index=token_indices[i]
            token=index_to_token.get(index.numpy())
            if token == '<pad>':
                break
            elif token == '[CLS]':
                continue
            elif token == '[SEP]':
                break
            elif token == '[MASK]':
                break
            elif token == '[UNK]':
                break
            elif token == '<start>':
                continue
            elif token == '<end>':
                break   
            tokens.append(index)
        # Convert indices to list of tokens
        
        return tokens
    
    def generate(self, input_data_batch,training=True):
        decoder1_input = input_data_batch['decoder_input1']
        decoder2_input = input_data_batch['decoder_input2']
        predictions = self.prediction_model({
            'monomer1_input': input_data_batch['monomer1_input'],
            'monomer2_input': input_data_batch['monomer2_input'],
            'group_input': input_data_batch['group_input'],
            'decoder_input1': decoder1_input,
            'decoder_input2': decoder2_input
        }, training=training)
        
        return predictions
    

        """Generate a pair of SMILES and groups"""


    # def generate(self, temperatures=[0.8], add_noise=False,
    #              smiles1=None,smiles2=None,group1=None,group2=None,trainging=True):
    #     """Generate a pair of SMILES and groups"""

    #     desired_groups = [group1, group2]

    #     tokens,tokens2  = tokenize_smiles([smiles1]),tokenize_smiles([smiles2])
    #     padded_tokens = pad_token(tokens, self.max_length, self.vocab)
    #     padded_tokens2 = pad_token(tokens2, self.max_length, self.vocab)
    #     input_seq = np.array(padded_tokens)  # Shape: (1, max_length)
    #     input_seq2 = np.array(padded_tokens2)  # Shape: (1, max_length)
    
    #     # Prepare group features
    #     group_features = encode_groups(desired_groups, Constants.GROUP_VOCAB)
    #     group_features = np.array([group_features])
    
    #     if add_noise:
    #         noisy_tokens1 = add_swap_noise(input_seq, 0.1)
    #         noisy_tokens2 = add_swap_noise(input_seq2, 0.1) 

    #         noisy_tokens1 = add_gaussian_noise(noisy_tokens1, 0.1)
    #         noisy_tokens2 = add_gaussian_noise(noisy_tokens2, 0.1)
    #         input_seq =noisy_tokens1
    #         input_seq2 =noisy_tokens2
    
       
    #     # Prepare group features
    #     group_features = encode_groups([group1, group2], Constants.GROUP_VOCAB)
    #     group_features = np.array([group_features])
        
    #     # Initialize decoder sequences
    #     decoder_seq1 = np.zeros((1, self.max_length))
    #     decoder_seq2 = np.zeros((1, self.max_length))
    #     decoder_seq1[0, 0] = self.vocab['<start>']
    #     decoder_seq2[0, 0] = self.vocab['<start>']
        
      
        
    #     def generate_monomer(training=True, single_toke=False):
    #         """Generate single monomer sequence"""
    #         # Initialize decoder sequence with proper shape
            
    #         decoder_seq1 = np.zeros((1, self.max_length))
    #         decoder_seq1[0, 0] = self.vocab['<start>']
    #         decoder_seq2 = np.zeros((1, self.max_length))
    #         decoder_seq2[0, 0] = self.vocab['<start>']
            

    #         all_tokens = []
            
    #         for temperature in temperatures:
    #             print(f"\nGenerating monomer pairs with temperature {temperature}:")
    #             print("=" * 80)
    #             print(f"Input SMILES: {smiles1}, {smiles2}")
    #             print(f"Desired Groups: {', '.join(desired_groups)}")
    #             print("-" * 80)
    #             generated_tokens = []
    #             generated_tokens2 = []

               
    #             if single_toke:
    #                 for i in range(5):  
    #                     output = self.prediction_model({
    #                         'monomer1_input': input_seq,
    #                         'monomer2_input': input_seq2,
    #                         'group_input': group_features,
    #                         'decoder_input1': decoder_seq1,
    #                         'decoder_input2': decoder_seq2
    #                         },training=training)
                        
                    
    #                     next_token_probs = output[0][0, i]
    #                     next_token, next_token_probs, next_token_probs_array = self.sample_with_temperature(next_token_probs, temperature)
    #                     #logprobs.append(np.log(next_token_probs[next_token] + 1e-7))

    #                     next_token_probs2 = output[1][0, i]
    #                     next_token2, next_token_probs2, next_token_probs_array2 = self.sample_with_temperature(next_token_probs2, temperature)
    #                     #logprobs2.append(np.log(next_token_probs2[next_token2] + 1e-7))
                    
    #                     generated_tokens.append(next_token)
    #                     generated_tokens2.append(next_token2)
    #                     if next_token == self.vocab['<end>']:
    #                         if next_token2 == self.vocab['<end>']:
    #                             break
    #                     if next_token2 == self.vocab['<end>']:
    #                         if next_token == self.vocab['<end>']:
    #                             break

    #                     if i < self.max_length - 2:
    #                         decoder_seq1[0, i + 1] = next_token
    #                         decoder_seq2[0, i + 1] = next_token
    #                 all_tokens.append([generated_tokens,generated_tokens2,temperature])

    #             else:
    #                  output = self.prediction_model({
    #                      'monomer1_input': input_seq,
    #                      'monomer2_input': input_seq2,
    #                      'group_input': group_features,
    #                      'decoder_input1': decoder_seq1,
    #                      'decoder_input2': decoder_seq2
    #                  },training=training)
    #                  output_logprobs = output[0][0]
    #                  output_logprobs2 = output[1][0]
    #                  tokens1 = self.get_tokens_from_probabilities(output_logprobs, self.vocab)  
    #                  tokens2 = self.get_tokens_from_probabilities(output_logprobs2, self.vocab)
    #                  all_tokens.append([tokens1,tokens2,temperature])
    #                  print(decode_smiles(tokens1))
    #                  print(decode_smiles(tokens2))
    #         return all_tokens
            
    #     tokens = generate_monomer(training=trainging, single_toke=False)
    #     return {
    #        'tokens': tokens
    #     }
    

    def shift_tokens_right(self,padded_tokens, pad_token_id):
        """Shift token sequence left and pad right with <pad>"""
        shifted = padded_tokens[:, :-1]
        pad_column = np.full((shifted.shape[0], 1), pad_token_id)
        return np.concatenate([shifted, pad_column], axis=1)


    def get_action_distribution(self, samples):
        """Get action distribution for samples"""
        action_distributions = []   
        for sample in samples:
            try:
                tokens1 = tokenize_smiles([sample['smiles1']])
                tokens2 = tokenize_smiles([sample['smiles2']])
                padded_tokens1 = pad_token(tokens1, self.max_length, self.vocab)
                padded_tokens2 = pad_token(tokens2, self.max_length, self.vocab)

                decoder_input1 = self.shift_tokens_right(np.array(padded_tokens1), self.vocab['<pad>'])
                decoder_input2 = self.shift_tokens_right(np.array(padded_tokens2), self.vocab['<pad>'])


                
                group_features = encode_groups(
                    [sample['group1'], sample['group2']], 
                    Constants.GROUP_VOCAB
                )
                group_features = np.array([group_features])
                
                output = self.prediction_model({
                    'monomer1_input': np.array(padded_tokens1),
                    'monomer2_input': np.array(padded_tokens2),
                    'group_input': group_features,
                    'decoder_input1': decoder_input1,#np.array(padded_tokens1),
                    'decoder_input2': decoder_input2#np.array(padded_tokens2)
                }, training=False)
                action_distribution1 = output[0][0]
                action_distribution2 = output[1][0]  # Second monomer
            
                combined_distribution = tf.concat([action_distribution1, action_distribution2], axis=0)
                action_distributions.append(combined_distribution)
            
            except Exception as e:
                print(f"Error getting action distribution: {e}")
                zero_dist = tf.zeros((2 * self.max_length, len(self.vocab)))
                action_distributions.append(zero_dist if not action_distributions else tf.zeros_like(action_distributions[0]))
        return tf.stack(action_distributions)



    def call(self, inputs):
        """Forward pass of the model"""
        return self.prediction_model(inputs)
    
    def get_config(self):
        config = super(GeneratorModel, self).get_config()
        config.update({
            'base_model': self.base_model,
            'smiles_vocab': self.vocab,
            'max_length': self.max_length,
        })
        return config

    @classmethod
    def from_config(cls, config):
        base_model = config.pop('base_model')
        smiles_vocab = config.pop('smiles_vocab')
        max_length = config.pop('max_length')
        return cls(base_model, smiles_vocab, max_length)
    

# if __name__ == "__main__":
#     save_dir_abs = os.path.join("Two_Monomers_Group", "pretrained_model", "saved_models_rl_gpu_3")
#     file_path = os.path.join('Two_Monomers_Group', 'Data', "smiles_orginal.xlsx")
#     based_model, smiles_vocab, model_params = load_and_retrain(save_dir=save_dir_abs)

#     generator = GeneratorModel(based_model, smiles_vocab, model_params['max_length'])
#     generator.prediction_model.load_weights(os.path.join(root_dir, "RLHF","Generator","models","group_based_rl_model_n2", "weights_model.weights.h5"))
#     smiles1 = "C1=CC=CC=C1"
#     smiles2 = "C1=CC=CC=C1"
#     group1 = "C=C"
#     group2 = "C=C"
#     tokens, tokens_probs, logprobs = generator.generate(temperatures=[0.8], add_noise=False,smiles1=smiles1,smiles2=smiles2,group1=group1,group2=group2,trainging=True)
#     print(decode_smiles(tokens[0][0]),"---------",decode_smiles(tokens[0][1]))
#     print(logprobs)






