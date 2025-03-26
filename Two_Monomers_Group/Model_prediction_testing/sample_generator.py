from Constants import *
from Data_Process_with_prevocab import *
from rdkit import Chem
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import os
from datetime import datetime
import json


from rdkit import Chem
from rdkit.Chem import Draw
import os
from datetime import datetime

def sample_with_temperature(predictions, temperature, repetition_penalty=1.2):
        """Sample from predictions with temperature scaling"""
        if temperature == 0:
            return np.argmax(predictions)
        
        sorted_probs = np.sort(predictions)[::-1]
        cumulative_probs = np.cumsum(sorted_probs)
        cutoff = float(sorted_probs[np.argmax(cumulative_probs > 0.9)])  # Use top 90% of probability mass
        predictions[predictions < cutoff] = 0
        predictions = predictions / np.sum(predictions)
        
        predictions = np.asarray(predictions).astype('float64')
        predictions = np.log(predictions + 1e-7) / temperature
        predictions = predictions / repetition_penalty
        exp_preds = np.exp(predictions)
        predictions = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, predictions, 1)
        return np.argmax(probas)


def add_swap_noise(tokens, swap_rate=0.1):
    """Randomly swap adjacent tokens"""
    noisy_tokens = tokens.copy()
    for i in range(len(tokens)):
        for j in range(1, len(tokens[i])-1):  # Avoid swapping start/end tokens
            # if np.random.random() < swap_rate:
            #     noisy_tokens[i][j], noisy_tokens[i][j+1] = \
            #     noisy_tokens[i][j+1], noisy_tokens[i][j]
            noisy_tokens[i][j], noisy_tokens[i][j+1] = noisy_tokens[i][j+1], noisy_tokens[i][j]
    return noisy_tokens

def add_gaussian_noise(tokens, noise_level=1):
    """Add Gaussian noise to token embeddings"""
    noise = np.random.normal(0, noise_level, tokens.shape)
    noisy_tokens = tokens + noise
    return noisy_tokens

def generate_monomer_pair_with_temperature(model, input_smiles, desired_groups, vocab, max_length, temperatures=[0.2], 
                                           group_smarts1=None, group_smarts2=None,add_noise=False):
    # Prepare input SMILES
    tokens,tokens2  = tokenize_smiles([input_smiles[0]]),tokenize_smiles([input_smiles[1]])
    padded_tokens = pad_token(tokens, max_length, vocab)
    padded_tokens2 = pad_token(tokens2, max_length, vocab)
    input_seq = np.array(padded_tokens)  # Shape: (1, max_length)
    input_seq2 = np.array(padded_tokens2)  # Shape: (1, max_length)

    if add_noise:
        noisy_tokens1 = add_swap_noise(input_seq, 0.1)
        noisy_tokens2 = add_swap_noise(input_seq2, 0.1) 

        noisy_tokens1 = add_gaussian_noise(noisy_tokens1, 0.1)
        noisy_tokens2 = add_gaussian_noise(noisy_tokens2, 0.1)
        input_seq =noisy_tokens1
        input_seq2 =noisy_tokens2


    # print("Original Smiles: ", decode_smiles(input_seq[0]), "--", decode_smiles(input_seq2[0]))
    # print("Noisy Smiles: ", decode_smiles(noisy_tokens1[0]), "--", decode_smiles(noisy_tokens2[0]))






    
    # Prepare group features
    group_features = encode_groups(desired_groups, Constants.GROUP_VOCAB)
    group_features = np.array([group_features])
    
   
    
    def generate_monomer():
        """Generate single monomer sequence"""
        # Initialize decoder sequence with proper shape
        
        decoder_seq = np.zeros((1, max_length))
        decoder_seq[0, 0] = vocab['<start>']
        decoder_seq2 = np.zeros((1, max_length))
        decoder_seq2[0, 0] = vocab['<start>']
        all_tokens = []
        for temperature in temperatures:
            generated_tokens = []
            generated_tokens2 = []
            #it should be 245
            for i in range(max_length):  
                output = model.predict({
                    'monomer1_input': input_seq,
                    'monomer2_input': input_seq2,
                    'group_input': group_features,
                    'decoder_input1': decoder_seq,
                    'decoder_input2': decoder_seq2
                    },verbose=0)
            
                next_token_probs = output[0][0, i]
                next_token = sample_with_temperature(next_token_probs, temperature)

                next_token_probs2 = output[1][0, i]
                next_token2 = sample_with_temperature(next_token_probs2, temperature)
            
                generated_tokens.append(next_token)
                generated_tokens2.append(next_token2)
                if next_token == vocab['<end>']:
                    if next_token2 == vocab['<end>']:
                        break
                if next_token2 == vocab['<end>']:
                    if next_token == vocab['<end>']:
                        break

                if i < max_length - 2:
                    decoder_seq[0, i + 1] = next_token
                    decoder_seq2[0, i + 1] = next_token
            all_tokens.append([generated_tokens,generated_tokens2,temperature])
        return all_tokens
    
    

    valid_pairs = []

    valid_output_file = "valid_pairs_new.json"
    valid_pairs_j = []

    try:
        with open(valid_output_file, 'r') as f:
            valid_pairs_j = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        valid_pairs_j = []
    
   
    tokens1 = generate_monomer()  # decoder1
    #tokens2 = generate_monomer(1)  # decoder2
    for i in range(len(tokens1)):
        # Convert to SMILES
        smiles1,smiles2 = decode_smiles(tokens1[i][0]), decode_smiles(tokens1[i][1])#decode_smiles(tokens1[i][0])
        #smiles2 = decode_smiles(tokens2[i][0])
        temperature = tokens1[i][2]
        
        # Check validity
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        valid1 = mol1 is not None and smiles1 != ""
        valid2 = mol2 is not None and smiles2 != ""
        
        print(f"\nPair {i+1}:")
        print(f"Monomer 1: {smiles1}")
        print(f"Valid: {valid1}")
        print(f"Monomer 2: {smiles2}")
        print(f"Valid: {valid2}")

        if valid1 and valid2:
            # Create dictionary with pair info
            pair_info = {
                "input_smiles": input_smiles,
                "temperature": temperature,
                "desired_groups": desired_groups,
                "Input Group SMARTS 1": group_smarts1,
                "Input Group SMARTS 2": group_smarts2,
                "monomer1": {
                    "smiles": smiles1,
                },
                "monomer2": {
                    "smiles": smiles2, 
                },
                "add_noise": add_noise
            }
                        
            valid_pairs.append((smiles1, smiles2))
            valid_pairs_j.append(pair_info)
            #save_smiles_pair_as_image(pair_info)
            with open(valid_output_file, 'w') as f:
                json.dump(valid_pairs_j, f, indent=4)

    return valid_pairs


# from rdkit import DataStructs
# from rdkit.Chem import AllChem
# import random
# from rdkit import Chem
# from Constants import *
# from Data_Process_with_prevocab import *
# from rdkit import Chem
# import numpy as np
# from rdkit import Chem
# from rdkit.Chem import Draw
# import os
# from datetime import datetime
# import json


# from rdkit import Chem
# from rdkit.Chem import Draw
# import os
# from datetime import datetime

# def calculate_diversity_penalty(smiles1, smiles2, existing_pairs):
#     """Calculate diversity penalty based on Tanimoto similarity"""
#     if not existing_pairs:
#         return 0.0
    
#     mol1 = Chem.MolFromSmiles(smiles1)
#     mol2 = Chem.MolFromSmiles(smiles2)
#     fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2)
#     fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2)
    
#     max_similarity = 0
#     for pair in existing_pairs:
#         existing_mol1 = Chem.MolFromSmiles(pair[0])
#         existing_mol2 = Chem.MolFromSmiles(pair[1])
#         existing_fp1 = AllChem.GetMorganFingerprintAsBitVect(existing_mol1, 2)
#         existing_fp2 = AllChem.GetMorganFingerprintAsBitVect(existing_mol2, 2)
        
#         sim1 = DataStructs.TanimotoSimilarity(fp1, existing_fp1)
#         sim2 = DataStructs.TanimotoSimilarity(fp2, existing_fp2)
#         max_similarity = max(max_similarity, (sim1 + sim2) / 2)
    
#     return max_similarity

# class BeamSearchNode:
#     def __init__(self, tokens1, tokens2, score, decoder_state1=None, decoder_state2=None):
#         self.tokens1 = tokens1
#         self.tokens2 = tokens2
#         self.score = score
#         self.decoder_state1 = decoder_state1
#         self.decoder_state2 = decoder_state2

# def generate_with_diverse_beam_search(model, input_smiles, desired_groups, vocab, max_length, 
#                                     beam_width=5, num_groups=3, diversity_penalty_weight=0.5):
#     """Generate sequences using diverse beam search"""
#     tokens, tokens2 = tokenize_smiles([input_smiles[0]]), tokenize_smiles([input_smiles[1]])
#     padded_tokens = pad_token(tokens, max_length, vocab)
#     padded_tokens2 = pad_token(tokens2, max_length, vocab)
#     input_seq = np.array(padded_tokens)
#     input_seq2 = np.array(padded_tokens2)
    
#     group_features = encode_groups(desired_groups, Constants.GROUP_VOCAB)
#     group_features = np.array([group_features])
    
#     # Initialize beam groups
#     beam_groups = [[] for _ in range(num_groups)]
#     for group in range(num_groups):
#         decoder_seq = np.zeros((1, max_length))
#         decoder_seq2 = np.zeros((1, max_length))
#         decoder_seq[0, 0] = vocab['<start>']
#         decoder_seq2[0, 0] = vocab['<start>']
        
#         initial_node = BeamSearchNode(
#             tokens1=[vocab['<start>']],
#             tokens2=[vocab['<start>']],
#             score=0.0
#         )
#         beam_groups[group].append(initial_node)
    
#     # Generate sequences
#     finished_sequences = []
#     for pos in range(max_length - 1):
#         for group_idx, group in enumerate(beam_groups):
#             if not group:
#                 continue
                
#             candidates = []
#             for node in group:
#                 decoder_seq = np.zeros((1, max_length))
#                 decoder_seq2 = np.zeros((1, max_length))
#                 decoder_seq[0, :len(node.tokens1)] = node.tokens1
#                 decoder_seq2[0, :len(node.tokens2)] = node.tokens2
                
#                 output = model.predict({
#                     'monomer1_input': input_seq,
#                     'monomer2_input': input_seq2,
#                     'group_input': group_features,
#                     'decoder_input1': decoder_seq,
#                     'decoder_input2': decoder_seq2
#                 }, verbose=0)
                
#                 probs1 = output[0][0, len(node.tokens1)-1]
#                 probs2 = output[1][0, len(node.tokens2)-1]
                
#                 # Get top-k candidates for each sequence
#                 top_k1 = np.argsort(probs1)[-beam_width:]
#                 top_k2 = np.argsort(probs2)[-beam_width:]
                
#                 for tok1 in top_k1:
#                     for tok2 in top_k2:
#                         new_tokens1 = node.tokens1 + [tok1]
#                         new_tokens2 = node.tokens2 + [tok2]
                        
#                         # Calculate score with diversity penalty
#                         score = node.score + np.log(probs1[tok1] + 1e-10) + np.log(probs2[tok2] + 1e-10)
                        
#                         # Add diversity penalty between groups
#                         if finished_sequences:
#                             diversity_penalty = calculate_diversity_penalty(
#                                 decode_smiles(new_tokens1),
#                                 decode_smiles(new_tokens2),
#                                 finished_sequences
#                             )
#                             score -= diversity_penalty_weight * diversity_penalty
                        
#                         candidates.append(BeamSearchNode(
#                             tokens1=new_tokens1,
#                             tokens2=new_tokens2,
#                             score=score
#                         ))
            
#             # Select top candidates for this group
#             candidates.sort(key=lambda x: x.score, reverse=True)
#             beam_groups[group_idx] = candidates[:beam_width]
            
#             # Check for completed sequences
#             for node in beam_groups[group_idx]:
#                 if vocab['<end>'] in node.tokens1 and vocab['<end>'] in node.tokens2:
#                     smiles1 = decode_smiles(node.tokens1)
#                     smiles2 = decode_smiles(node.tokens2)
#                     if Chem.MolFromSmiles(smiles1) and Chem.MolFromSmiles(smiles2):
#                         finished_sequences.append((smiles1, smiles2))
    
#     return finished_sequences

# def generate_with_mcmc(model, input_smiles, desired_groups, vocab, max_length, 
#                       num_steps=100, temperature=1.0):
#     """Generate sequences using MCMC sampling"""
#     def propose_modification(tokens):
#         """Propose a small modification to the sequence"""
#         new_tokens = tokens.copy()
#         pos = random.randint(1, len(tokens)-2)  # Don't modify start/end tokens
#         new_tokens[pos] = random.randint(0, len(vocab)-1)
#         return new_tokens
    
#     def sequence_probability(tokens1, tokens2):
#         """Calculate sequence probability under the model"""
#         decoder_seq = np.zeros((1, max_length))
#         decoder_seq2 = np.zeros((1, max_length))
#         decoder_seq[0, :len(tokens1)] = tokens1
#         decoder_seq2[0, :len(tokens2)] = tokens2
        
#         output = model.predict({
#             'monomer1_input': input_seq,
#             'monomer2_input': input_seq2,
#             'group_input': group_features,
#             'decoder_input1': decoder_seq,
#             'decoder_input2': decoder_seq2
#         }, verbose=0)
        
#         log_prob = 0
#         for i in range(len(tokens1)-1):
#             log_prob += np.log(output[0][0, i, tokens1[i+1]] + 1e-10)
#         for i in range(len(tokens2)-1):
#             log_prob += np.log(output[1][0, i, tokens2[i+1]] + 1e-10)
#         return log_prob
    
#     # Initialize
#     tokens, tokens2 = tokenize_smiles([input_smiles[0]]), tokenize_smiles([input_smiles[1]])
#     padded_tokens = pad_token(tokens, max_length, vocab)
#     padded_tokens2 = pad_token(tokens2, max_length, vocab)
#     input_seq = np.array(padded_tokens)
#     input_seq2 = np.array(padded_tokens2)
    
#     group_features = encode_groups(desired_groups, Constants.GROUP_VOCAB)
#     group_features = np.array([group_features])
    
#     current_tokens1 = [vocab['<start>']] + [random.randint(0, len(vocab)-1) for _ in range(max_length-2)] + [vocab['<end>']]
#     current_tokens2 = [vocab['<start>']] + [random.randint(0, len(vocab)-1) for _ in range(max_length-2)] + [vocab['<end>']]
    
#     current_prob = sequence_probability(current_tokens1, current_tokens2)
    
#     accepted_sequences = []
#     for step in range(num_steps):
#         # Propose new sequences
#         proposed_tokens1 = propose_modification(current_tokens1)
#         proposed_tokens2 = propose_modification(current_tokens2)
        
#         # Calculate acceptance probability
#         proposed_prob = sequence_probability(proposed_tokens1, proposed_tokens2)
#         acceptance_ratio = np.exp((proposed_prob - current_prob) / temperature)
        
#         if random.random() < acceptance_ratio:
#             current_tokens1 = proposed_tokens1
#             current_tokens2 = proposed_tokens2
#             current_prob = proposed_prob
            
#             # Check validity and add to accepted sequences
#             smiles1 = decode_smiles(current_tokens1)
#             smiles2 = decode_smiles(current_tokens2)
#             if Chem.MolFromSmiles(smiles1) and Chem.MolFromSmiles(smiles2):
#                 accepted_sequences.append((smiles1, smiles2))
    
#     return accepted_sequences

# def generate_monomer_pair_with_temperature2(model, input_smiles, desired_groups, vocab, max_length, 
#                                          temperatures=[0.2], group_smarts1=None, group_smarts2=None,
#                                          use_beam_search=True, use_mcmc=True):
#     valid_pairs = []
    
#     # Generate using diverse beam search
#     if use_beam_search:
#         beam_pairs = generate_with_diverse_beam_search(
#             model, input_smiles, desired_groups, vocab, max_length,
#             beam_width=5, num_groups=3, diversity_penalty_weight=0.5
#         )
#         valid_pairs.extend(beam_pairs)
    
#     # Generate using MCMC sampling
#     if use_mcmc:
#         mcmc_pairs = generate_with_mcmc(
#             model, input_smiles, desired_groups, vocab, max_length,
#             num_steps=100, temperature=1.0
#         )
#         valid_pairs.extend(mcmc_pairs)
    
#     # # Original temperature-based sampling
#     # temperature_pairs = generate_with_temperature(
#     #     model, input_smiles, desired_groups, vocab, max_length,
#     #     temperatures, group_smarts1, group_smarts2
#     # )
#     # valid_pairs.extend(temperature_pairs)
    
#     # Remove duplicates while preserving order
#     seen = set()
#     unique_pairs = []
#     for pair in valid_pairs:
#         if pair not in seen:
#             seen.add(pair)
#             unique_pairs.append(pair)
    
#     print(unique_pairs)
#     print(seen)
#     print(len(unique_pairs))
#     print(len(seen))
#     return unique_pairs
