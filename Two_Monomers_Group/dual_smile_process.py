import Constants
import pandas as pd
import numpy as np
import tensorflow as tf
from rdkit import Chem
from Data_Process_with_prevocab import *
import random

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
   
    all_groups = []
    
    for m1, m2 in zip(monomer1_list, monomer2_list):
        found_groups_m1 = []
        found_groups_m2 = []
        
        # Check for each group in monomer 1
        if count_functional_groups(m1, Constants.EPOXY_SMARTS) >= 2:
            found_groups_m1.append("C1OC1")
        if count_functional_groups(m1, Constants.IMINE_SMARTS) >= 2:
            found_groups_m1.append("NC")
        if count_functional_groups(m1, Constants.THIOL_SMARTS) >= 2:
            found_groups_m1.append("CCS")
        if count_functional_groups(m1, Constants.ACRYL_SMARTS) >= 2:
            found_groups_m1.append("C=C(C=O)")
        if count_functional_groups(m1, Constants.VINYL_SMARTS) >= 2:
            found_groups_m1.append("C=C")
        if not (count_functional_groups(m1, Constants.EPOXY_SMARTS) >= 2 or 
                count_functional_groups(m1, Constants.IMINE_SMARTS) >= 2 or
                count_functional_groups(m1, Constants.THIOL_SMARTS) >= 2 or 
                count_functional_groups(m1, Constants.ACRYL_SMARTS) >= 2 or
                count_functional_groups(m1, Constants.VINYL_SMARTS) >= 2):
            found_groups_m1.append("No group")
        # Check for each group in monomer 2
        if count_functional_groups(m2, Constants.EPOXY_SMARTS) >= 2:
            found_groups_m2.append("C1OC1")
        if count_functional_groups(m2, Constants.IMINE_SMARTS) >= 2:
            found_groups_m2.append("NC")
        if count_functional_groups(m2, Constants.THIOL_SMARTS) >= 2:
            found_groups_m2.append("CCS")
        if count_functional_groups(m2, Constants.ACRYL_SMARTS) >= 2:
            found_groups_m2.append("C=C(C=O)")
        if count_functional_groups(m2, Constants.VINYL_SMARTS) >= 2:
            found_groups_m2.append("C=C")
        if not (count_functional_groups(m2, Constants.EPOXY_SMARTS) >= 2 or 
                count_functional_groups(m2, Constants.IMINE_SMARTS) >= 2 or
                count_functional_groups(m2, Constants.THIOL_SMARTS) >= 2 or 
                count_functional_groups(m2, Constants.ACRYL_SMARTS) >= 2 or
                count_functional_groups(m2, Constants.VINYL_SMARTS) >= 2):
            found_groups_m2.append("No group")
        
        # Combine groups from both monomers
        combined_groups = found_groups_m1 + found_groups_m2
        
        all_groups.append(combined_groups)
    
    # Encode groups using the vocabulary
    encoded_groups = [encode_groups(groups, Constants.GROUP_VOCAB) for groups in all_groups]
    
    return encoded_groups

def prepare_training_data2(max_length, vocab,file_path):
    monomer1_list, monomer2_list = process_dual_monomer_data(file_path)
    valid_reaction = reaction_valid_samples(monomer1_list,monomer2_list)
    monomer1_list, monomer2_list = monomer1_list[:10], monomer2_list[:10]
    group_features = encode_functional_groups(monomer1_list, monomer2_list)
    tokens1 = tokenize_smiles(monomer1_list)
    tokens2 = tokenize_smiles(monomer2_list)
    
    # Add 1 to max_length to match model's expected shape
    padded_tokens1 = pad_token(tokens1, max_length + 1, vocab)
    padded_tokens2 = pad_token(tokens2, max_length + 1, vocab)

    decoder_input1,decoder_output1 = make_target(padded_tokens1)
    decoder_input2,decoder_output2 = make_target(padded_tokens2)

    
    # Convert to numpy arrays
    padded_tokens1 = np.array(padded_tokens1)
    padded_tokens2 = np.array(padded_tokens2)
    group_features = np.array(group_features)

    decoder_input1 = np.array(decoder_input1)
    decoder_output1 = np.array(decoder_output1)
    decoder_input2 = np.array(decoder_input2)
    decoder_output2 = np.array(decoder_output2)
    
    
    # Ensure group_features has the correct shape (batch_size, num_groups)
    if len(group_features.shape) > 2:
        group_features = group_features.reshape(group_features.shape[0], -1)
    
    # Print shapes for debugging
    print("Input shapes:")
    print(f"monomer1_input shape: {padded_tokens1.shape}")
    print(f"monomer2_input shape: {padded_tokens2.shape}")
    print(f"group_input shape: {group_features.shape}")
    print(f"decoder_input1 shape: {decoder_input1.shape}")
    print(f"decoder_input2 shape: {decoder_input2.shape}")
    
    # Create target data (shifted by one position)
    # target1 = tf.keras.utils.to_categorical(padded_tokens1[:, 1:], num_classes=len(vocab))
    # target2 = tf.keras.utils.to_categorical(padded_tokens2[:, 1:], num_classes=len(vocab))

    target1 = tf.keras.utils.to_categorical(padded_tokens1[:, 1:], num_classes=len(vocab))
    target2 = tf.keras.utils.to_categorical(padded_tokens2[:, 1:], num_classes=len(vocab))
    # target1 = padded_tokens1[:, 1:]  # Next token prediction

    orginal_tokens1 = pad_token(tokens1, max_length , vocab)
    orginal_tokens2 = pad_token(tokens2, max_length , vocab)

    # originial_smiles1 = tf.keras.utils.to_categorical(orginal_tokens1, num_classes=len(vocab),dtype=np.int32)
    # originial_smiles2 = tf.keras.utils.to_categorical(orginal_tokens2, num_classes=len(vocab),dtype=np.int32)
    originial_smiles1 = tf.keras.utils.to_categorical(orginal_tokens1, num_classes=len(vocab))
    originial_smiles2 = tf.keras.utils.to_categorical(orginal_tokens2, num_classes=len(vocab))
    # target2 = padded_tokens2[:, 1:] 
    # target1 = target1.reshape(target1.shape[0], target1.shape[1], 1)
    # target2 = target2.reshape(target2.shape[0], target2.shape[1], 1)
    # print(target1.shape)
    # print(target2.shape)
    
    print("Target shapes:")
    print(f"target1 shape: {target1.shape}")
    print(f"target2 shape: {target2.shape}")
    print(f"decoder_output1 shape: {decoder_output1[:, :].shape}")
    print(f"decoder_output2 shape: {decoder_output2[:, :].shape}")
    
    # Return properly formatted dictionaries
    inputs = {
        'monomer1_input': padded_tokens1[:, :-1],
        'monomer2_input': padded_tokens2[:, :-1],
        'group_input': group_features,
        'decoder_input1': decoder_input1[:, :-1],
        'decoder_input2': decoder_input2[:, :-1],
        'original_monomer1': originial_smiles1,
        'original_monomer2': originial_smiles2
    }
    
    outputs = {
        'monomer1_output': target1,
        'monomer2_output': target2
    }
    
    return inputs, outputs

def reaction_valid_samples(smiles1,smiles2):
 
    valid_reaction = []
    invalid_reaction = []
    for i in range(len(smiles1)):
        reaction_valid = filter_valid_groups(smiles1[i], smiles2[i])
        if reaction_valid:
            valid_reaction.append([smiles1[i],smiles2[i]])
        else:
            invalid_reaction.append([smiles1[i],smiles2[i]])

    print(len(valid_reaction))
    print(len(invalid_reaction))
    random_invalid_reaction = random.sample(invalid_reaction, 259)
    valid_reaction.extend(random_invalid_reaction) 
    print(len(valid_reaction))
    return valid_reaction
    

def prepare_training_data(max_length, vocab, file_path):
    monomer1_list, monomer2_list = process_dual_monomer_data(file_path)
    #monomer1_list, monomer2_list = monomer1_list[:2], monomer2_list[:2]
    valid_reaction = reaction_valid_samples(monomer1_list,monomer2_list)
    monomer1_list = [i[0] for i in valid_reaction]
    monomer2_list = [i[1] for i in valid_reaction]
  

    monomer1_list, monomer2_list = monomer1_list[:2], monomer2_list[:2]

    # Encode functional groups
    group_features = encode_functional_groups(monomer1_list, monomer2_list)
    
    # Tokenize SMILES
    tokens1 = tokenize_smiles(monomer1_list)
    tokens2 = tokenize_smiles(monomer2_list)

    # Pad tokens (add 1 to max_length for consistency)
    padded_tokens1 = pad_token(tokens1, max_length + 1, vocab)
    padded_tokens2 = pad_token(tokens2, max_length + 1, vocab)

    # Generate decoder inputs and outputs
    decoder_input1, decoder_output1 = make_target(padded_tokens1)
    decoder_input2, decoder_output2 = make_target(padded_tokens2)

    # Convert to numpy arrays using reduced data types
    padded_tokens1 = np.array(padded_tokens1, dtype=np.int32)  # Use int32 instead of float64
    padded_tokens2 = np.array(padded_tokens2, dtype=np.int32)
    group_features = np.array(group_features, dtype=np.float32)  # Use float32 instead of float64

    decoder_input1 = np.array(decoder_input1, dtype=np.int32)
    decoder_output1 = np.array(decoder_output1, dtype=np.int32)
    decoder_input2 = np.array(decoder_input2, dtype=np.int32)
    decoder_output2 = np.array(decoder_output2, dtype=np.int32)

    # Ensure group_features has correct shape
    if len(group_features.shape) > 2:
        group_features = group_features.reshape(group_features.shape[0], -1)

    # ✅ Use integer encoding instead of one-hot to reduce memory
    target1 = padded_tokens1[:, 1:] 
    target2 = padded_tokens2[:, 1:]

    # ✅ Integer encoding instead of one-hot encoding
    original_tokens1 = pad_token(tokens1, max_length, vocab)
    original_tokens2 = pad_token(tokens2, max_length, vocab)
    original_smiles1 = np.array(original_tokens1, dtype=np.int32)
    original_smiles2 = np.array(original_tokens2, dtype=np.int32)
    print(original_smiles1.shape)
    print(original_smiles2.shape)
    print("Example of original_smiles1: ", decode_smiles(original_smiles1[0]))
    print("Example of original_smiles2: ", decode_smiles(original_smiles2[0]))

    # Debugging info
    print("Input shapes:")
    print(f"monomer1_input shape: {padded_tokens1.shape}")
    print(f"monomer2_input shape: {padded_tokens2.shape}")
    print(f"group_input shape: {group_features.shape}")
    print(f"decoder_input1 shape: {decoder_input1.shape}")
    print(f"decoder_input2 shape: {decoder_input2.shape}")

    print("Target shapes:")
    print(f"target1 shape: {target1.shape}")
    print(f"target2 shape: {target2.shape}")
    print(f"decoder_output1 shape: {decoder_output1.shape}")
    print(f"decoder_output2 shape: {decoder_output2.shape}")

    # ✅ Return as dictionary
    inputs = {
        'monomer1_input': padded_tokens1[:, :-1],
        'monomer2_input': padded_tokens2[:, :-1],
        'group_input': group_features,
        'decoder_input1': decoder_input1[:, :-1],
        'decoder_input2': decoder_input2[:, :-1],
        'original_monomer1': original_smiles1,
        'original_monomer2': original_smiles2
    }
    
    outputs = {
        'monomer1_output': target1,
        'monomer2_output': target2
    }
    
    return inputs, outputs


def check_reaction_validity(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return False,[]
    if count_functional_groups(smiles1, Constants.EPOXY_SMARTS) >= 2 and count_functional_groups(smiles2, Constants.IMINE_SMARTS) >= 2:
        return True,['C1OC1','NC']
    if count_functional_groups(smiles1, Constants.IMINE_SMARTS) >= 2 and count_functional_groups(smiles2, Constants.EPOXY_SMARTS) >= 2:
        return True,['NC','C1OC1']
    if count_functional_groups(smiles1, Constants.VINYL_SMARTS) >= 2 and count_functional_groups(smiles2, Constants.THIOL_SMARTS) >= 2:
        return True,['C=C','CCS']
    if count_functional_groups(smiles1, Constants.THIOL_SMARTS) >= 2 and count_functional_groups(smiles2, Constants.VINYL_SMARTS) >= 2:
        return True,['CCS','C=C']
    if count_functional_groups(smiles1, Constants.VINYL_SMARTS) >= 2 and count_functional_groups(smiles2, Constants.ACRYL_SMARTS) >= 2:
        return True,['C=C','C=C(C=O)']
    if count_functional_groups(smiles1, Constants.ACRYL_SMARTS) >= 2 and count_functional_groups(smiles2, Constants.VINYL_SMARTS) >= 2:
        return True,['C=C(C=O)','C=C']  
    return False,[]

def check_reaction_validity_with_invalid_groups(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    pairs = [
        (Constants.VINYL_SMARTS, Constants.THIOL_SMARTS, ['C=C', 'CCS']),
        (Constants.THIOL_SMARTS, Constants.VINYL_SMARTS, ['CCS', 'C=C']),
        (Constants.VINYL_SMARTS, Constants.ACRYL_SMARTS, ['C=C', 'C=C(C=O)']),
        (Constants.ACRYL_SMARTS, Constants.VINYL_SMARTS, ['C=C(C=O)', 'C=C']),
        (Constants.EPOXY_SMARTS, Constants.IMINE_SMARTS, ['C1OC1', 'NC']),
        (Constants.IMINE_SMARTS, Constants.EPOXY_SMARTS, ['NC', 'C1OC1']),
        (Constants.VINYL_SMARTS, Constants.VINYL_SMARTS, ['C=C', 'C=C']),
        
    ]
    labels = ["No_group","No_group"]
    total_count = 0
    found = False
    for smarts1, smarts2, labels in pairs:
        count1 = count_functional_groups(smiles1, smarts1)
        count2 = count_functional_groups(smiles2, smarts2)
        total = count1 + count2
        if count1 >= 2 and count2 >= 2:
            labels[0] = smarts1
            labels[1] = smarts2
            total_count = total
            found = True
            break
        elif count1 > 0 and count2 > 0:
            labels[0] = smarts1
            labels[1] = smarts2
            total_count = total
            found = True
            break
        elif count1 > 0 and count2 == 0:
            labels[0] = smarts1
            labels[1] = "No_group"
            total_count = count1
            found = True
            break
        elif count1 == 0 and count2 > 0:
            labels[0] = "No_group"
            labels[1] = smarts2
            total_count = count2
            found = True
            break
        else:
            labels[0] = "No_group"
            labels[1] = "No_group"
            total_count = 0
            found = False
        
        
    
    if found:
        return labels, total_count
    else:
        return ["No_group", "No_group"], 0
    
def filter_valid_groups(smiles1, smiles2):
    pairs = [
        (Constants.VINYL_SMARTS, Constants.THIOL_SMARTS, ['C=C', 'CCS']),
        (Constants.THIOL_SMARTS, Constants.VINYL_SMARTS, ['CCS', 'C=C']),
        (Constants.VINYL_SMARTS, Constants.ACRYL_SMARTS, ['C=C', 'C=C(C=O)']),
        (Constants.ACRYL_SMARTS, Constants.VINYL_SMARTS, ['C=C(C=O)', 'C=C']),
        (Constants.EPOXY_SMARTS, Constants.IMINE_SMARTS, ['C1OC1', 'NC']),
        (Constants.IMINE_SMARTS, Constants.EPOXY_SMARTS, ['NC', 'C1OC1']),
        (Constants.VINYL_SMARTS, Constants.VINYL_SMARTS, ['C=C', 'C=C']),
        
    ]
    for smarts1, smarts2, labels in pairs:
        count1 = count_functional_groups(smiles1, smarts1)
        count2 = count_functional_groups(smiles2, smarts2)
        if count1 >= 2 and count2 >= 2:
            return True
       
        else:
            return False
    
    
    
def add_gaussian_noise(tokens, noise_level=1):
    """Add Gaussian noise to token embeddings"""
    noise = np.random.normal(0, noise_level, tokens.shape)
    noisy_tokens = tokens + noise
    return noisy_tokens

def add_dropout_noise(tokens, dropout_rate=0.1):
    """Randomly zero out some tokens"""
    mask = np.random.binomial(1, 1-dropout_rate, tokens.shape)
    return tokens * mask

def add_swap_noise2(tokens, swap_rate=0.1):
    """Randomly swap adjacent tokens"""
    noisy_tokens = tokens.copy()
    for i in range(len(tokens)):
        for j in range(1, len(tokens[i])-1):  # Avoid swapping start/end tokens
            noisy_tokens[i][j], noisy_tokens[i][j+1] = \
                noisy_tokens[i][j+1], noisy_tokens[i][j]
            #if np.random.random() < swap_rate:
            #    noisy_tokens[i][j], noisy_tokens[i][j+1] = \
            #    noisy_tokens[i][j+1], noisy_tokens[i][j]
    return noisy_tokens

def add_mask_noise(tokens, vocab, mask_rate=0.1):
    """Randomly mask tokens with MASK token"""
    mask_token = vocab.get('[MASK]', len(vocab)-1)  # Use last token if no mask token
    noisy_tokens = tokens.copy()
    mask = np.random.random(tokens.shape) < mask_rate
    noisy_tokens[mask] = mask_token
    return noisy_tokens

def prepare_training_data_with_noise2(max_length, vocab, file_path, noise_config=None):
    """
    Prepare training data with various types of noise
    
    noise_config = {
        'gaussian': {'enabled': True, 'level': 0.1},
        'dropout': {'enabled': True, 'rate': 0.1},
        'swap': {'enabled': True, 'rate': 0.1},
        'mask': {'enabled': True, 'rate': 0.1}
    }
    """
    # Default noise configuration
    if noise_config is None:
        noise_config = {
            'gaussian': {'enabled': False, 'level': 0.1},
            'dropout': {'enabled': False, 'rate': 0.1},
            'swap': {'enabled': False, 'rate': 0.1},
            'mask': {'enabled': False, 'rate': 0.05}
        }

    # Original data preparation
    monomer1_list, monomer2_list = process_dual_monomer_data(file_path)
    valid_reaction = reaction_valid_samples(monomer1_list,monomer2_list)
    monomer1_list = [i[0] for i in valid_reaction]
    monomer2_list = [i[1] for i in valid_reaction]
    monomer1_list, monomer2_list = monomer1_list[:2], monomer2_list[:2]
    group_features = encode_functional_groups(monomer1_list, monomer2_list)
    tokens1 = tokenize_smiles(monomer1_list)
    tokens2 = tokenize_smiles(monomer2_list)
    
    padded_tokens1 = pad_token(tokens1, max_length + 1, vocab)
    padded_tokens2 = pad_token(tokens2, max_length + 1, vocab)
    
    # Convert to numpy arrays
    padded_tokens1 = np.array(padded_tokens1)
    padded_tokens2 = np.array(padded_tokens2)
    
    # Apply noise layers sequentially
    noisy_tokens1 = padded_tokens1.copy()
    noisy_tokens2 = padded_tokens2.copy()
    
    
        
    if noise_config['swap']['enabled']:
        print("Swap noise enabled")
        noisy_tokens1 = add_swap_noise2(noisy_tokens1, 
                                     noise_config['swap']['rate'])
        noisy_tokens2 = add_swap_noise2(noisy_tokens2, 
                                     noise_config['swap']['rate'])
    if noise_config['gaussian']['enabled']:
        print("Gaussian noise enabled")
        noisy_tokens1 = add_gaussian_noise(noisy_tokens1, 
                                         noise_config['gaussian']['level'])
        noisy_tokens2 = add_gaussian_noise(noisy_tokens2, 
                                         noise_config['gaussian']['level'])
        
    if noise_config['dropout']['enabled']:
        print("Dropout noise enabled")
        noisy_tokens1 = add_dropout_noise(noisy_tokens1, 
                                        noise_config['dropout']['rate'])
        noisy_tokens2 = add_dropout_noise(noisy_tokens2, 
                                        noise_config['dropout']['rate'])
        
    if noise_config['mask']['enabled']:
        print("Mask noise enabled")
        noisy_tokens1 = add_mask_noise(noisy_tokens1, vocab, 
                                     noise_config['mask']['rate'])
        noisy_tokens2 = add_mask_noise(noisy_tokens2, vocab, 
                                     noise_config['mask']['rate'])

    # Create decoder inputs/outputs with noisy data
    decoder_input1, decoder_output1 = make_target(noisy_tokens1)
    decoder_input2, decoder_output2 = make_target(noisy_tokens2)
    
    # Store original (non-noisy) data for comparison
    orginal_tokens1 = pad_token(tokens1, max_length, vocab)
    orginal_tokens2 = pad_token(tokens2, max_length, vocab)
    originial_smiles1 = tf.keras.utils.to_categorical(orginal_tokens1, num_classes=len(vocab))
    originial_smiles2 = tf.keras.utils.to_categorical(orginal_tokens2, num_classes=len(vocab))
    
    # Create targets
    target1 = tf.keras.utils.to_categorical(padded_tokens1[:, 1:], num_classes=len(vocab))
    target2 = tf.keras.utils.to_categorical(padded_tokens2[:, 1:], num_classes=len(vocab))
    
    # Return formatted dictionaries with both noisy and original data
    inputs = {
        'monomer1_input': noisy_tokens1[:, :-1],  # Use noisy tokens for input
        'monomer2_input': noisy_tokens2[:, :-1],
        'group_input': group_features,
        'decoder_input1': decoder_input1[:, :-1],
        'decoder_input2': decoder_input2[:, :-1],
        'original_monomer1': originial_smiles1,  # Keep original data for comparison
        'original_monomer2': originial_smiles2
    }
    
    outputs = {
        'monomer1_output': target1,  # Keep clean targets
        'monomer2_output': target2
    }
    
    return inputs, outputs


def add_swap_noise(tokens, swap_rate=0.3, context_size=3):
    noisy_tokens = tokens.copy()
    print(len(tokens))
    for i in range(len(tokens)):
        num_swaps = int(len(tokens[i]) * swap_rate)
        for _ in range(num_swaps):
            idx = np.random.randint(1, len(tokens[i]) - 1)
            start = max(1, idx - context_size)
            end = min(len(tokens[i]) - 1, idx + context_size)
            swap_idx = np.random.randint(start, end + 1)
            if swap_idx != idx:
                noisy_tokens[i][idx], noisy_tokens[i][swap_idx] = noisy_tokens[i][swap_idx], noisy_tokens[i][idx]
    return noisy_tokens

def prepare_training_data_with_noise(max_length, vocab, file_path,noise_config=None):
    if noise_config is None:
        noise_config = {
            'gaussian': {'enabled': False, 'level': 0.1},
            'dropout': {'enabled': False, 'rate': 0.1},
            'swap': {'enabled': False, 'rate': 0.1},
            'mask': {'enabled': False, 'rate': 0.05}
        }
    monomer1_list, monomer2_list = process_dual_monomer_data(file_path)
    monomer1_list, monomer2_list = monomer1_list[:50], monomer2_list[:50]

    # Encode functional groups
    group_features = encode_functional_groups(monomer1_list, monomer2_list)
    
    # Tokenize SMILES
    tokens1 = tokenize_smiles(monomer1_list)
    tokens2 = tokenize_smiles(monomer2_list)

    # Pad tokens (add 1 to max_length for consistency)
    padded_tokens1 = pad_token(tokens1, max_length + 1, vocab)
    padded_tokens2 = pad_token(tokens2, max_length + 1, vocab)

    noisy_tokens1 = padded_tokens1.copy()
    noisy_tokens2 = padded_tokens2.copy()

    if noise_config['swap']['enabled']:
        print("Swap noise enabled")
        noisy_tokens1 = add_swap_noise(noisy_tokens1, 
                                     noise_config['swap']['rate'])
        noisy_tokens2 = add_swap_noise(noisy_tokens2, 
                                     noise_config['swap']['rate'])
    if noise_config['gaussian']['enabled']:
        print("Gaussian noise enabled")
        noisy_tokens1 = add_gaussian_noise(noisy_tokens1, 
                                         noise_config['gaussian']['level'])
        noisy_tokens2 = add_gaussian_noise(noisy_tokens2, 
                                         noise_config['gaussian']['level'])
        
    if noise_config['dropout']['enabled']:
        print("Dropout noise enabled")
        noisy_tokens1 = add_dropout_noise(noisy_tokens1, 
                                        noise_config['dropout']['rate'])
        noisy_tokens2 = add_dropout_noise(noisy_tokens2, 
                                        noise_config['dropout']['rate'])
        
    if noise_config['mask']['enabled']:
        print("Mask noise enabled")
        noisy_tokens1 = add_mask_noise(noisy_tokens1, vocab, 
                                     noise_config['mask']['rate'])
        noisy_tokens2 = add_mask_noise(noisy_tokens2, vocab, 
                                     noise_config['mask']['rate'])

    # Generate decoder inputs and outputs
    decoder_input1, decoder_output1 = make_target(noisy_tokens1)
    decoder_input2, decoder_output2 = make_target(noisy_tokens2)

    # Convert to numpy arrays using reduced data types
    padded_tokens1 = np.array(padded_tokens1, dtype=np.int32)  # Use int32 instead of float64
    padded_tokens2 = np.array(padded_tokens2, dtype=np.int32)
    group_features = np.array(group_features, dtype=np.float32)  # Use float32 instead of float64

    decoder_input1 = np.array(decoder_input1, dtype=np.int32)
    decoder_output1 = np.array(decoder_output1, dtype=np.int32)
    decoder_input2 = np.array(decoder_input2, dtype=np.int32)
    decoder_output2 = np.array(decoder_output2, dtype=np.int32)

    # Ensure group_features has correct shape
    if len(group_features.shape) > 2:
        group_features = group_features.reshape(group_features.shape[0], -1)

    # ✅ Use integer encoding instead of one-hot to reduce memory
    target1 = padded_tokens1[:, 1:] 
    target2 = padded_tokens2[:, 1:]

    # ✅ Integer encoding instead of one-hot encoding
    original_tokens1 = pad_token(tokens1, max_length, vocab)
    original_tokens2 = pad_token(tokens2, max_length, vocab)
    original_smiles1 = np.array(original_tokens1, dtype=np.int32)
    original_smiles2 = np.array(original_tokens2, dtype=np.int32)
    print(original_smiles1.shape)
    print(original_smiles2.shape)
    print("Example of original_smiles1: ", decode_smiles(original_smiles1[0]))
    print("Example of original_smiles2: ", decode_smiles(original_smiles2[0]))

    # Debugging info
    print("Input shapes:")
    print(f"monomer1_input shape: {padded_tokens1.shape}")
    print(f"monomer2_input shape: {padded_tokens2.shape}")
    print(f"group_input shape: {group_features.shape}")
    print(f"decoder_input1 shape: {decoder_input1.shape}")
    print(f"decoder_input2 shape: {decoder_input2.shape}")

    print("Target shapes:")
    print(f"target1 shape: {target1.shape}")
    print(f"target2 shape: {target2.shape}")
    print(f"decoder_output1 shape: {decoder_output1.shape}")
    print(f"decoder_output2 shape: {decoder_output2.shape}")

    # ✅ Return as dictionary
    inputs = {
        'monomer1_input': padded_tokens1[:, :-1],
        'monomer2_input': padded_tokens2[:, :-1],
        'group_input': group_features,
        'decoder_input1': decoder_input1[:, :-1],
        'decoder_input2': decoder_input2[:, :-1],
        'original_monomer1': original_smiles1,
        'original_monomer2': original_smiles2
    }
    
    outputs = {
        'monomer1_output': target1,
        'monomer2_output': target2
    }
    
    return inputs, outputs





if __name__ == "__main__":
    tokens1 = tokenize_smiles(["CBrNFC1OC2"])

    tokens=add_swap_noise2(tokens1,0.3)
    print(decode_smiles(tokens[0]))
    




