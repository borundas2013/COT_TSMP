import Constants
import pandas as pd
import numpy as np
import tensorflow as tf
from rdkit import Chem
from Data_Process_with_prevocab import *
import random

def process_dual_monomer_data(excel_path):

    try:
        # Read Excel file
        df = pd.read_excel(excel_path)
        
        # Check if required columns exist
        required_cols = ['SMILES', 'Er', 'Tg']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in Excel file")
        
        # Initialize lists for storing data
        smiles1_list = []
        smiles2_list = []
        er_list = []
        tg_list = []
        
        # Process each row
        for _, row in df.iterrows():
            try:
                # Extract the two SMILES from the SMILES column
                smiles_pair = eval(row['SMILES'])  # Safely evaluate string representation of list
                if len(smiles_pair) == 2:
                    smiles1, smiles2 = smiles_pair[0], smiles_pair[1]
                    smiles1_list.append(smiles1)
                    smiles2_list.append(smiles2)
                    er_list.append(row['Er'])
                    tg_list.append(row['Tg'])
            except:
                print(f"Skipping malformed SMILES pair: {row['SMILES']}")
                continue
                
        return smiles1_list, smiles2_list, er_list, tg_list
        
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
        if not combined_groups:
            combined_groups.append('No group')
        
        all_groups.append(combined_groups)
    
    # Encode groups using the vocabulary
    encoded_groups = [encode_groups(groups, Constants.GROUP_VOCAB) for groups in all_groups]
    
    return encoded_groups

def add_gaussian_noise(tokens, noise_level=1):
    """Add Gaussian noise to token embeddings"""
    noise = np.random.normal(0, noise_level, tokens.shape)
    noisy_tokens = tokens + noise
    return noisy_tokens

def add_dropout_noise(tokens, dropout_rate=0.1):
    """Randomly zero out some tokens"""
    mask = np.random.binomial(1, 1-dropout_rate, tokens.shape)
    return tokens * mask

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

def add_mask_noise(tokens, vocab, mask_rate=0.1):
    """Randomly mask tokens with MASK token"""
    mask_token = vocab.get('[MASK]', len(vocab)-1)  # Use last token if no mask token
    noisy_tokens = tokens.copy()
    mask = np.random.random(tokens.shape) < mask_rate
    noisy_tokens[mask] = mask_token
    return noisy_tokens

def reaction_valid_samples(smiles1,smiles2,er_list,tg_list):
 
    valid_reaction = []
    invalid_reaction = []
    for i in range(len(smiles1)):
        reaction_valid = filter_valid_groups(smiles1[i], smiles2[i])
        if reaction_valid:
            valid_reaction.append([smiles1[i],smiles2[i],er_list[i],tg_list[i]])
        else:
            invalid_reaction.append([smiles1[i],smiles2[i],er_list[i],tg_list[i]])

    print(len(valid_reaction))
    print(len(invalid_reaction))
    random_invalid_reaction = random.sample(invalid_reaction, 259)
    valid_reaction.extend(random_invalid_reaction) 
    print(len(valid_reaction))
    return valid_reaction

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

def prepare_training_data(max_length, vocab,file_path,noise_config=None):

    # Default noise configuration
    if noise_config is None:
        noise_config = {
            'gaussian': {'enabled': False, 'level': 0.1},
            'dropout': {'enabled': False, 'rate': 0.1},
            'swap': {'enabled': False, 'rate': 0.1},
            'mask': {'enabled': False, 'rate': 0.05}
        }
    monomer1_list, monomer2_list, er_list, tg_list = process_dual_monomer_data(file_path)
    valid_reaction = reaction_valid_samples(monomer1_list,monomer2_list,er_list,tg_list)
    monomer1_list = [i[0] for i in valid_reaction]
    monomer2_list = [i[1] for i in valid_reaction]
    er_list = [i[2] for i in valid_reaction]
    tg_list = [i[3] for i in valid_reaction]
    monomer1_list, monomer2_list, er_list, tg_list = monomer1_list[:2], monomer2_list[:2],er_list[:2], tg_list[:2]
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



    # Apply noise layers sequentially
    noisy_tokens1 = padded_tokens1.copy()
    noisy_tokens2 = padded_tokens2.copy()
    
    
        
    if noise_config['swap']['enabled']:
        print("Swap noise enabled")
        noisy_tokens1 = add_swap_noise(noisy_tokens1, 
                                     noise_config['swap']['rate'])
        noisy_tokens2 = add_swap_noise(noisy_tokens2, 
                                     noise_config['swap']['rate'])
        
    if noise_config['mask']['enabled']:
        print("Mask noise enabled")
        noisy_tokens1 = add_mask_noise(noisy_tokens1, vocab, 
                                     noise_config['mask']['rate'])
        noisy_tokens2 = add_mask_noise(noisy_tokens2, vocab, 
                                     noise_config['mask']['rate'])
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

    decoder_input1 = np.array(decoder_input1)
    decoder_output1 = np.array(decoder_output1)
    decoder_input2 = np.array(decoder_input2)
    decoder_output2 = np.array(decoder_output2)
    er_list = np.array(er_list)
    tg_list = np.array(tg_list)
    
    # Ensure group_features has the correct shape (batch_size, num_groups)
    if len(group_features.shape) > 2:
        group_features = group_features.reshape(group_features.shape[0], -1)
    
    # Print shapes for debugging
    print("Input shapes:")
    print(f"monomer1_input shape: {padded_tokens1[:, :].shape}")
    print(f"monomer2_input shape: {padded_tokens2[:, :].shape}")
    print(f"group_input shape: {group_features.shape}")
    print(f"decoder_input1 shape: {decoder_input1[:, :].shape}")
    print(f"decoder_input2 shape: {decoder_input2[:, :].shape}")
    print(f"er_list shape: {er_list.shape}")    
    print(f"tg_list shape: {tg_list.shape}")    
    # Create target data (shifted by one position)
    target1 = tf.keras.utils.to_categorical(padded_tokens1[:, 1:], num_classes=len(vocab))
    target2 = tf.keras.utils.to_categorical(padded_tokens2[:, 1:], num_classes=len(vocab))


    orginal_tokens1 = pad_token(tokens1, max_length , vocab)
    orginal_tokens2 = pad_token(tokens2, max_length , vocab)
    originial_smiles1 = tf.keras.utils.to_categorical(orginal_tokens1, num_classes=len(vocab))
    originial_smiles2 = tf.keras.utils.to_categorical(orginal_tokens2, num_classes=len(vocab))
   
    
    print("Target shapes:")
    # print(f"target1 shape: {target1.shape}")
    # print(f"target2 shape: {target2.shape}")
    print(f"decoder_output1 shape: {decoder_output1[:, :].shape}")
    print(f"decoder_output2 shape: {decoder_output2[:, :].shape}")
    
    # Return properly formatted dictionaries
    inputs = {
        'monomer1_input': padded_tokens1[:, :-1],
        'monomer2_input': padded_tokens2[:, :-1],
        'group_input': group_features,
        'decoder_input1': decoder_input1[:, :-1],
        'decoder_input2': decoder_input2[:, :-1],
        'er_list': er_list,
        'tg_list': tg_list,
        'original_monomer1': originial_smiles1,
        'original_monomer2': originial_smiles2
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


