import Constants
import pandas as pd
import numpy as np
import tensorflow as tf
from rdkit import Chem
from Data_Process_with_prevocab import *

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

def prepare_training_data(max_length, vocab,file_path):
    monomer1_list, monomer2_list = process_dual_monomer_data(file_path)
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
    print(f"monomer1_input shape: {padded_tokens1[:, :-1].shape}")
    print(f"monomer2_input shape: {padded_tokens2[:, :-1].shape}")
    print(f"group_input shape: {group_features.shape}")
    print(f"decoder_input1 shape: {decoder_input1[:, :-1].shape}")
    print(f"decoder_input2 shape: {decoder_input2[:, :-1].shape}")
    
    # Create target data (shifted by one position)
    target1 = tf.keras.utils.to_categorical(padded_tokens1[:, 1:], num_classes=len(vocab))
    target2 = tf.keras.utils.to_categorical(padded_tokens2[:, 1:], num_classes=len(vocab))
    # target1 = padded_tokens1[:, 1:]  # Next token prediction
    # target2 = padded_tokens2[:, 1:] 
    # target1 = target1.reshape(target1.shape[0], target1.shape[1], 1)
    # target2 = target2.reshape(target2.shape[0], target2.shape[1], 1)
    # print(target1.shape)
    # print(target2.shape)
    
    print("Target shapes:")
    # print(f"target1 shape: {target1.shape}")
    # print(f"target2 shape: {target2.shape}")
    print(f"decoder_output1 shape: {decoder_output1[:, 1:].shape}")
    print(f"decoder_output2 shape: {decoder_output2[:, 1:].shape}")
    
    # Return properly formatted dictionaries
    inputs = {
        'monomer1_input': padded_tokens1[:, :-1],
        'monomer2_input': padded_tokens2[:, :-1],
        'group_input': group_features,
        'decoder_input1': decoder_input1[:, :-1],
        'decoder_input2': decoder_input2[:, :-1]
    }
    
    outputs = {
        'monomer1_output': target1,
        'monomer2_output': target2
    }
    
    return inputs, outputs
