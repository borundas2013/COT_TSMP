import Constants
import pandas as pd
import numpy as np
import tensorflow as tf
from rdkit import Chem
from Data_Process_with_prevocab import *

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

        
        # Combine groups from both monomers
        combined_groups = found_groups_m1 + found_groups_m2
        if not combined_groups:
            combined_groups.append('No group')
        
        all_groups.append(combined_groups)
    
    # Encode groups using the vocabulary
    encoded_groups = [encode_groups(groups, Constants.GROUP_VOCAB) for groups in all_groups]
    
    return encoded_groups

def prepare_training_data(max_length, vocab,file_path):
    monomer1_list, monomer2_list, er_list, tg_list = process_dual_monomer_data(file_path)
    monomer1_list, monomer2_list, er_list, tg_list = monomer1_list[:100], monomer2_list[:100], er_list[:100], tg_list[:100]
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
    er_list = np.array(er_list)
    tg_list = np.array(tg_list)
    
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
    print(f"er_list shape: {er_list.shape}")    
    print(f"tg_list shape: {tg_list.shape}")    
    # Create target data (shifted by one position)
    target1 = tf.keras.utils.to_categorical(padded_tokens1[:, 1:], num_classes=len(vocab))
    target2 = tf.keras.utils.to_categorical(padded_tokens2[:, 1:], num_classes=len(vocab))
   
    
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
        'decoder_input2': decoder_input2[:, :-1],
        'er_list': er_list,
        'tg_list': tg_list
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


