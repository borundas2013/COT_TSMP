
import sys
import os

# Add the parent folder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from rdkit import Chem
import tensorflow as tf
from Data_Process_with_prevocab import *
import json
import os
from LoadPreTrainedModel import *
from Two_Monomer_Predictor import *


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

def encode_functional_groups_count(monomer1_list, monomer2_list):
    # SMARTS patterns for different functional groups
    epoxy_smarts = "[OX2]1[CX3][CX3]1"    # Epoxy group
    imine_smarts = "[NX2]=[CX3]"          # Imine group
    vinyl_smarts = "C=C"                   # Vinyl group
    thiol_smarts = "CCS"                   # Thiol group
    acryl_smarts = "C=C(C=O)"             # Acrylic group
    
    features = []
    for m1, m2 in zip(monomer1_list, monomer2_list):
        # Count functional groups for both monomers
        epoxy_count_m1 = count_functional_groups(m1, epoxy_smarts)
        imine_count_m1 = count_functional_groups(m1, imine_smarts)
        vinyl_count_m1 = count_functional_groups(m1, vinyl_smarts)
        thiol_count_m1 = count_functional_groups(m1, thiol_smarts)
        acryl_count_m1 = count_functional_groups(m1, acryl_smarts)
        
        epoxy_count_m2 = count_functional_groups(m2, epoxy_smarts)
        imine_count_m2 = count_functional_groups(m2, imine_smarts)
        vinyl_count_m2 = count_functional_groups(m2, vinyl_smarts)
        thiol_count_m2 = count_functional_groups(m2, thiol_smarts)
        acryl_count_m2 = count_functional_groups(m2, acryl_smarts)
        
        # Check valid combinations
        # 1. Epoxy-Imine combination
        if (epoxy_count_m1 >= 2 and imine_count_m2 >= 2):
            features.append([epoxy_count_m1, imine_count_m2])
        elif (epoxy_count_m2 >= 2 and imine_count_m1 >= 2):
            features.append([epoxy_count_m2, imine_count_m1])
        
        # 2. Vinyl-Thiol combination
        elif (vinyl_count_m1 >= 2 and thiol_count_m2 >= 2):
            features.append([vinyl_count_m1, thiol_count_m2])
        elif (vinyl_count_m2 >= 2 and thiol_count_m1 >= 2):
            features.append([vinyl_count_m2, thiol_count_m1])
            
        # 3. Vinyl-Vinyl combination
        elif (vinyl_count_m1 >= 2 and vinyl_count_m2 >= 2):
            features.append([vinyl_count_m1, vinyl_count_m2])
            
        # 4. Vinyl-Acrylic combination
        elif (vinyl_count_m1 >= 2 and acryl_count_m2 >= 2):
            features.append([vinyl_count_m1, acryl_count_m2])
        elif (vinyl_count_m2 >= 2 and acryl_count_m1 >= 2):
            features.append([vinyl_count_m2, acryl_count_m1])
        else:
            features.append([0, 0])
    # Count valid combinations (excluding zeros)
    valid_combinations = sum(1 for feature in features if feature != [0, 0])
    print(f"\nNumber of valid group combinations found: {valid_combinations}")
    return features
    

def calculate_group_presence(smiles_sequence, group_smarts):
    try:
        mol = Chem.MolFromSmiles(smiles_sequence)
        if mol is None:
            return 0
        count = len(mol.GetSubstructMatches(Chem.MolFromSmarts(group_smarts)))
        return 1 if count >= 2 else 0
    except:
        return 0


def get_training_data(monomer1_list, monomer2_list, max_length, vocab):
     valid_pairs = []
     for m1, m2 in zip(monomer1_list, monomer2_list):
         valid_pairs.append((str(m1), str(m2)))
     tokens1= tokenize_smiles(monomer1_list)
     tokens2= tokenize_smiles(monomer2_list)
     padded_tokens1= pad_token(tokens1, max_length, vocab)
     encoder_input = np.array(padded_tokens1)
     padded_tokens2= pad_token(tokens2, max_length, vocab)
     decoder_input = np.array([np.concatenate(([0], seq[:-1])) for seq in padded_tokens2])
     decoder_target = np.expand_dims(padded_tokens2, axis=-1)
    #  decoder_input = np.array(padded_tokens2[:, :-1]) 
    #  decoder_target = np.array(padded_tokens2[:, 1:]) 

     group_features = encode_functional_groups(monomer1_list, monomer2_list)

     encoder_input = np.array(encoder_input)
     decoder_input = np.array(decoder_input)
     decoder_target = np.array(decoder_target)
     group_features = np.array(group_features)

     processed_features = {
        'encoder_input': encoder_input,
        'decoder_input': decoder_input,
        'decoder_target': decoder_target,
        'group_features': group_features,
        'valid_pairs': valid_pairs,
        'valid_monomer1': monomer1_list,
        'valid_monomer2': monomer2_list
    }

     return processed_features
def select_training_data(processed_features):
    return {
        'input_data': {
            'smiles_input': processed_features['encoder_input'],
            'decoder_input': processed_features['decoder_input'],
            'group_input': processed_features['group_features']
        },
        'target_data': processed_features['decoder_target']
    }
     
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
    

def save_model_info(model, smiles_vocab, model_params, save_dir="saved_dual_monomer_model"):
    """
    Save model, vocabulary and parameters
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Save the model
        model.save(f"{save_dir}/model.keras")
        
        # Save SMILES vocabulary
        with open(f"{save_dir}/smiles_vocab.json", "w") as f:
            json.dump(smiles_vocab, f, indent=2)
        
        # Save model parameters
        with open(f"{save_dir}/model_params.json", "w") as f:
            json.dump(model_params, f, indent=2)
            
        print(f"Model information saved successfully to {save_dir}")
        
    except Exception as e:
        print(f"Error saving model information: {str(e)}")
        raise

def load_dual_monomer_model(save_dir="saved_dual_monomer_model"):
    """
    Load saved model, vocabulary and parameters
    """
    try:
        # Load model
        model = tf.keras.models.load_model(
            f"{save_dir}/model.keras",
            custom_objects={'CustomMonomerLoss': CustomLoss}
        )
        
        # Load SMILES vocabulary
        with open(f"{save_dir}/smiles_vocab.json", "r") as f:
            smiles_vocab = json.load(f)
            
        # Load model parameters
        with open(f"{save_dir}/model_params.json", "r") as f:
            model_params = json.load(f)
            
        print(f"Model information loaded successfully from {save_dir}")
        
        return model, smiles_vocab, model_params
        
    except Exception as e:
        print(f"Error loading model information: {str(e)}")
        raise

if __name__ == "__main__":
    model, smiles_vocab, model_params = load_and_retrain()
    monomer1_list, monomer2_list = process_dual_monomer_data("Small_Data/smiles.xlsx")
    features = encode_functional_groups(monomer1_list, monomer2_list)
    processed_features = get_training_data(monomer1_list, monomer2_list,
                                           max_length=model_params['max_length'], vocab=smiles_vocab)
    train_data = select_training_data(processed_features)

    
    print("\nInput and Target Data Shapes:")
    print(f"Encoder Input Shape: {train_data['input_data']['smiles_input'].shape}")
    print(f"Decoder Input Shape: {train_data['input_data']['decoder_input'].shape}")
    print(f"Group Features Shape: {train_data['input_data']['group_input'].shape}")
    print(f"Target Data Shape: {train_data['target_data'].shape}")
    model.fit(train_data['input_data'], train_data['target_data'], epochs=100)

    
    # Save the model information
    save_model_info(
        model=model,
        smiles_vocab=smiles_vocab,
        model_params=model_params,
        save_dir="saved_dual_monomer_model"
    )
    
    loaded_model, loaded_vocab, loaded_params = load_dual_monomer_model(
        save_dir="saved_dual_monomer_model"
    )
    
    # Generate multiple pairs
    generated_pairs = generate_multiple_pairs(
        model=loaded_model,
        smiles_vocab=loaded_vocab,
        model_params=loaded_params,
        input_file="path/to/your/input_file.xlsx",
        num_input_pairs=5,    # Number of input pairs to randomly select
        samples_per_pair=3    # Number of new samples per input pair
    )
    
    # Print results
    print("\nAll Generated Pairs:")
    for i, pair in enumerate(generated_pairs, 1):
        print(f"\nPair {i}:")
        print(f"Input Monomer 1: {pair['input_monomer1']}")
        print(f"Generated Monomer 1 (with {pair['groups1']}): {pair['generated_monomer1']}")
        print(f"Input Monomer 2: {pair['input_monomer2']}")
        print(f"Generated Monomer 2 (with {pair['groups2']}): {pair['generated_monomer2']}")
    
    # Optionally save results to file
    results_df = pd.DataFrame({
        'Input_Monomer1': [p['input_monomer1'] for p in generated_pairs],
        'Generated_Monomer1': [p['generated_monomer1'] for p in generated_pairs],
        'Groups1': [p['groups1'] for p in generated_pairs],
        'Input_Monomer2': [p['input_monomer2'] for p in generated_pairs],
        'Generated_Monomer2': [p['generated_monomer2'] for p in generated_pairs],
        'Groups2': [p['groups2'] for p in generated_pairs]
    })
    
    results_df.to_excel('generated_monomer_pairs.xlsx', index=False)

