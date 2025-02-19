from saveandload import load_model
from LoadPreTrainedModel import load_and_retrain
import os
from pathlib import Path
from Data_Process_with_prevocab import *
from dual_smile_process import *
import random
import json
from datetime import datetime
from rdkit import DataStructs
from rdkit.Chem import AllChem

def get_project_root():
    """Get the path to the project root directory"""
    current_file = Path(__file__).resolve()
    root_dir = current_file.parent
    return root_dir


def sample_with_temperature(predictions, temperature):
    if temperature == 0:
        return np.argmax(predictions)
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions + 1e-7) / temperature
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, predictions, 1)
    return np.argmax(probas)

def load_model_and_params():
    root_dir = get_project_root()
    save_dir_abs = os.path.join(root_dir,"saved_models_new")
    save_dir_abs = os.path.normpath(save_dir_abs)
    weights_path = os.path.join(root_dir, "saved_models_new_DLOSS_TWO_H","weights.weights.h5")
    params_path = os.path.join(root_dir, "saved_models_new_DLOSS_TWO_H","params.json")
    pretrained_model, smiles_vocab, model_params = load_and_retrain(save_dir=save_dir_abs)
    model,params = load_model(weights_path, params_path, pretrained_model)
    return model,params,smiles_vocab, model_params

def save_generation_results(smiles1, smiles2, input_smiles, temperature, group1, group2):
    """
    Save the generation results to a JSON file
    Args:
        smiles1 (str): Generated SMILES for monomer 1
        smiles2 (str): Generated SMILES for monomer 2
        input_smiles (list): List of input SMILES [monomer1, monomer2]
        temperature (float): Temperature used for generation
        group1 (str): First functional group
        group2 (str): Second functional group
    """
    # Check groups present in each monomer
    groups_in_monomer1 = {
        group1: group1 in smiles1,
        group2: group2 in smiles1
    }
    groups_in_monomer2 = {
        group1: group1 in smiles2,
        group2: group2 in smiles2
    }
    
    # Create result dictionary
    result = {
        "input_smiles": {
            "monomer1": input_smiles[0],
            "monomer2": input_smiles[1]
        },
        "temperature": temperature,
        "generated_smiles": {
            "monomer1": smiles1,
            "monomer2": smiles2
        },
        "groups_present": {
            "monomer1": groups_in_monomer1,
            "monomer2": groups_in_monomer2
        },
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save results to JSON file
    output_file = os.path.join(get_project_root(), "Two_monomers_Code","latest_1" "generated_results_new_H.json")
    
    # If file exists, load existing results and append new ones
    all_results = []
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            all_results = json.load(f)
    
    all_results.append(result)
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Generated valid SMILES pair at temperature {temperature}")
    print(f"Monomer 1: {smiles1}")
    print(f"Monomer 2: {smiles2}")
    print("Groups present:")
    print(f"Monomer 1: {groups_in_monomer1}")
    print(f"Monomer 2: {groups_in_monomer2}")
    print("\n")

def calculate_similarity(smiles1, smiles2):
    """
    Calculate Tanimoto similarity between two SMILES strings
    Args:
        smiles1: First SMILES string
        smiles2: Second SMILES string
    Returns:
        float: Tanimoto similarity score (0-1)
    """
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if mol1 is None or mol2 is None:
            return 1.0  # Return high similarity for invalid SMILES to filter them out
        
        # Generate Morgan fingerprints
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, 1024)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, 1024)
        
        # Calculate Tanimoto similarity
        similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
        return similarity
    except Exception as e:
        print(f"Error calculating similarity: {str(e)}")
        return 1.0  # Return high similarity for errors to filter them out

def filter_diverse_generations(generated_smiles1, generated_smiles2, input_smiles1, input_smiles2, threshold=0.6):
    """
    Filter generated SMILES pairs based on similarity to input SMILES
    Args:
        generated_smiles1: Generated SMILES for monomer 1
        generated_smiles2: Generated SMILES for monomer 2
        input_smiles1: Input SMILES for monomer 1
        input_smiles2: Input SMILES for monomer 2
        threshold: Maximum similarity threshold (default 0.5)
    Returns:
        bool: True if the generated pair is diverse enough (below threshold)
    """
    print("Generated SMILES: ", generated_smiles1 +" -> "+generated_smiles2+"\n")
    sim1 = calculate_similarity(generated_smiles1, input_smiles1)
    sim2 = calculate_similarity(generated_smiles2, input_smiles2)
    
    print(f"Similarity scores:")
    print(f"Monomer 1: {sim1:.3f}")
    print(f"Monomer 2: {sim2:.3f}")
    
    return sim1 < threshold or sim2 < threshold

def generate_data():
    try:
        # Initialize data loading
        root_dir = get_project_root()
        file_path = os.path.join(root_dir, 'Data', "smiles.xlsx")
        print(f"Processing data from: {file_path}")
        
        # Load monomer data
        monomer1_list, monomer2_list = process_dual_monomer_data(file_path)
        print(f"Loaded {len(monomer1_list)} monomer pairs")
        
        prediction_list = []
        max_attempts = 5  # Maximum attempts per temperature
        total_attempts = 0
        successful_generations = 0
        
        for idx, (monomer1, monomer2) in enumerate(zip(monomer1_list, monomer2_list)):
            print(f"\nProcessing monomer pair {idx + 1}/{len(monomer1_list)}")
            smiles_list = [monomer1, monomer2]
            print(f"Input SMILES: {smiles_list}")
            
            # Load model and parameters
            model, params, smiles_vocab, model_params = load_model_and_params()
            
            # Get desired groups
            desired_groups = [["CCS", "C=C"], ["C1OC1", "NC"],
                            ["C=C", "C=C(C=O)"], ["C=C", "C=C(C=O)"]]
            group1, group2 = random.choice(desired_groups)
            
            # Prepare input data
            max_length = model_params['max_length']
            tokens, tokens2 = tokenize_smiles([smiles_list[0]]), tokenize_smiles([smiles_list[1]])
            padded_tokens = pad_token(tokens, max_length, smiles_vocab)
            padded_tokens2 = pad_token(tokens2, max_length, smiles_vocab)
            input_seq = np.array(padded_tokens)
            input_seq2 = np.array(padded_tokens2)
            
            # Prepare group features
            group_features = encode_groups([group1, group2], Constants.GROUP_VOCAB)
            group_features = np.array([group_features])
            
            temperature_list = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
            
            for temperature in temperature_list:
                print(f"\nTrying temperature: {temperature}")
                attempts_this_temp = 0
                success = False
                
                while attempts_this_temp < max_attempts:
                    attempts_this_temp += 1
                    total_attempts += 1
                    print(f"Attempt {attempts_this_temp}/{max_attempts}")
                    
                    try:
                        # Initialize decoder sequences
                        decoder_seq = np.zeros((1, max_length))
                        decoder_seq[0, 0] = smiles_vocab['<start>']
                        decoder_seq2 = np.zeros((1, max_length))
                        decoder_seq2[0, 0] = smiles_vocab['<start>']
                        
                        # Add noise to decoder sequences
                        noise = np.random.normal(0, 0.1, decoder_seq.shape)
                        decoder_seq = decoder_seq + noise
                        decoder_seq2 = decoder_seq2 + noise
                        
                        tokens1 = []
                        tokens2 = []
                        is_monomer1_complete = False
                        is_monomer2_complete = False
                        assigned_groups_monomer1 = set()
                        assigned_groups_monomer2 = set()
                        
                        # Generate SMILES tokens
                        for i in range(max_length):
                            output = model.predict({
                                'monomer1_input': input_seq,
                                'monomer2_input': input_seq2,
                                'group_input': group_features,
                                'decoder_input1': decoder_seq,
                                'decoder_input2': decoder_seq2
                            }, verbose=0)
                            
                            # Get and modify probabilities
                            next_token_probs1 = output[0][0, i]
                            next_token_probs2 = output[1][0, i]
                            
                            # Check current partial SMILES
                            partial_smiles1 = decode_smiles(tokens1)
                            partial_smiles2 = decode_smiles(tokens2)
                            
                            # Update assigned groups
                            if group1 not in assigned_groups_monomer1 and group1 not in assigned_groups_monomer2:
                                if partial_smiles1 and group1 in partial_smiles1:
                                    assigned_groups_monomer1.add(group1)
                                elif partial_smiles2 and group1 in partial_smiles2:
                                    assigned_groups_monomer2.add(group1)
                            
                            if group2 not in assigned_groups_monomer1 and group2 not in assigned_groups_monomer2:
                                if partial_smiles1 and group2 in partial_smiles1:
                                    assigned_groups_monomer1.add(group2)
                                elif partial_smiles2 and group2 in partial_smiles2:
                                    assigned_groups_monomer2.add(group2)
                            
                            # Sample tokens
                            next_token = sample_with_temperature(next_token_probs1, temperature)
                            next_token2 = sample_with_temperature(next_token_probs2, temperature)
                            
                            # Process monomer 1
                            if not is_monomer1_complete:
                                tokens1.append(next_token)
                                if next_token == smiles_vocab['<end>']:
                                    is_monomer1_complete = True
                                elif i < max_length - 2:
                                    decoder_seq[0, i + 1] = next_token
                            
                            # Process monomer 2
                            if not is_monomer2_complete:
                                tokens2.append(next_token2)
                                if next_token2 == smiles_vocab['<end>']:
                                    is_monomer2_complete = True
                                elif i < max_length - 2:
                                    decoder_seq2[0, i + 1] = next_token2
                            
                            if is_monomer1_complete and is_monomer2_complete:
                                break
                        
                        # Decode and validate SMILES
                        smiles1 = decode_smiles(tokens1)
                        smiles2 = decode_smiles(tokens2)
                        mol1 = Chem.MolFromSmiles(smiles1)
                        mol2 = Chem.MolFromSmiles(smiles2)
                        
                        if mol1 is not None and mol2 is not None:
                            # Check if generated SMILES are diverse enough
                            if filter_diverse_generations(smiles1, smiles2, smiles_list[0], smiles_list[1]):
                                save_generation_results(smiles1, smiles2, smiles_list, temperature, group1, group2)
                                prediction_list.append([smiles1, smiles2])
                                successful_generations += 1
                                success = True
                                print(f"Success! Generated diverse SMILES pair #{successful_generations}")
                                print(f"Monomer 1: {smiles1}")
                                print(f"Monomer 2: {smiles2}")
                            else:
                                print("Generated SMILES too similar to input, trying again...")
                        else:
                            print("Invalid SMILES generated:")
                            if mol1 is None:
                                print(f"Invalid monomer 1: {smiles1}")
                            if mol2 is None:
                                print(f"Invalid monomer 2: {smiles2}")
                    
                    except Exception as e:
                        print(f"Error during generation: {str(e)}")
                        if attempts_this_temp == max_attempts:
                            print(f"Failed all {max_attempts} attempts at temperature {temperature}")
                        continue
                
                print(f"Temperature {temperature} complete. Success: {success}, Attempts: {attempts_this_temp}")
        
        print("\nGeneration Summary:")
        print(f"Total attempts: {total_attempts}")
        print(f"Successful generations: {successful_generations}")
        print(f"Success rate: {(successful_generations/total_attempts)*100:.2f}%")
        return prediction_list

    except Exception as e:
        print(f"Fatal error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    try:
        smiles_list = generate_data()
    except Exception as e:
        print(f"Error occurred: {str(e)}")





