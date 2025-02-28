import json
import tensorflow as tf
from Data_Process_with_prevocab_gen import *
import random
import keras
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
import numpy as np
from CustomLoss import *
     

def sample_with_temperature(predictions, temperature=1.0):
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions + 1e-7) / temperature  # Add small value to avoid log(0)
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, predictions, 1)
    return np.argmax(probas)


def get_random_groups(min_groups=1, max_groups=3):
    available_groups = list(Constants.GROUP_VOCAB.keys())
    num_groups = random.randint(min_groups, min(max_groups, len(available_groups)))
    selected_groups = random.sample(available_groups, num_groups)
    return selected_groups

def get_random_groups2(input_smile):
    groups = extract_group_smarts2(input_smile)
    return groups



def predict_smiles(model, smiles_vocab, model_params, temperatures=[0.4, 0.6, 0.8]):
    smiles = read_smiles_from_file(Constants.TRAINING_FILE)
    results = []  # List to store all results
    
    for idx, input_smile in enumerate(smiles[:1000]):
        print(f"\n{'='*50}")
        print(f"Processing Input SMILES {idx+1}/1000: {input_smile}")
        print('='*50)
        
        # Store results for this input SMILES
        input_results = {
            'input_smiles': input_smile,
            'generations': []
        }
        
        #selected_groups = get_random_groups()
        selected_groups = get_random_groups2(input_smile)
        groups = [encode_groups(selected_groups, Constants.GROUP_VOCAB)]
        group_input = np.array(groups)
        
        for temp in temperatures:
            print(f"\nTrying Temperature: {temp}")
            print("-"*30)
            
            tokens = tokenize_smiles([input_smile],Constants.TOKENIZER_PATH)
            padded_tokens = pad_token(tokens, model_params["max_length"], smiles_vocab)
            input_seq = np.array(padded_tokens[0:1])
            
            decoder_seq = np.zeros((1, model_params["max_length"]))
            decoder_seq[0, 0] = smiles_vocab['<start>']
            
            # Generate tokens
            generated_tokens = []
            for i in range(model_params["max_length"]-1):
                output = model.predict([input_seq, group_input, decoder_seq], verbose=0)
                next_token_probs = output[0, i]
                next_token = sample_with_temperature(next_token_probs, temp)
                generated_tokens.append(next_token)
                if next_token == smiles_vocab['<end>']:
                    break
                if i < model_params["max_length"] - 2:
                    decoder_seq[0, i + 1] = next_token
            
            try:
                generated_smiles = decode_smiles(generated_tokens,Constants.TOKENIZER_PATH)
                mol = Chem.MolFromSmiles(generated_smiles)
                if mol is not None and generated_smiles != "":
                    # Check groups presence
                    present_groups = []
                    not_present_groups = []
                    for smarts in selected_groups:
                        if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts)):
                            present_groups.append(smarts)
                        else:
                            not_present_groups.append(smarts)
                    
                    # Store generation results
                    generation_result = {
                        'temperature': temp,
                        'required_groups': selected_groups,
                        'present_groups': present_groups,
                        'missing_groups': not_present_groups,
                        'generated_smiles': generated_smiles
                    }
                    input_results['generations'].append(generation_result)
                    
                    # Print results
                    print(f"Required Groups: {selected_groups}")
                    print(f"Present Groups: {present_groups}")
                    print(f"Not Present Groups: {not_present_groups}")
                    print(f"Generated SMILES: {generated_smiles}")
                    
            except Exception as e:
                print(f"Failed to generate valid SMILES at temp {temp}: {str(e)}")
                continue
        
        results.append(input_results)
    
    # Save results to file
    output_file = "generation_results_2000.json"
    with open(output_file, 'w') as f:
        json.dump({
            'generation_parameters': {
                'temperatures': temperatures,
                'num_input_smiles': len(smiles)
            },
            'results': results
        }, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Results saved to {output_file}")
    print(f"Total valid SMILES generated: {sum(len(r['generations']) for r in results)}")
    
    return results

def generate_new_smiles(model=None, smiles_vocab=None, model_params=None):
    predicted_smiles = predict_smiles(model, smiles_vocab, model_params)
    return predicted_smiles


