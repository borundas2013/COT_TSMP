from Constants import *
from Data_Process_with_prevocab import *
from LoadPreTrainedModel import *
from pretrained_weights import *
from saveandload import *
from dual_smile_process import *
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs 

def validate_predictions(model, vocab, max_length, file_path,num_samples=5):
    """
    Validate model predictions by comparing input and generated SMILES
    """
    # Get validation data
    monomer1_list, monomer2_list = process_dual_monomer_data(file_path)
    group_features = encode_functional_groups(monomer1_list, monomer2_list)
    
    # Ensure group_features has correct shape (batch_size, num_groups)
    group_features = np.array(group_features)
    if len(group_features.shape) > 2:
        group_features = group_features.reshape(group_features.shape[0], -1)
    
    # Prepare input data
    tokens1 = tokenize_smiles(monomer1_list)
    tokens2 = tokenize_smiles(monomer2_list)
    padded_tokens1 = pad_token(tokens1, max_length + 1, vocab)
    padded_tokens2 = pad_token(tokens2, max_length + 1, vocab)
    
    # Convert to numpy arrays and ensure all have same first dimension
    padded_tokens1 = np.array(padded_tokens1)[:, :-1]
    padded_tokens2 = np.array(padded_tokens2)[:, :-1]
    decoder_input1 = np.array(padded_tokens1)
    decoder_input2 = np.array(padded_tokens2)
    
    # Print shapes for debugging
    print("Validation data shapes:")
    print(f"monomer1 shape: {padded_tokens1.shape}")
    print(f"monomer2 shape: {padded_tokens2.shape}")
    print(f"group shape: {group_features.shape}")
    print(f"decoder_input1 shape: {decoder_input1.shape}")
    print(f"decoder_input2 shape: {decoder_input2.shape}")
    
    # Get predictions
    predictions = model.predict({
        'monomer1_input': padded_tokens1,
        'monomer2_input': padded_tokens2,
        'group_input': group_features,
        'decoder_input1': decoder_input1,
        'decoder_input2': decoder_input2
    })
    
    
    # Convert indices to SMILES
    idx_to_token = {idx: token for token, idx in vocab.items()}
    

    def decode_smiles(pred_sequence):
        pred_indices = np.argmax(pred_sequence, axis=-1)
        tokenizer = PreTrainedTokenizerFast.from_pretrained("vocab/smiles_tokenizer")
        decoded = tokenizer.decode(pred_indices, skip_special_tokens=True).replace(" ","")
        smiles = ''.join(decoded)
        return smiles
    
    def is_valid_smiles(smiles):
        """Check if SMILES string is valid"""
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None

    def calculate_similarity(smi1, smi2):
        """Calculate Tanimoto similarity between two SMILES strings"""
        # Only calculate if both SMILES are valid
        if not (is_valid_smiles(smi1) and is_valid_smiles(smi2)):
            return None
            
        try:
            mol1 = Chem.MolFromSmiles(smi1)
            mol2 = Chem.MolFromSmiles(smi2)
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2)
            return DataStructs.TanimotoSimilarity(fp1, fp2)
        except:
            return None

    # Print comparisons
    print("\nValidation Results:")
    print("=" * 80)
    
    for i in range(min(num_samples, len(monomer1_list))):
        print(f"\nSample {i+1}:")
        print("-" * 40)
        
        # Monomer 1
        input_smiles1 = monomer1_list[i]
        pred_smiles1 = decode_smiles(predictions[0][i])
        print("Monomer 1:")
        print(f"Input SMILES:     {input_smiles1}")
        print(f"Predicted SMILES: {pred_smiles1}")
        valid1_input = is_valid_smiles(input_smiles1)
        valid1_pred = is_valid_smiles(pred_smiles1)
        print(f"Input Valid: {valid1_input}, Prediction Valid: {valid1_pred}")
        
        # Monomer 2
        input_smiles2 = monomer2_list[i]
        pred_smiles2 = decode_smiles(predictions[1][i])
        print("\nMonomer 2:")
        print(f"Input SMILES:     {input_smiles2}")
        print(f"Predicted SMILES: {pred_smiles2}")
        valid2_input = is_valid_smiles(input_smiles2)
        valid2_pred = is_valid_smiles(pred_smiles2)
        print(f"Input Valid: {valid2_input}, Prediction Valid: {valid2_pred}")
        
        # Calculate similarity only if both SMILES are valid
        print("\nSimilarity Scores:")
        if valid1_input and valid1_pred:
            sim1 = calculate_similarity(input_smiles1, pred_smiles1)
            print(f"Monomer 1 Similarity: {sim1:.3f}")
        else:
            print("Monomer 1 Similarity: Not calculated (invalid SMILES)")
            
        if valid2_input and valid2_pred:
            sim2 = calculate_similarity(input_smiles2, pred_smiles2)
            print(f"Monomer 2 Similarity: {sim2:.3f}")
        else:
            print("Monomer 2 Similarity: Not calculated (invalid SMILES)")
            
        print("=" * 80)
