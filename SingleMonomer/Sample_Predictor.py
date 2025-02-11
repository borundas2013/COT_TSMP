import json
import tensorflow as tf
from Data_Process_with_prevocab_gen import *
import random
import keras

def calculate_group_similarity(target_groups, generated_groups):
    """Calculate similarity between target and generated functional groups"""
    target_set = set(target_groups)
    generated_set = set(generated_groups)
    
    intersection = len(target_set.intersection(generated_set))
    union = len(target_set.union(generated_set))
    
    return intersection / max(union, 1)

@keras.saving.register_keras_serializable()
def custom_loss(y_true, y_pred, group_smarts_list, reverse_vocab):
    # Convert inputs to tensors and fix shapes
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)
    print(y_true.shape)
    print(y_pred.shape)
    
    if len(y_true.shape) == 3:
        y_true = tf.squeeze(y_true, axis=-1)
    
    y_true = tf.cast(y_true, tf.int32)
    
    # Calculate reconstruction loss
    reconstruction_loss = tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False)
    )

     # Helper function to decode predictions to SMILES
    def decode_predictions(pred_tensor):
        pred_indices = tf.argmax(pred_tensor, axis=-1)
        predicted_smiles=[]
        for tokens in pred_indices:
            smiles = decode_smiles(tokens,Constants.TOKENIZER_PATH)
            predicted_smiles.append(smiles)
       
        return predicted_smiles
    
    # Validity check function
    def check_validity(pred_tensor):
        smiles_list = decode_predictions(pred_tensor)
        validity_scores = []
        valid_smiles=[]
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None and smiles != "":
                    valid_smiles.append(smiles)
                    validity_scores.append(0.0)
                else:
                    validity_scores.append(1.0)
            except:
                validity_scores.append(1.0)
        return tf.constant(validity_scores, dtype=tf.float32), valid_smiles
    
    
    def check_groups(pred_tensor, group_smarts_list):
        validity_scores, smiles_list = check_validity(pred_tensor)
        group_scores = []
        if len(smiles_list) == 0:
            return tf.constant([1.0], dtype=tf.float32),validity_scores
        
        for smile in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smile)
                generated_groups = []
                for smarts in Constants.GROUP_VOCAB.keys():
                    if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts)):
                        generated_groups.append(smarts)
            
            # Calculate similarity score
                similarity = calculate_group_similarity(group_smarts_list, generated_groups)
                group_scores.append(1.0 - similarity)  # Convert to loss
            except:
                group_scores.append(1.0)
        return tf.constant(group_scores, dtype=tf.float32),validity_scores
    
    reconstruction_weight = 1.0
    validity_weight = 0.5
    group_weight = 0.5

    group_loss_score,validity_scores = check_groups(y_pred,Constants.GROUP_VOCAB.keys())

  
    validity_loss = tf.reduce_mean(validity_scores)
    group_loss = tf.reduce_mean(group_loss_score)
    
    print('\nReconstruction Loss: ',reconstruction_loss.numpy())
    print('Validity Loss: ',validity_loss.numpy())
    print('Group Loss: ',group_loss.numpy())
    total_loss = (reconstruction_weight * reconstruction_loss + 
                 validity_weight * validity_loss + 
                 group_weight * group_loss)
 
    
    # Return only reconstruction loss for now to ensure stable training
    return total_loss

@keras.saving.register_keras_serializable()
class CustomLoss(tf.keras.losses.Loss):
    def __init__(self, group_smarts_list, reverse_vocab, name='custom_loss', reduction='sum_over_batch_size'):
        super().__init__(name=name, reduction=reduction)
        self.group_smarts_list = group_smarts_list
        self.reverse_vocab = reverse_vocab
    
    def call(self, y_true, y_pred):
        return custom_loss(y_true, y_pred, self.group_smarts_list, self.reverse_vocab)
    
    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "group_smarts_list": list(self.group_smarts_list),  # Convert to list if it's not already
            "reverse_vocab": self.reverse_vocab
        }
    
    @classmethod
    def from_config(cls, config):
        return cls(**config) 
def load_and_retrain(save_dir="saved_model_prevocab_gru"):
    # Load the vocabulary
    with open(f"{save_dir}/smiles_vocab.json", "r") as f:
        smiles_vocab = json.load(f)
    
    # Load model parameters
    with open(f"{save_dir}/model_params.json", "r") as f:
        model_params = json.load(f)
    
    # Load the model with updated path
    model = tf.keras.models.load_model(f"{save_dir}/model.keras")
    
    return model, smiles_vocab, model_params


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

def fix_ring_closures(smiles):
    """Fix ring closure issues in a SMILES string"""
    try:
        # Find all digits (ring numbers) in the SMILES
        ring_numbers = [int(d) for d in smiles if d.isdigit()]
        
        # Count occurrences of each ring number
        for num in set(ring_numbers):
            count = ring_numbers.count(num)
            if count % 2 != 0:  # If number appears odd times
                # Remove the last occurrence of this number
                last_idx = smiles.rindex(str(num))
                smiles = smiles[:last_idx] + smiles[last_idx + 1:]
        
        # Validate the fixed SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return smiles
            # Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            # fixed_smiles = Chem.MolToSmiles(mol, kekuleSmiles=True)
            # if fixed_smiles is not None:
            #     return fixed_smiles
    except:
        pass
    return None

def generate_input_sequence(mode="random", model_params=None, smiles_vocab=None):

    max_length = model_params["max_length"]
    vocab_size = len(smiles_vocab)
    
    if mode == "random":
        # Random integers between 0 and vocab_size
        seq = np.random.randint(0, vocab_size, size=(1, max_length))
    
    elif mode == "zeros":
        # All zeros
        seq = np.zeros((1, max_length))
    
    elif mode == "ones":
        # All ones
        seq = np.ones((1, max_length))
    
    elif mode == "alternating":
        # Alternating 0s and 1s
        seq = np.array([[i % 2 for i in range(max_length)]])
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return seq.astype(np.int32)


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
        
        selected_groups = get_random_groups()
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

def generate_new_smiles(model, smiles_vocab, model_params):
    #model, smiles_vocab, model_params = load_and_retrain()
    predict_smiles(model, smiles_vocab, model_params)

#generate_new_smiles()

