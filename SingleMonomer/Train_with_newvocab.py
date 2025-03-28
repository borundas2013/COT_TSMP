from Model import *
from Data_Process_with_prevocab_gen import *
import tensorflow as tf
import tensorflow.keras.backend as K
import random
import keras
import os
import json
from collections import defaultdict
from Sample_Predictor import *
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold




def calculate_group_similarity(target_groups, generated_groups):
    """Calculate similarity between target and generated functional groups"""
    target_set = set(target_groups)
    generated_set = set(generated_groups)
    
    intersection = len(target_set.intersection(generated_set))
    union = len(target_set.union(generated_set))
    
    return intersection / max(union, 1)

def check_exact_match(target_groups, generated_groups):
    """
    Check if target and generated functional groups exactly match
    Returns:
    - 1.0 if exact match
    - 0.0 if no match
    """
    target_set = set(target_groups)
    generated_set = set(generated_groups)
    
    return 1.0 if target_set == generated_set else 0.0
# Define custom loss function
@keras.saving.register_keras_serializable()
def custom_loss(y_true, y_pred):
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
        print(pred_tensor.shape)
        pred_indices = tf.argmax(pred_tensor, axis=-1)
        predicted_smiles=[]
        for tokens in pred_indices:
            smiles = decode_smiles(tokens,Constants.TOKENIZER_PATH)
            predicted_smiles.append(smiles)
       
        return predicted_smiles

    def calculate_tanimoto(smiles1, smiles2):
        """Calculate Tanimoto similarity between two SMILES strings"""
        try:
            mol1 = Chem.MolFromSmiles(smiles1)
            mol2 = Chem.MolFromSmiles(smiles2)
            if mol1 is None or mol2 is None:
                return 1.0
            
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
            
            return DataStructs.TanimotoSimilarity(fp1, fp2)
        except:
            return 1.0
        
    def diversity_loss(y_pred,input_smiles):
        smiles_list = decode_predictions(y_pred)
        tanimoto_scores = []
        for smiles in smiles_list:
            if smiles != "":
                tanimoto_scores.append(calculate_tanimoto(input_smiles,smiles))
            else:
                tanimoto_scores.append(1.0)
        return tf.constant(tanimoto_scores, dtype=tf.float32)
    
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
    def calculate_scaffold_entropy(y_pred):
        """Calculate molecular scaffold diversity using entropy"""
        smiles_list = decode_predictions(y_pred)
        scaffolds = defaultdict(int)
        total_mols = 0
        
        for smiles in smiles_list:
            if smiles != "":
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        # Get Murcko scaffold
                        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                        if scaffold is not None:
                            scaffold_smiles = Chem.MolToSmiles(scaffold, canonical=True)
                            scaffolds[scaffold_smiles] += 1
                            total_mols += 1
                except:
                    continue
    
        if total_mols == 0:
            return tf.constant(1.0, dtype=tf.float32)  # Maximum loss when no valid molecules
    
        # Calculate entropy of scaffold distribution
        probs = [count/total_mols for count in scaffolds.values()]
        
        # Add small epsilon to prevent log(0)
        epsilon = 1e-10
        entropy = tf.reduce_sum([-p * tf.math.log(p + epsilon) for p in probs])
        
        # Add epsilon to denominator to prevent division by zero
        max_possible_entropy = tf.math.log(tf.cast(len(scaffolds), tf.float32) + epsilon)
        
        # Ensure we don't divide by zero
        if max_possible_entropy > epsilon:
            normalized_entropy = entropy / max_possible_entropy
        else:
            normalized_entropy = tf.constant(0.0, dtype=tf.float32)
            
        # Ensure the result is between 0 and 1
        normalized_entropy = tf.clip_by_value(normalized_entropy, 0.0, 1.0)
        
        return 1.0 - normalized_entropy  # Convert entropy to loss: high entropy -> low loss

    def calculate_size_distribution(y_pred):
        """Calculate molecular size diversity using entropy"""
        smiles_list = decode_predictions(y_pred)
        sizes = defaultdict(int)
        total_mols = 0
        
        for smiles in smiles_list:
            if smiles != "":
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        # Get heavy atom count (exclude hydrogens)
                        size = mol.GetNumHeavyAtoms()
                        if size > 0:
                            sizes[size] += 1
                            total_mols += 1
                except:
                    continue
    
        if total_mols == 0:
            return tf.constant(1.0, dtype=tf.float32)  # Maximum loss when no valid molecules
    
        # Calculate entropy of size distribution
        probs = [count/total_mols for count in sizes.values()]
        
        # Add small epsilon to prevent log(0)
        epsilon = 1e-10
        entropy = tf.reduce_sum([-p * tf.math.log(p + epsilon) for p in probs])
        
        # Add epsilon to denominator to prevent division by zero
        max_possible_entropy = tf.math.log(tf.cast(len(sizes), tf.float32) + epsilon)
        
        # Ensure we don't divide by zero
        if max_possible_entropy > epsilon:
            normalized_entropy = entropy / max_possible_entropy
        else:
            normalized_entropy = tf.constant(0.0, dtype=tf.float32)
            
        # Ensure the result is between 0 and 1
        normalized_entropy = tf.clip_by_value(normalized_entropy, 0.0, 1.0)
        
        return 1.0 - normalized_entropy  # Convert entropy to loss: high entropy -> low loss

    def calculate_ring_diversity(y_pred):
        """Calculate ring system diversity using entropy"""
        smiles_list = decode_predictions(y_pred)
        ring_systems = defaultdict(int)
        total_mols = 0
        
        for smiles in smiles_list:
            if smiles != "":
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        # Get ring information
                        n_rings = Chem.rdMolDescriptors.CalcNumRings(mol)
                        ring_systems[n_rings] += 1
                        total_mols += 1
                except:
                    continue
    
        if total_mols == 0:
            return tf.constant(1.0, dtype=tf.float32)  # Maximum loss when no valid molecules
    
        # Calculate entropy of ring distribution
        probs = [count/total_mols for count in ring_systems.values()]
        
        # Add small epsilon to prevent log(0)
        epsilon = 1e-10
        entropy = tf.reduce_sum([-p * tf.math.log(p + epsilon) for p in probs])
        
        # Add epsilon to denominator to prevent division by zero
        max_possible_entropy = tf.math.log(tf.cast(len(ring_systems), tf.float32) + epsilon)
        
        # Ensure we don't divide by zero
        if max_possible_entropy > epsilon:
            normalized_entropy = entropy / max_possible_entropy
        else:
            normalized_entropy = tf.constant(0.0, dtype=tf.float32)
            
        # Ensure the result is between 0 and 1
        normalized_entropy = tf.clip_by_value(normalized_entropy, 0.0, 1.0)
        
        return 1.0 - normalized_entropy  # Convert entropy to loss: high entropy -> low loss
    
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

        # Write valid SMILES and their groups to file
        with open(Constants.GENERATED_TRAINING_SMILES_DIR_1, 'a') as f:
            for smile in smiles_list:
                try:
                    mol = Chem.MolFromSmiles(smile)
                    if mol is not None:
                        f.write(f"\nSMILES: {smile}\n")
                        f.write("Groups: ")
                        for smarts in Constants.GROUP_VOCAB.keys():
                            if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts)):
                                f.write(f"{smarts}, ")
                        f.write("\n")
                except:
                    continue
        return tf.constant(group_scores, dtype=tf.float32),validity_scores
    
    # reconstruction_weight = 1.0
    # validity_weight = 0.6
    # group_weight = 0.8
    # diversity_weight = 0.8
    reconstruction_weight = 1.0
    validity_weight = 0.6
    group_weight = 0.4
    scaffold_weight = 0.7
    size_weight = 0.5
    ring_weight = 0.5

    group_loss_score,validity_scores = check_groups(y_pred,Constants.GROUP_VOCAB.keys())

  
    validity_loss = tf.reduce_mean(validity_scores)
    group_loss = tf.reduce_mean(group_loss_score)
    diversity_loss = tf.reduce_mean(diversity_loss(y_pred,y_true))

    scaffold_loss = calculate_scaffold_entropy(y_pred)
    size_loss = calculate_size_distribution(y_pred)
    ring_loss = calculate_ring_diversity(y_pred)

    print('\nLoss Components:')
    print(f'Reconstruction Loss: {reconstruction_loss.numpy():.4f}')
    print(f'Validity Loss: {validity_loss.numpy():.4f}')
    print(f'Group Loss: {group_loss.numpy():.4f}')
    print(f'Scaffold Diversity Loss: {scaffold_loss:.4f}')
    print(f'Size Diversity Loss: {size_loss:.4f}')
    print(f'Ring Diversity Loss: {ring_loss:.4f}')
    # total_loss = (reconstruction_weight * reconstruction_loss + 
    #              validity_weight * validity_loss + 
    #              group_weight * group_loss +
    #              diversity_weight * diversity_loss)
    total_loss = (reconstruction_weight * reconstruction_loss +
                 validity_weight * validity_loss +
                 group_weight * group_loss +
                 scaffold_weight * scaffold_loss +
                 size_weight * size_loss +
                 ring_weight * ring_loss)
 
    
    # Return only reconstruction loss for now to ensure stable training
    return total_loss

@keras.saving.register_keras_serializable()
class CustomLoss(tf.keras.losses.Loss):
    def __init__(self, group_smarts_list, reverse_vocab, name='custom_loss', reduction='sum_over_batch_size'):
        super().__init__(name=name, reduction=reduction)
        self.group_smarts_list = group_smarts_list
        self.reverse_vocab = reverse_vocab
    
    def call(self, y_true, y_pred):
        return custom_loss(y_true, y_pred)
    
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

smiles_list = read_smiles_from_file(Constants.TRAINING_FILE)
smiles_list = smiles_list[:128]
x_smiles, x_groups, decoder_input, y, vocab_size, max_length,smiles_vocab =\
      make_training_data(smiles_list, Constants.VOCAB_PATH, Constants.TOKENIZER_PATH)
model = build_gru_model2(max_length, vocab_size, group_size=Constants.GROUP_SIZE)
X_train = [x_smiles, x_groups, decoder_input]
y_train = y
print(X_train[0].shape, X_train[1].shape, X_train[2].shape, y_train.shape)
print("Vocab Size: ",vocab_size)
print("Max Length: ",max_length)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001,
        clipnorm=1.0
    ),
    loss=CustomLoss(Constants.GROUP_VOCAB.keys(), smiles_vocab),
    run_eagerly=True
)
# early_stopping = tf.keras.callbacks.EarlyStopping(
#     monitor='val_loss',
#     patience=5,
#     restore_best_weights=True
# )

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=0.0001
    ),
    tf.keras.callbacks.ModelCheckpoint(
        f"{Constants.MODEL_SAVED_DIR_NEW_MODEL}/model_checkpoint.keras",
        monitor='val_loss',
        save_best_only=True
    )
]

model.fit(X_train, y_train, epochs=Constants.EPOCHS, 
          batch_size=Constants.BATCH_SIZE,validation_split=0.2,
          callbacks=callbacks,shuffle=True)

# After model training, save the model and vocabulary
def save_model_and_vocab(model, smiles_vocab, save_dir=Constants.MODEL_SAVED_DIR_NEW_MODEL):
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save the model with .keras extension
    model.save(f"{save_dir}/model.keras")
    
    # Save the vocabulary
    with open(f"{save_dir}/smiles_vocab.json", "w") as f:
        json.dump(smiles_vocab, f)
    
    # Save model parameters
    model_params = {
        "max_length": max_length,
        "vocab_size": vocab_size,
        "embedding_dim": Constants.EMBEDDING_DIM,
        "latent_dim": Constants.LATENT_DIM,
        "group_size": Constants.GROUP_SIZE
    }
    with open(f"{save_dir}/model_params.json", "w") as f:
        json.dump(model_params, f)

# # Function to load and retrain the model
def load_and_retrain(save_dir=Constants.MODEL_SAVED_DIR_NEW_MODEL):
    # Load the vocabulary
    with open(f"{save_dir}/smiles_vocab.json", "r") as f:
        smiles_vocab = json.load(f)
    
    # Load model parameters
    with open(f"{save_dir}/model_params.json", "r") as f:
        model_params = json.load(f)
    
    # Load the model with updated path
    model = tf.keras.models.load_model(
        f"{save_dir}/model.keras",
        custom_objects={
            "custom_loss": CustomLoss(Constants.GROUP_VOCAB.keys(), smiles_vocab)
        }
    )
    

    
    return model,smiles_vocab,model_params

# # Add these lines after your original training
save_model_and_vocab(model, smiles_vocab)
model, smiles_vocab, model_params = load_and_retrain()
generate_new_smiles(model, smiles_vocab, model_params)
