import tensorflow as tf
from rdkit import Chem
from Data_Process_with_prevocab import *
from LoadPreTrainedModel import *
from datetime import datetime
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

@keras.saving.register_keras_serializable()
class CombinedLoss(tf.keras.losses.Loss):
    def __init__(self, weights=[1.0, 0.8, 0.8,0.8],decoder_name=None, **kwargs):
        name = f"{decoder_name}_combined_loss" if decoder_name else "combined_loss"
        super().__init__(name=name, **kwargs)
        self.weights = weights
        self.decoder_name = decoder_name
        
    def call(self, y_true, y_pred):
        # Reconstruction loss
        recon_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False)
        recon_loss = tf.reduce_mean(recon_loss)


        def check_rdkit_valid(inputs):
            pred_logits, y_true = inputs
            
            # Get predicted SMILES
            pred_tokens = tf.argmax(pred_logits, axis=-1)
            true_tokens = tf.argmax(y_true, axis=-1)
            
            tokenizer = PreTrainedTokenizerFast.from_pretrained(Constants.TOKENIZER_PATH)
            
            try:
                # Decode both predicted and true SMILES
                pred_smiles = tokenizer.decode(pred_tokens.numpy(), skip_special_tokens=True)
                true_smiles = tokenizer.decode(true_tokens.numpy(), skip_special_tokens=True)
                
                # Get molecules and canonical SMILES
                pred_mol = Chem.MolFromSmiles(pred_smiles)
                true_mol = Chem.MolFromSmiles(true_smiles)
                
                # Save to file if valid
                if pred_mol is not None or true_mol is not None:
                    valid_pairs = {
                        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                        "true": {
                            "original_smiles": true_smiles,
                            "canonical_smiles": Chem.MolToSmiles(true_mol, canonical=True) if true_mol else None,
                            "valid": true_mol is not None
                        },
                        "predicted": {
                            "original_smiles": pred_smiles,
                            "canonical_smiles": Chem.MolToSmiles(pred_mol, canonical=True) if pred_mol else None,
                            "valid": pred_mol is not None
                        }
                    }
                    
                    # Append to JSON file
                    output_file = "Code/Two_monomers_Code/valid_pairs_during_training.json"
                    try:
                        with open(output_file, 'r') as f:
                            all_pairs = json.load(f)
                    except (FileNotFoundError, json.JSONDecodeError):
                        all_pairs = []
                    
                    all_pairs.append(valid_pairs)
                    with open(output_file, 'w') as f:
                        json.dump(all_pairs, f, indent=4)
                
                return tf.constant(1.0) if pred_mol is not None else tf.constant(0.0)
            except:
                return tf.constant(0.0)
        
        # Prepare inputs for map_fn
        inputs = (y_pred, y_true)
        
        # Map over batched inputs
        validity_scores = tf.map_fn(
            check_rdkit_valid,
            inputs,
            fn_output_signature=tf.float32
        )
        valid_loss = 1.0 - tf.reduce_mean(validity_scores)


        
        # Group presence loss
        def check_groups(pred_logits, y_true):
            # Get predicted SMILES
            pred_tokens = tf.argmax(pred_logits, axis=-1)
            # Get true SMILES tokens
            true_tokens = tf.argmax(y_true, axis=-1)
            
            tokenizer = PreTrainedTokenizerFast.from_pretrained(Constants.TOKENIZER_PATH)
            
            try:
                # Decode both predicted and true SMILES
                pred_smiles = tokenizer.decode(pred_tokens.numpy(), skip_special_tokens=True)
                true_smiles = tokenizer.decode(true_tokens.numpy(), skip_special_tokens=True)
                
                # Get molecules for both
                pred_mol = Chem.MolFromSmiles(pred_smiles)
                true_mol = Chem.MolFromSmiles(true_smiles)
                
                if pred_mol is None or true_mol is None:
                    return tf.constant(0.0)
                
                group_patterns = {
                    "C=C": "C=C",
                    "NC": "[NX2]=[CX3]",
                    "C1OC1": "[OX2]1[CX3][CX3]1",
                    "CCS": "CCS",
                    "C=C(C=O)": "C=C(C=O)"
                }
                
                # Find groups in predicted SMILES
                pred_groups = []
                for name, pattern in group_patterns.items():
                    if pred_mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                        pred_groups.append(name)
                        
                # Find groups in true SMILES
                true_groups = []
                for name, pattern in group_patterns.items():
                    if true_mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                        true_groups.append(name)
                
                # Calculate similarity between predicted and true groups
                common_groups = set(pred_groups).intersection(set(true_groups))
                all_groups = set(pred_groups).union(set(true_groups))
                
                similarity = len(common_groups) / max(len(all_groups), 1)
                return tf.constant(similarity)
                
            except:
                return tf.constant(0.0)
        def check_groups_wrapper(inputs):
            pred_logits, y_true = inputs
            return check_groups(pred_logits, y_true)
        inputs = (y_pred, y_true)
        
        group_scores = tf.map_fn(
            check_groups_wrapper,
            inputs,
            fn_output_signature=tf.float32
        )
        group_loss = 1.0 - tf.reduce_mean(group_scores)


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
        
        def diversity_loss(inputs):
            y_pred, y_true = inputs
            # Convert tensors to predicted tokens and true tokens first
            pred_tokens = tf.argmax(y_pred, axis=-1)
            true_tokens = tf.argmax(y_true, axis=-1)
            
            tokenizer = PreTrainedTokenizerFast.from_pretrained(Constants.TOKENIZER_PATH)
            try:
                # Decode tokens to SMILES
                pred_smiles = tokenizer.decode(pred_tokens.numpy(), skip_special_tokens=True)
                true_smiles = tokenizer.decode(true_tokens.numpy(), skip_special_tokens=True)
                
                # Calculate Tanimoto similarity
                similarity = calculate_tanimoto(pred_smiles, true_smiles)
                return tf.constant(similarity, dtype=tf.float32)
            except:
                return tf.constant(1.0, dtype=tf.float32)

        # Map over batched inputs
        inputs = (y_pred, y_true)
        dl_loss = tf.map_fn(
            diversity_loss,
            inputs,
            fn_output_signature=tf.float32
        )
        dl_loss = tf.reduce_mean(dl_loss)
        # Combine losses
        total_loss = (self.weights[0] * recon_loss + 
                     self.weights[1] * valid_loss + 
                     self.weights[2] * group_loss + 
                     self.weights[3] * dl_loss )
  
        
        return total_loss
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "desired_groups": self.desired_groups,
            "weights": self.weights,
            'decoder_name': self.decoder_name
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

# def create_weighted_losses():
#     """Create weighted loss functions with RDKit MOL validation and group checking"""
    
#     def reconstruction_loss(y_true, y_pred):
#         """Standard reconstruction loss"""
#         return tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    
#     def rdkit_validity_loss(y_pred):
#         """Check if predicted SMILES can create valid MOL file"""
#         def check_rdkit_valid(pred_logits):
#             pred_tokens = tf.argmax(pred_logits, axis=-1)
#             tokenizer = PreTrainedTokenizerFast.from_pretrained("code/vocab/smiles_tokenizer")
#             try:
#                 smiles = tokenizer.decode(pred_tokens.numpy(), skip_special_tokens=True)
#                 mol = Chem.MolFromSmiles(smiles)
#                 return tf.constant(1.0) if mol is not None else tf.constant(0.0)
#             except:
#                 return tf.constant(0.0)
            
#         validity_scores = tf.map_fn(
#             check_rdkit_valid,
#             y_pred,
#             fn_output_signature=tf.float32
#         )
        
#         return 1.0 - tf.reduce_mean(validity_scores)
    
#     def group_presence_loss(y_pred, desired_groups):
#         """Check for presence of desired functional groups using RDKit"""
#         def check_groups(pred_logits):
#             pred_tokens = tf.argmax(pred_logits, axis=-1)
#             tokenizer = PreTrainedTokenizerFast.from_pretrained("code/vocab/smiles_tokenizer")
#             try:
#                 smiles = tokenizer.decode(pred_tokens.numpy(), skip_special_tokens=True)
#                 mol = Chem.MolFromSmiles(smiles)
#                 if mol is None:
#                     return tf.constant(0.0)
                
#                 # Check for each desired group
#                 group_patterns = {
#                     "C=C": "C=C",              # Vinyl group
#                     "NC": "[NX2]=[CX3]",       # Imine group
#                     "C1OC1": "[OX2]1[CX3][CX3]1",  # Epoxy group
#                     "CCS": "CCS",              # Thiol group
#                     "C=C(C=O)": "C=C(C=O)"     # Acrylic group
#                 }
                
#                 group_count = 0
#                 for group in desired_groups:
#                     if group in group_patterns:
#                         pattern = group_patterns[group]
#                         if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
#                             group_count += 1
                
#                 return tf.constant(sum(group_scores) / len(desired_groups))
#             except:
#                 return tf.constant(0.0)
            
#         group_scores = tf.map_fn(
#             check_groups,
#             y_pred,
#             fn_output_signature=tf.float32
#         )
        
#         return 1.0 - tf.reduce_mean(group_scores)
    
#     return reconstruction_loss, rdkit_validity_loss, group_presence_loss

# def combined_loss(desired_groups, weights=[1.0, 0.5, 0.5]):
#     """Combine all three losses with weights"""
#     recon_loss, valid_loss, group_loss = create_weighted_losses()
    
#     def loss_function(y_true, y_pred):
#         # Calculate losses
#         r_loss = recon_loss(y_true, y_pred)
#         v_loss = valid_loss(y_pred)
#         g_loss = group_loss(y_pred, desired_groups)
        
#         # Print for monitoring
#         tf.print("\nLosses:", {
#             "reconstruction": r_loss,
#             "rdkit_validity": v_loss,
#             "group_presence": g_loss
#         })
        
#         return weights[0] * r_loss + weights[1] * v_loss + weights[2] * g_loss
    
#     return loss_function
