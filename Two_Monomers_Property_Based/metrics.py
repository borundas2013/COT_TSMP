import tensorflow as tf
from rdkit import Chem
from rdkit import DataStructs
from transformers import PreTrainedTokenizerFast
import keras

@keras.saving.register_keras_serializable()
class SMILESQualityMetrics(tf.keras.metrics.Metric):
    def __init__(self, name='smiles_quality_metrics', **kwargs):
        super().__init__(name=name, **kwargs)
        # Initialize counters
        self.valid_count = self.add_weight(name='valid_count', initializer='zeros')
        self.group_count = self.add_weight(name='group_count', initializer='zeros')
        self.similarity_sum = self.add_weight(name='similarity_sum', initializer='zeros')
        self.total_sequences = self.add_weight(name='total_sequences', initializer='zeros')
        
        # Load tokenizer
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained("code/vocab/smiles_tokenizer")
    
    def generate_smiles(self, logits):
        """Convert logits to SMILES strings"""
        pred_tokens = tf.argmax(logits, axis=-1)
        
        def tokens_to_smiles(tokens):
            try:
                smiles = self.tokenizer.decode(tokens.numpy(), skip_special_tokens=True)
                return smiles.replace(" ", "")
            except:
                return ""
        
        smiles_list = tf.map_fn(
            lambda x: tf.py_function(tokens_to_smiles, [x], tf.string),
            pred_tokens,
            fn_output_signature=tf.string
        )
        return smiles_list
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Generate SMILES strings
        pred_smiles = self.generate_smiles(y_pred)
        true_smiles = self.generate_smiles(y_true)
        batch_size = tf.cast(tf.shape(pred_smiles)[0], tf.float32)
        self.total_sequences.assign_add(batch_size)
        
        # 1. Check validity
        def check_validity(smiles):
            print("ACC SMILES",smiles)
            try:
                mol = Chem.MolFromSmiles(smiles.numpy().decode())
                return tf.constant(1.0) if mol is not None else tf.constant(0.0)
            except:
                return tf.constant(0.0)
        
        valid_scores = tf.map_fn(
            lambda x: tf.py_function(check_validity, [x], tf.float32),
            pred_smiles,
            fn_output_signature=tf.float32
        )
        self.valid_count.assign_add(tf.reduce_sum(valid_scores))
        
        # 2. Check required groups
        def check_groups(smiles_pair):
            pred_smiles, true_smiles = smiles_pair
            try:
                pred_mol = Chem.MolFromSmiles(pred_smiles.numpy().decode())
                true_mol = Chem.MolFromSmiles(true_smiles.numpy().decode())
                if pred_mol is None or true_mol is None:
                    return tf.constant(0.0)
                
                # Common functional group SMARTS patterns
                patterns = {
                    "C=C": "C=C",
                    "NC": "[NX2]=[CX3]",
                    "C1OC1": "[OX2]1[CX3][CX3]1",
                    "CCS": "CCS",
                    "C=C(C=O)": "C=C(C=O)"
                }
                
                # Check which patterns are present in true SMILES
                required_patterns = []
                for pattern in patterns.values():
                    if true_mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                        required_patterns.append(pattern)
                
                # If no patterns found in true SMILES, return 1.0
                if not required_patterns:
                    return tf.constant(1.0)
                
                # Check if predicted SMILES has all the required patterns
                for pattern in required_patterns:
                    if not pred_mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                        return tf.constant(0.0)
                return tf.constant(1.0)
            except:
                return tf.constant(0.0)
        
        # Stack the predicted and true SMILES for processing
        smiles_pairs = tf.stack([pred_smiles, true_smiles], axis=1)
        group_scores = tf.map_fn(
            lambda x: tf.py_function(check_groups, [x], tf.float32),
            smiles_pairs,
            fn_output_signature=tf.float32
        )
        self.group_count.assign_add(tf.reduce_sum(group_scores))
        
        # 3. Calculate structural similarity
        def calculate_similarity(smiles_pair):
            pred_smiles, true_smiles = smiles_pair
            try:
                pred_mol = Chem.MolFromSmiles(pred_smiles.numpy().decode())
                true_mol = Chem.MolFromSmiles(true_smiles.numpy().decode())
                if pred_mol is None or true_mol is None:
                    return tf.constant(0.0)
                
                pred_fp = Chem.RDKFingerprint(pred_mol)
                true_fp = Chem.RDKFingerprint(true_mol)
                similarity = DataStructs.TanimotoSimilarity(pred_fp, true_fp)
                return tf.constant(float(similarity))
            except:
                return tf.constant(0.0)
        
        similarities = tf.map_fn(
            lambda x: tf.py_function(calculate_similarity, [x], tf.float32),
            smiles_pairs,
            fn_output_signature=tf.float32
        )
        self.similarity_sum.assign_add(tf.reduce_sum(similarities))
    
    def result(self):
        # Avoid division by zero
        total = tf.maximum(self.total_sequences, 1e-7)
        
        return {
            'validity_rate': self.valid_count / total,
            'group_accuracy': self.group_count / total,
            'structural_similarity': self.similarity_sum / total,
            'valid_group_rate': tf.where(  # Only consider group accuracy for valid molecules
                self.valid_count > 0,
                self.group_count / self.valid_count,
                0.0
            )
        }
    
    def reset_state(self):
        self.valid_count.assign(0.)
        self.group_count.assign(0.)
        self.similarity_sum.assign(0.)
        self.total_sequences.assign(0.)
    
    def get_config(self):
        return super().get_config()
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)