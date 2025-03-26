from Model import *
from Data_Process_with_prevocab_gen import *
import tensorflow as tf
import tensorflow.keras.backend as K
import random
import keras
from collections import defaultdict
from Sample_Predictor import *
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from Reward_Score import *


@keras.saving.register_keras_serializable()
class DiffusionCustomLoss(tf.keras.losses.Loss):
    def __init__(self, group_smarts_list, name='custom_molecule_loss',
                  reduction='sum_over_batch_size',lambda_rl=0.1, clip_epsilon=0.2):
        super().__init__(name=name, reduction=reduction)
        self.group_smarts_list = group_smarts_list
        
        # Initialize weights
        self.diversity_weight = 1.0
        self.validity_weight = 0.9
        self.scaffold_weight = 0.7
        self.size_weight = 0.7
        self.ring_weight = 0.7
        self.reward_weight = 0.8
        self.kl_weight = 0.6
        self.entropy_weight = 0.6


        self.lambda_rl = lambda_rl  # RL weight scaling factor
        self.clip_epsilon = clip_epsilon

    def _decode_predictions(self, pred_tensor):
        pred_indices = tf.argmax(pred_tensor, axis=-1)
        predicted_smiles = []
        for tokens in pred_indices:
            smiles = decode_smiles(tokens, Constants.TOKENIZER_PATH)
            predicted_smiles.append(smiles)
        return predicted_smiles

    def _calculate_tanimoto(self, smiles1, smiles2):
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

    def _diversity_loss(self, y_pred, y_true):
        smiles_list = self._decode_predictions(y_pred)
        tanimoto_scores = []
        for i in range(len(smiles_list)):
            input_smiles = decode_smiles(y_true[i], Constants.TOKENIZER_PATH)
            if smiles_list[i] != "":
                score = self._calculate_tanimoto(input_smiles, smiles_list[i])
                if score <0.5:
                    tanimoto_scores.append(score)
                else:
                    tanimoto_scores.append(1.0)
            else:
                tanimoto_scores.append(1.0)
        return tf.constant(tanimoto_scores, dtype=tf.float32)

    def _check_validity(self, pred_tensor):
        smiles_list = self._decode_predictions(pred_tensor)
        validity_loss = []
        valid_smiles = []
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None and smiles != "":
                    valid_smiles.append(smiles)
                    validity_loss.append(0.0)
                else:
                    validity_loss.append(1.0)
            except:
                validity_loss.append(1.0)
        return tf.constant(validity_loss, dtype=tf.float32), valid_smiles

    def _calculate_scaffold_entropy(self, y_pred):
        smiles_list = self._decode_predictions(y_pred)
        scaffolds = defaultdict(int)
        total_mols = 0
        
        for smiles in smiles_list:
            if smiles != "":
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                        if scaffold is not None:
                            scaffold_smiles = Chem.MolToSmiles(scaffold, canonical=True)
                            scaffolds[scaffold_smiles] += 1
                            total_mols += 1
                except:
                    continue

        if total_mols == 0:
            return tf.constant(1.0, dtype=tf.float32)

        probs = [count/total_mols for count in scaffolds.values()]
        epsilon = 1e-10
        entropy = tf.reduce_sum([-p * tf.math.log(p + epsilon) for p in probs])
        max_possible_entropy = tf.math.log(tf.cast(len(scaffolds), tf.float32) + epsilon)
        
        if max_possible_entropy > epsilon:
            normalized_entropy = entropy / max_possible_entropy
        else:
            normalized_entropy = tf.constant(0.0, dtype=tf.float32)
            
        normalized_entropy = tf.clip_by_value(normalized_entropy, 0.0, 1.0)
        return 1.0 - normalized_entropy

    def _calculate_size_distribution(self, y_pred):
        smiles_list = self._decode_predictions(y_pred)
        sizes = defaultdict(int)
        total_mols = 0
        
        for smiles in smiles_list:
            if smiles != "":
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        size = mol.GetNumHeavyAtoms()
                        if size > 0:
                            sizes[size] += 1
                            total_mols += 1
                except:
                    continue

        if total_mols == 0:
            return tf.constant(1.0, dtype=tf.float32)

        probs = [count/total_mols for count in sizes.values()]
        epsilon = 1e-10
        entropy = tf.reduce_sum([-p * tf.math.log(p + epsilon) for p in probs])
        max_possible_entropy = tf.math.log(tf.cast(len(sizes), tf.float32) + epsilon)
        
        if max_possible_entropy > epsilon:
            normalized_entropy = entropy / max_possible_entropy
        else:
            normalized_entropy = tf.constant(0.0, dtype=tf.float32)
            
        normalized_entropy = tf.clip_by_value(normalized_entropy, 0.0, 1.0)
        return 1.0 - normalized_entropy

    def _calculate_ring_diversity(self, y_pred):
        smiles_list = self._decode_predictions(y_pred)
        ring_systems = defaultdict(int)
        total_mols = 0
        
        for smiles in smiles_list:
            if smiles != "":
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        n_rings = Chem.rdMolDescriptors.CalcNumRings(mol)
                        ring_systems[n_rings] += 1
                        total_mols += 1
                except:
                    continue

        if total_mols == 0:
            return tf.constant(1.0, dtype=tf.float32)

        probs = [count/total_mols for count in ring_systems.values()]
        epsilon = 1e-10
        entropy = tf.reduce_sum([-p * tf.math.log(p + epsilon) for p in probs])
        max_possible_entropy = tf.math.log(tf.cast(len(ring_systems), tf.float32) + epsilon)
        
        if max_possible_entropy > epsilon:
            normalized_entropy = entropy / max_possible_entropy
        else:
            normalized_entropy = tf.constant(0.0, dtype=tf.float32)
            
        normalized_entropy = tf.clip_by_value(normalized_entropy, 0.0, 1.0)
        return 1.0 - normalized_entropy
    
    # def mmi_loss(self,y_true, y_pred):
    #     cross_entropy = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    #     reverse_entropy = tf.keras.losses.sparse_categorical_crossentropy(y_pred, y_true)
    #     return tf.reduce_mean(cross_entropy - 0.1 * reverse_entropy)
    
    def kl_divergence_loss(self,y_true, y_pred):
        return tf.reduce_mean(tf.keras.losses.KLDivergence()(y_true, y_pred))
    
    def call(self, y_true, y_pred):
        y_pred = tf.map_fn(lambda x: tf.abs(tf.cast(x, tf.int32)), y_pred, dtype=tf.int32)
        y_true = tf.map_fn(lambda x: tf.abs(tf.cast(x, tf.int32)), y_true, dtype=tf.int32)





        validity_scores, valid_smiles = self._check_validity(y_pred)
        validity_loss = tf.reduce_mean(validity_scores)
        scaffold_loss = self._calculate_scaffold_entropy(y_pred)
        size_loss = self._calculate_size_distribution(y_pred)
        ring_loss = self._calculate_ring_diversity(y_pred)

        print('\nLoss Components:')
        print(f'Validity Loss: {validity_loss.numpy():.4f}')
        print(f'Scaffold Diversity Loss: {scaffold_loss:.4f}')
        print(f'Size Diversity Loss: {size_loss:.4f}')
        print(f'Ring Diversity Loss: {ring_loss:.4f}')
        


        total_loss = (
                     self.validity_weight * validity_loss +
                     self.scaffold_weight * scaffold_loss +
                     self.size_weight * size_loss +
                     self.ring_weight * ring_loss )
                  

        predicted_smiles = self._decode_predictions(y_pred)

        # Calculate rewards for each prediction
        batch_reward_score = []
        batch_diversity_reward = []
        for i in range(len(predicted_smiles)):
            try:
                input_smiles = decode_smiles(y_true[i], Constants.TOKENIZER_PATH)
                target_groups = extract_group_smarts(input_smiles)
                reward_score = calculate_reward_score(predicted_smiles[i], target_groups)
                diversity_reward = calculate_diversity_reward(input_smiles, predicted_smiles[i])
                if reward_score is None or not np.isfinite(reward_score):
                    reward_score = 0.0
                batch_reward_score.append(float(reward_score))  # Ensure float type
                batch_diversity_reward.append(float(diversity_reward))
            except Exception as e:
                print(f"Error calculating reward for molecule {i}: {e}")
                batch_reward_score.append(0.0)

        # Convert rewards to tensor and normalize
        if not batch_reward_score:
            batch_reward_score = [0.0]
        if not batch_diversity_reward:
            batch_diversity_reward = [0.0]
        

        rewards = np.array(batch_reward_score, dtype=np.float32)
        diversity_rewards = np.array(batch_diversity_reward, dtype=np.float32)

        # Avoid division by zero in normalization
        reward_std = np.std(rewards)
        diversity_reward_std = np.std(diversity_rewards)
        if reward_std < 1e-8:
            normalized_rewards = np.zeros_like(rewards, dtype=np.float32)
        else:
            normalized_rewards = (rewards - np.mean(rewards)) / (reward_std + 1e-8)
        
        if diversity_reward_std < 1e-8:
            normalized_diversity_rewards = np.zeros_like(diversity_rewards, dtype=np.float32)
        else:
            normalized_diversity_rewards = (diversity_rewards - np.mean(diversity_rewards)) / (diversity_reward_std + 1e-8)
        
        policy_loss = -np.mean(normalized_rewards)
        diversity_loss = -np.mean(normalized_diversity_rewards)
        # Combine with other losses
        final_loss = total_loss + self.lambda_rl * policy_loss + self.lambda_rl * diversity_loss

        print(f'Reward Mean: {tf.reduce_mean(rewards):.4f}')
        print(f'Policy Loss: {policy_loss:.4f}')
        print(f'Final Loss: {final_loss:.4f}')

        return final_loss

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "group_smarts_list": list(self.group_smarts_list),
        }
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

