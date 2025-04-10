import tensorflow as tf
from rdkit import Chem
from rdkit.Chem import DataStructs
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem import Descriptors

class GroupRewardScorePredictor:
    def __init__(self, model_path=None):
        # Get the root directory of the project
        self.root_dir = Path(__file__).parent.parent.parent  # Goes up three levels to COT_TSMP
        self.model_dir = self.root_dir /"RLHF" / "saved_models"
        
        self.feature_size = 200
        self.radius = 2
        
        # Create models
        self.score_model = self._build_model()
        
        # Initialize scalers
        self.feature_scaler = StandardScaler()
        
        if model_path:
            self.load_models(model_path)

    # def _build_model(self):
    #     model = Sequential([
    #         Dense(256, activation='relu', input_shape=(208,)),  # Change input shape from 200 to 206
    #         Dense(128, activation='relu'),
    #         Dense(64, activation='relu'),
    #         Dense(1)
    #     ])
    #     model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    #     return model
    def _build_model(self):
        model = Sequential([
            # Input layer with dropout
            Dense(512, activation='relu', input_shape=(216,), 
                  kernel_regularizer=l2(0.01)),
           
            Dropout(0.3),
            
            # First hidden layer
            Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
          
            Dropout(0.2),
            
            # Second hidden layer
            Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
            
            Dropout(0.2),
            
            # Third hidden layer
            Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        
            Dropout(0.1),
            
            # Output layer
            Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae', 'mse']
        )
        return model


    def _get_molecular_features(self, smiles1, smiles2):
        """Generate molecular features from SMILES pairs"""
        try:
            # Convert SMILES to RDKit molecules
            mol1 = Chem.MolFromSmiles(smiles1)
            mol2 = Chem.MolFromSmiles(smiles2)

            if mol1 is None or mol2 is None:
                raise ValueError("Invalid SMILES string")

            # Generate Morgan fingerprints
            morgan_gen = GetMorganGenerator(radius=self.radius, fpSize=self.feature_size)
            fp1 = np.array(morgan_gen.GetFingerprintAsNumPy(mol1), dtype=np.float32)
            fp2 = np.array(morgan_gen.GetFingerprintAsNumPy(mol2), dtype=np.float32)

            # Combine features (you can modify this combination strategy)
            combined_fp = fp1 + fp2  # Simple addition of fingerprints

            # Add additional descriptors if needed
            desc1 = np.array([
                Descriptors.MolWt(mol1),
                Descriptors.MolLogP(mol1),
                Descriptors.TPSA(mol1)
            ], dtype=np.float32)

            desc2 = np.array([
                Descriptors.MolWt(mol2),
                Descriptors.MolLogP(mol2),
                Descriptors.TPSA(mol2)
            ], dtype=np.float32)

            # Combine all features
            features = np.concatenate([combined_fp, desc1, desc2])

            return features.reshape(1, -1)

        except Exception as e:
            print(f"Error in feature generation: {str(e)}")
            return None
        
    def get_group_onehot_encoding(self,group_label):
    # Define all possible functional groups
        GROUP_MAPPING = {
        'C=C': [1, 0, 0, 0, 0],      # Vinyl
        'CCS': [0, 1, 0, 0, 0],      # Thiol
        'C=C(C=O)': [0, 0, 1, 0, 0], # Acryl
        'C1OC1': [0, 0, 0, 1, 0],    # Epoxy
        'NC': [0, 0, 0, 0, 1],       # Imine
        'No_group': [0, 0, 0, 0, 0]  # No functional group
        }
        return GROUP_MAPPING.get(group_label, [0, 0, 0, 0, 0])



    def train(self, train_data, validation_split=0.2, epochs=100):
        """
        Train the property prediction models
        
        train_data: dict containing:
            'smiles1': list of first monomer SMILES
            'smiles2': list of second monomer SMILES
            'group1': list of group1 values
            'group2': list of group2 values
            'score': list of score values
        """
        # Generate features for all pairs
        features = []
        group1_onehot = []
        group2_onehot = []
        scores = []
        for s1, s2, g1, g2, score in zip(train_data['smiles1'], train_data['smiles2'], 
                                 train_data['group1'], train_data['group2'],train_data['score']):
            feat = self._get_molecular_features(s1, s2)
            g1_onehot = self.get_group_onehot_encoding(g1)
            g2_onehot = self.get_group_onehot_encoding(g2)
            if feat is not None:
                # Add ratio features to the molecular features
                feat_with_ratios = np.concatenate([feat.squeeze(), g1_onehot, g2_onehot])
                features.append(feat_with_ratios)
                group1_onehot.append(g1_onehot)
                group2_onehot.append(g2_onehot)
                scores.append(score)
        features = np.array(features)
        group1_onehot = np.array(group1_onehot)
        group2_onehot = np.array(group2_onehot)
        scores = np.array(scores)
        # Scale features and targets
        self.feature_scaler.fit(features)
        features_scaled = self.feature_scaler.transform(features)
        
    
    
        
        # Train models
        print("Training Reward model...")
        self.score_model.fit(
            features_scaled, scores,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=32,
            verbose=1
        )
        

    def predict(self, smiles1, smiles2, group1, group2):
        """Predict Er and Tg values for a SMILES pair"""
        features = self._get_molecular_features(smiles1, smiles2)
        if features is None:
            return None
            
       
        
        # Scale features
        
        group1_onehot = self.get_group_onehot_encoding(group1)  
        group2_onehot = self.get_group_onehot_encoding(group2)
        features_with_ratios = np.concatenate([features.squeeze(), group1_onehot, group2_onehot])
        features_with_ratios = features_with_ratios.reshape(1, -1)
        features_scaled = self.feature_scaler.transform(features_with_ratios)
        
        # Make predictions
        score_pred_scaled = self.score_model.predict(features_scaled)
        print(score_pred_scaled)
        

        
        return score_pred_scaled[0][0]

    def save_models(self, save_dir=None):
        """Save models and scalers"""
        if save_dir is None:
            save_dir = self.model_dir
        
        # Convert to Path object if it's a string
        save_dir = Path(save_dir)
        
        # Create directory if it doesn't exist
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save neural network models
        self.score_model.save(save_dir / 'score_model.keras')
        
        # Save scalers
        joblib.dump(self.feature_scaler, save_dir / 'feature_scaler.pkl')
        
        print(f"Models and scalers saved to {save_dir}")

    def load_models(self, model_dir=None):
        """Load saved models and scalers"""
        if model_dir is None:
            model_dir = self.model_dir
            
        # Convert to Path object if it's a string
        model_dir = Path(model_dir)
        
        try:
            # Check if model files exist
            score_model_path = model_dir / 'score_model.keras'
            
            if not score_model_path.exists():
                print(f"No saved models found in {model_dir}")
                return
            
            # Load neural network models
            self.score_model = tf.keras.models.load_model(score_model_path)
            
            # Load scalers
            self.feature_scaler = joblib.load(model_dir / 'feature_scaler.pkl')
            
            print(f"Models and scalers loaded successfully from {model_dir}")
            
        except Exception as e:
            print(f"Error loading models from {model_dir}: {str(e)}")
            print("Initializing new models")

# # Example usage:
# if __name__ == "__main__":
#     # Example training data
#     train_data = {
#         'smiles1': ["CC=CC", "CCC=O", "c1ccccc1"],
#         'smiles2': ["CCCC", "CCCN", "c1ccccc1"],
#         'er': [0.5, 0.7, 0.3],
#         'tg': [100, 150, 200]
#     }
    
#     # Initialize predictor
#     predictor = PropertyPredictor()
    
#     # Train models
#     history_er, history_tg = predictor.train(train_data, epochs=10)  # Reduced epochs for example
    
#     # Save models
#     predictor.save_models()
    
#     # Make predictions
#     er_pred, tg_pred = predictor.predict("CC=CC", "CCCC")
#     print(f"Predicted Er: {er_pred:.2f}")
#     print(f"Predicted Tg: {tg_pred:.2f}")