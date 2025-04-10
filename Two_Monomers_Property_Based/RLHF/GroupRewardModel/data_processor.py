import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import numpy as np
from typing import Dict, List, Tuple

class DataProcessor:

    @staticmethod
    def load_data(data_path):
        try:
            # Read Excel file
            df = pd.read_csv(data_path)
            
            # Check if required columns exist
            required_cols = ['monomer1', 'monomer2','group1','group2','score']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' not found in Excel file")
            
            # Initialize lists for storing data
            smiles1_list = []
            smiles2_list = []
            group1_list = []
            group2_list = []
            score_list = []
            # Process each row
            for _, row in df.iterrows():
                try:
                    # Extract the two SMILES from the SMILES column
                    
                    smiles1, smiles2 = row['monomer1'], row['monomer2']
                    smiles1_list.append(smiles1)
                    smiles2_list.append(smiles2)
                    group1_list.append(row['group1'])
                    group2_list.append(row['group2'])
                    score_list.append(row['score'])
                except:
                    print(f"Skipping malformed SMILES pair: {row['SMILES']}")
                    continue

            return {
                'smiles1': smiles1_list,
                'smiles2': smiles2_list,
                'group1': group1_list,
                'group2': group2_list,
                'score': score_list
            }
                    
            
            
        except Exception as e:
            print(f"Error processing Excel file: {str(e)}")
            raise

    @staticmethod
    def split_data(data: Dict[str, List], 
                   train_ratio: float = 0.8) -> Tuple[Dict[str, List], Dict[str, List]]:
        """Split data into training and testing sets"""
        n = len(data['smiles1'])
        print(n)
        indices = np.random.permutation(n)
        train_size = int(n * train_ratio)
        
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        train_data = {
            'smiles1': [data['smiles1'][i] for i in train_indices],
            'smiles2': [data['smiles2'][i] for i in train_indices],
            'group1': [data['group1'][i] for i in train_indices],
            'group2': [data['group2'][i] for i in train_indices],
            'score': [data['score'][i] for i in train_indices]
        }
        
        test_data = {
            'smiles1': [data['smiles1'][i] for i in test_indices],
            'smiles2': [data['smiles2'][i] for i in test_indices],
            'group1': [data['group1'][i] for i in test_indices],
            'group2': [data['group2'][i] for i in test_indices],
            'score': [data['score'][i] for i in test_indices]
        }
        
        return train_data, test_data 