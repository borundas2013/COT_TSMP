import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import numpy as np
from typing import Dict, List, Tuple

class DataProcessor:

    @staticmethod
    def load_data(excel_path):
        try:
            # Read Excel file
            df = pd.read_excel(excel_path)
            
            # Check if required columns exist
            required_cols = ['SMILES', 'Er', 'Tg','Ratio_1','Ratio_2']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' not found in Excel file")
            
            # Initialize lists for storing data
            smiles1_list = []
            smiles2_list = []
            er_list = []
            tg_list = []
            ratio_1_list = []
            ratio_2_list = []
            # Process each row
            for _, row in df.iterrows():
                try:
                    # Extract the two SMILES from the SMILES column
                    smiles_pair = eval(row['SMILES'])  # Safely evaluate string representation of list
                    if len(smiles_pair) == 2:
                        smiles1, smiles2 = smiles_pair[0], smiles_pair[1]
                        smiles1_list.append(smiles1)
                        smiles2_list.append(smiles2)
                        er_list.append(row['Er'])
                        tg_list.append(row['Tg'])
                        ratio_1_list.append(row['Ratio_1'])
                        ratio_2_list.append(row['Ratio_2'])
                except:
                    print(f"Skipping malformed SMILES pair: {row['SMILES']}")
                    continue

            return {
                'smiles1': smiles1_list,
                'smiles2': smiles2_list,
                'er': er_list,
                'tg': tg_list,
                'ratio_1': ratio_1_list,
                'ratio_2': ratio_2_list
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
            'er': [data['er'][i] for i in train_indices],
            'tg': [data['tg'][i] for i in train_indices],
            'ratio_1': [data['ratio_1'][i] for i in train_indices],
            'ratio_2': [data['ratio_2'][i] for i in train_indices]
        }
        
        test_data = {
            'smiles1': [data['smiles1'][i] for i in test_indices],
            'smiles2': [data['smiles2'][i] for i in test_indices],
            'er': [data['er'][i] for i in test_indices],
            'tg': [data['tg'][i] for i in test_indices],
            'ratio_1': [data['ratio_1'][i] for i in test_indices],
            'ratio_2': [data['ratio_2'][i] for i in test_indices]
        }
        
        return train_data, test_data 