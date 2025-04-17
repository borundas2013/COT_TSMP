# import json
# import csv
# import pandas as pd
# from dual_smile_process import *


# monomer1,monomer2, er,tg = process_dual_monomer_data('LLM_Tuned_COT/Data/unique_smiles_Er.csv','LLM_Tuned_COT/Data/smiles.xlsx')
# monomer1, monomer2 = np.array(monomer1), np.array(monomer2)

# def load_tsmp_data(file_path):
    
#     try:
#         with open(file_path, 'r') as f:
#             data_list = json.load(f)
        
#         # Initialize list to store all unique pairs
#         all_unique_pairs = []
        
#         # Process each dictionary in the list
#         for data_dict in data_list:
#             # Extract unique pairs from each dictionary
#             unique_pairs = data_dict.get('unique_pairs', [])
            
#             # Process pairs from this dictionary
#             for pair in unique_pairs:
#                 extracted_pair = {
#                     'smile1': pair.get('smile1', ''),
#                     'smile2': pair.get('smile2', ''),
#                     'reaction': pair.get('reaction', False),
#                     'temperature': pair.get('temperature', 0),
#                     'groups': pair.get('groups', []),
#                     'prompt': pair.get('prompt', '')
#                 }
#                 all_unique_pairs.append(extracted_pair)
            
#         return all_unique_pairs
    
#     except FileNotFoundError:
#         print(f"Error: File {file_path} not found")
#         return []
#     except json.JSONDecodeError:
#         print(f"Error: Invalid JSON format in {file_path}")
#         return []
#     except Exception as e:
#         print(f"Error occurred while loading data: {str(e)}")
#         return []

# def filter_by_reaction(data, reaction_status=True):
 
#     return [item for item in data if item.get('reaction') == reaction_status]

# def analyze_and_save_tsmp_data(input_files):
#     """
#     Analyze TSMP data and save results to separate CSV files for reactive and non-reactive pairs.
    
#     Args:
#         input_files (list): List of input JSON file paths
#     """
#     # Initialize lists to store all reactive and non-reactive pairs
#     all_reactive_pairs = []
#     all_non_reactive_pairs = []
    
#     # Process each input file
#     for input_file in input_files:
#         # Load and analyze data
#         all_pairs = load_tsmp_data(input_file)
#         reactive_pairs = filter_by_reaction(all_pairs, True)
#         non_reactive_pairs = filter_by_reaction(all_pairs, False)
        
#         # Add to combined lists
#         all_reactive_pairs.extend([{
#             'SMILE1': pair['smile1'],
#             'SMILE2': pair['smile2'],
#             'Temperature': pair.get('temperature', 0),
#             'Prompt': pair.get('prompt', ''),
#             'Source_File': input_file.split('/')[-1]
#         } for pair in reactive_pairs])
        
#         all_non_reactive_pairs.extend([{
#             'SMILE1': pair['smile1'],
#             'SMILE2': pair['smile2'],
#             'Temperature': pair.get('temperature', 0),
#             'Prompt': pair.get('prompt', ''),
#             'Source_File': input_file.split('/')[-1]
#         } for pair in non_reactive_pairs])
        
#         # Print statistics for each file
#         print(f"\nStatistics for {input_file.split('/')[-1]}:")
#         print(f"Total pairs analyzed: {len(all_pairs)}")
#         print(f"Reactive pairs found: {len(reactive_pairs)}")
#         print(f"Non-reactive pairs found: {len(non_reactive_pairs)}")
    
#     # Save reactive pairs to CSV
#     reactive_df = pd.DataFrame(all_reactive_pairs)
#     reactive_df.to_csv('LLM_Tuned_COT/Output/all_reactive_pairs.csv', index=False)
    
#     # Save non-reactive pairs to CSV
#     non_reactive_df = pd.DataFrame(all_non_reactive_pairs)
#     non_reactive_df.to_csv('LLM_Tuned_COT/Output/all_non_reactive_pairs.csv', index=False)
    
#     # Print total statistics
#     print("\nTotal Statistics:")
#     print(f"Total reactive pairs: {len(all_reactive_pairs)}")
#     print(f"Total non-reactive pairs: {len(all_non_reactive_pairs)}")

# # Input files
# input_files = [
#     'LLM_Tuned_COT/Output/properties_generated_responses_large.json',
#     'LLM_Tuned_COT/Output/groups_generated_responses_large.json',
#     'LLM_Tuned_COT/Output/mix_generated_responses_large.json'
# ]

# analyze_and_save_tsmp_data(input_files)
# def check_combinations_in_dataset():
#     """
#     Check if SMILE combinations from reactive and non-reactive pairs exist in the original monomer dataset
#     and display statistics
#     """
#     # Load the CSV files
#     reactive_df = pd.read_csv('LLM_Tuned_COT/Output/all_reactive_pairs.csv')
#     non_reactive_df = pd.read_csv('LLM_Tuned_COT/Output/all_non_reactive_pairs.csv')
    
#     # Initialize counters
#     reactive_matches = 0
#     non_reactive_matches = 0
#     reactive_reverse_matches = 0
#     non_reactive_reverse_matches = 0


#     reactive_smile1_matches = 0
#     reactive_smile2_matches = 0
#     non_reactive_smile1_matches = 0
#     non_reactive_smile2_matches = 0
    
#     # Check reactive pairs
#     for _, row in reactive_df.iterrows():
#         smile1, smile2 = row['SMILE1'], row['SMILE2']
        
#         # Check both forward and reverse combinations
#         forward_match = np.any((monomer1 == smile1) & (monomer2 == smile2))
#         reverse_match = np.any((monomer1 == smile2) & (monomer2 == smile1))
        
#         if forward_match:
#             reactive_matches += 1
#         if reverse_match:
#             reactive_reverse_matches += 1


#         if np.any(monomer1 == smile1) or np.any(monomer2 == smile1):
#             reactive_smile1_matches += 1
#         if np.any(monomer1 == smile2) or np.any(monomer2 == smile2):
#             reactive_smile2_matches += 1
    
#     # Check non-reactive pairs
#     for _, row in non_reactive_df.iterrows():
#         smile1, smile2 = row['SMILE1'], row['SMILE2']
        
#         # Check both forward and reverse combinations
#         forward_match = np.any((monomer1 == smile1) & (monomer2 == smile2))
#         reverse_match = np.any((monomer1 == smile2) & (monomer2 == smile1))
        
#         if forward_match:
#             non_reactive_matches += 1
#         if reverse_match:
#             non_reactive_reverse_matches += 1

#         if np.any(monomer1 == smile1) or np.any(monomer2 == smile1):
#             non_reactive_smile1_matches += 1
#         if np.any(monomer1 == smile2) or np.any(monomer2 == smile2):
#             non_reactive_smile2_matches += 1
    
#     # Print detailed statistics
#     print("\nMatching Statistics:")
#     print("\nReactive Pairs:")
#     print(f"Total pairs checked: {len(reactive_df)}")
#     print(f"Forward matches found: {reactive_matches}")
#     print(f"Reverse matches found: {reactive_reverse_matches}")
#     print(f"Total matches (forward + reverse): {reactive_matches + reactive_reverse_matches}")
#     print(f"Percentage of matches: {((reactive_matches + reactive_reverse_matches) / len(reactive_df)) * 100:.2f}%")
#     print("\nIndividual SMILE matches in reactive pairs:")
#     print(f"SMILE1 matches: {reactive_smile1_matches} ({(reactive_smile1_matches / len(reactive_df)) * 100:.2f}%)")
#     print(f"SMILE2 matches: {reactive_smile2_matches} ({(reactive_smile2_matches / len(reactive_df)) * 100:.2f}%)")
    
#     print("\nNon-Reactive Pairs:")
#     print(f"Total pairs checked: {len(non_reactive_df)}")
#     print(f"Forward matches found: {non_reactive_matches}")
#     print(f"Reverse matches found: {non_reactive_reverse_matches}")
#     print(f"Total matches (forward + reverse): {non_reactive_matches + non_reactive_reverse_matches}")
#     print(f"Percentage of matches: {((non_reactive_matches + non_reactive_reverse_matches) / len(non_reactive_df)) * 100:.2f}%")
#     print("\nIndividual SMILE matches in non-reactive pairs:")
#     print(f"SMILE1 matches: {non_reactive_smile1_matches} ({(non_reactive_smile1_matches / len(non_reactive_df)) * 100:.2f}%)")
#     print(f"SMILE2 matches: {non_reactive_smile2_matches} ({(non_reactive_smile2_matches / len(non_reactive_df)) * 100:.2f}%)")
    
#     print("\nOverall Statistics:")

#     total_pairs = len(reactive_df) + len(non_reactive_df)
#     total_pair_matches = (reactive_matches + reactive_reverse_matches + 
#                          non_reactive_matches + non_reactive_reverse_matches)
#     total_smile1_matches = reactive_smile1_matches + non_reactive_smile1_matches
#     total_smile2_matches = reactive_smile2_matches + non_reactive_smile2_matches
#     total_pairs = len(reactive_df) + len(non_reactive_df)
#     total_matches = (reactive_matches + reactive_reverse_matches + 
#                     non_reactive_matches + non_reactive_reverse_matches)
#     print(f"Total pairs analyzed: {total_pairs}")
#     print(f"Total pair matches found: {total_pair_matches}")
#     print(f"Overall percentage of pair matches: {(total_pair_matches / total_pairs) * 100:.2f}%")
#     print(f"Total individual SMILE1 matches: {total_smile1_matches} ({(total_smile1_matches / total_pairs) * 100:.2f}%)")
#     print(f"Total individual SMILE2 matches: {total_smile2_matches} ({(total_smile2_matches / total_pairs) * 100:.2f}%)")

# # Run the analysis
# check_combinations_in_dataset()

# # Analyze and save data

import json
import csv
import pandas as pd
import numpy as np
from dual_smile_process import *
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import os

class TSMPAnalyzer:
    def __init__(self):
        # Load reference data
        self.monomer1, self.monomer2, self.er, self.tg = process_dual_monomer_data(
            'Data/unique_smiles_Er.csv',
            'Data/smiles.xlsx'
        )
        self.monomer1 = np.array(self.monomer1)
        self.monomer2 = np.array(self.monomer2)
        
    def load_tsmp_data(self, file_path):
        """Load TSMP data from JSON file"""
        try:
            with open(file_path, 'r') as f:
                data_list = json.load(f)
            
            all_unique_pairs = []
            for data_dict in data_list:
                unique_pairs = data_dict.get('unique_pairs', [])
                for pair in unique_pairs:
                    extracted_pair = {
                        'smile1': pair.get('smile1', ''),
                        'smile2': pair.get('smile2', ''),
                        'reaction': pair.get('reaction', False),
                        'temperature': pair.get('temperature', 0),
                        'groups': pair.get('groups', []),
                        'prompt': pair.get('prompt', '')
                    }
                    all_unique_pairs.append(extracted_pair)
            
            return all_unique_pairs
        
        except FileNotFoundError:
            print(f"Error: File {file_path} not found")
            return []
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {file_path}")
            return []
        except Exception as e:
            print(f"Error occurred while loading data: {str(e)}")
            return []

    def filter_by_reaction(self, data, reaction_status=True):
        """Filter pairs based on reaction status"""
        return [item for item in data if item.get('reaction') == reaction_status]

    def analyze_and_save_tsmp_data(self, input_files):
        """Analyze TSMP data and save to CSV files"""
        all_reactive_pairs = []
        all_non_reactive_pairs = []
        
        for input_file in input_files:
            all_pairs = self.load_tsmp_data(input_file)
            reactive_pairs = self.filter_by_reaction(all_pairs, True)
            non_reactive_pairs = self.filter_by_reaction(all_pairs, False)
            
            # Process reactive pairs
            all_reactive_pairs.extend([{
                'SMILE1': pair['smile1'],
                'SMILE2': pair['smile2'],
                'Temperature': pair.get('temperature', 0),
                'Prompt': pair.get('prompt', ''),
                'Reactive_groups': pair.get('groups', []),
                'Source_File': input_file.split('/')[-1]
            } for pair in reactive_pairs])
            
            # Process non-reactive pairs
            all_non_reactive_pairs.extend([{
                'SMILE1': pair['smile1'],
                'SMILE2': pair['smile2'],
                'Temperature': pair.get('temperature', 0),
                'Prompt': pair.get('prompt', ''),
                'Non_Reactive_groups': pair.get('groups', []),
                'Source_File': input_file.split('/')[-1]
            } for pair in non_reactive_pairs])
            
            print(f"\nStatistics for {input_file.split('/')[-1]}:")
            print(f"Total pairs analyzed: {len(all_pairs)}")
            print(f"Reactive pairs found: {len(reactive_pairs)}")
            print(f"Non-reactive pairs found: {len(non_reactive_pairs)}")
        
        # Create DataFrames and add SL column
        reactive_df = pd.DataFrame(all_reactive_pairs)
        reactive_df.insert(0, 'SL', range(1, len(reactive_df) + 1))
        
        non_reactive_df = pd.DataFrame(all_non_reactive_pairs)
        non_reactive_df.insert(0, 'SL', range(1, len(non_reactive_df) + 1))
        
        # Save to CSV
        reactive_df.to_csv('Lama32/Output/large_model/all_reactive_pairs.csv', index=False)
        non_reactive_df.to_csv('Lama32/Output/large_model/all_non_reactive_pairs.csv', index=False)
        
        print("\nTotal Statistics:")
        print(f"Total reactive pairs: {len(all_reactive_pairs)}")
        print(f"Total non-reactive pairs: {len(all_non_reactive_pairs)}")

    def check_combinations_in_dataset(self):
        """Check combinations and individual SMILE matches"""
        reactive_df = pd.read_csv('Lama32/Output/large_model/all_reactive_pairs.csv')
        non_reactive_df = pd.read_csv('Lama32/Output/large_model/all_non_reactive_pairs.csv')
        
        # Initialize counters
        stats = {
            'reactive': {
                'matches': 0,
                'reverse_matches': 0,
                'smile1_matches': 0,
                'smile2_matches': 0
            },
            'non_reactive': {
                'matches': 0,
                'reverse_matches': 0,
                'smile1_matches': 0,
                'smile2_matches': 0
            }
        }
        
        # Check reactive pairs
        for _, row in reactive_df.iterrows():
            self._process_pair(row, stats['reactive'])
        
        # Check non-reactive pairs
        for _, row in non_reactive_df.iterrows():
            self._process_pair(row, stats['non_reactive'])
        
        self._print_statistics(reactive_df, non_reactive_df, stats)

    def _process_pair(self, row, stats):
        """Process individual pair and update statistics"""
        smile1, smile2 = row['SMILE1'], row['SMILE2']
        
        # Check combinations
        forward_match = np.any((self.monomer1 == smile1) & (self.monomer2 == smile2))
        reverse_match = np.any((self.monomer1 == smile2) & (self.monomer2 == smile1))
        
        if forward_match:
            stats['matches'] += 1
        if reverse_match:
            stats['reverse_matches'] += 1
            
        # Check individual smiles
        if np.any(self.monomer1 == smile1) or np.any(self.monomer2 == smile1):
            stats['smile1_matches'] += 1
        if np.any(self.monomer1 == smile2) or np.any(self.monomer2 == smile2):
            stats['smile2_matches'] += 1

    def _print_statistics(self, reactive_df, non_reactive_df, stats):
        """Print detailed statistics"""
        # Print reactive statistics
        print("\nMatching Statistics:")
        self._print_category_statistics("Reactive Pairs", reactive_df, stats['reactive'])
        self._print_category_statistics("Non-Reactive Pairs", non_reactive_df, stats['non_reactive'])
        
        # Print overall statistics
        total_pairs = len(reactive_df) + len(non_reactive_df)
        total_matches = (stats['reactive']['matches'] + stats['reactive']['reverse_matches'] +
                        stats['non_reactive']['matches'] + stats['non_reactive']['reverse_matches'])
        total_smile1_matches = stats['reactive']['smile1_matches'] + stats['non_reactive']['smile1_matches']
        total_smile2_matches = stats['reactive']['smile2_matches'] + stats['non_reactive']['smile2_matches']
        
        print("\nOverall Statistics:")
        print(f"Total pairs analyzed: {total_pairs}")
        print(f"Total pair matches found: {total_matches}")
        print(f"Overall percentage of pair matches: {(total_matches / total_pairs) * 100:.2f}%")
        print(f"Total individual SMILE1 matches: {total_smile1_matches} ({(total_smile1_matches / total_pairs) * 100:.2f}%)")
        print(f"Total individual SMILE2 matches: {total_smile2_matches} ({(total_smile2_matches / total_pairs) * 100:.2f}%)")

    def _print_category_statistics(self, category_name, df, stats):
        """Print statistics for a specific category"""
        print(f"\n{category_name}:")
        print(f"Total pairs checked: {len(df)}")
        print(f"Forward matches found: {stats['matches']}")
        print(f"Reverse matches found: {stats['reverse_matches']}")
        total_matches = stats['matches'] + stats['reverse_matches']
        print(f"Total matches (forward + reverse): {total_matches}")
        print(f"Percentage of matches: {(total_matches / len(df)) * 100:.2f}%")
        print("\nIndividual SMILE matches:")
        print(f"SMILE1 matches: {stats['smile1_matches']} ({(stats['smile1_matches'] / len(df)) * 100:.2f}%)")
        print(f"SMILE2 matches: {stats['smile2_matches']} ({(stats['smile2_matches'] / len(df)) * 100:.2f}%)")

    def draw_molecule_pairs(self, output_dir='Lama32/Output/large_model/Molecules'):
        
        # Create output directories if they don't exist
        reactive_dir = os.path.join(output_dir, 'reactive')
        non_reactive_dir = os.path.join(output_dir, 'non_reactive')
        os.makedirs(reactive_dir, exist_ok=True)
        os.makedirs(non_reactive_dir, exist_ok=True)

        # Load the CSV files
        reactive_df = pd.read_csv('Lama32/Output/large_model/all_reactive_pairs.csv')
        non_reactive_df = pd.read_csv('Lama32/Output/large_model/all_non_reactive_pairs.csv')

        # Process reactive pairs
        print("\nDrawing Reactive Pairs...")
        self._draw_pairs(reactive_df, reactive_dir, "reactive")

        # Process non-reactive pairs
        print("\nDrawing Non-Reactive Pairs...")
        self._draw_pairs(non_reactive_df, non_reactive_dir, "non_reactive")

    def _draw_pairs(self, df, output_dir, pair_type):
      
        for _, row in df.iterrows():
            try:
                # Convert SMILES to RDKit molecules
              
                mol1 = Chem.MolFromSmiles(row['SMILE1'])
                mol2 = Chem.MolFromSmiles(row['SMILE2'])

                if mol1 is None or mol2 is None:
                    print(f"Warning: Could not parse SMILES for pair {row['SL']}")
                    continue

                # Generate 2D coordinates for better visualization
                AllChem.Compute2DCoords(mol1)
                AllChem.Compute2DCoords(mol2)

                # Prepare legends based on pair type and reactive groups
                legend1 = f'SMILE1\nTemp: {row["Temperature"]}'
                
                # Handle reactive groups for both types
                if pair_type == "reactive":
                    groups = row.get("Reactive_groups", [])
                    legend2 = f'SMILE2\nReactive groups: {groups}' if groups and groups != '[]' else 'SMILE2'
                else:
                    groups = row.get("Non_Reactive_groups", [])
                    legend2 = f'SMILE2\nNon-reactive groups: {groups}' if groups and groups != '[]' else 'SMILE2'

                # Add SL to legends
                #legend1 = f'SL: {row["SL"]}\n' + legend1
                #legend2 = f'SL: {row["SL"]}\n' + legend2

                # Create a combined image
                img = Draw.MolsToGridImage(
                    [mol1, mol2],
                    legends=[legend1, legend2],
                    subImgSize=(400, 400),
                    returnPNG=False
                )

                # Save the image using SL in filename
                filename = f"{row['SL']}.png"  # Zero-padded 4-digit number
                img.save(os.path.join(output_dir, filename))

                if (row['SL']) % 100 == 0:
                    print(f"Processed {row['SL']} {pair_type} pairs")
              

            except Exception as e:
                print(f"Error processing pair SL{row['SL']}: {str(e)}")


# Usage example
if __name__ == "__main__":
    analyzer = TSMPAnalyzer()
    
    input_files = [
        'Lama32/Output/properties_generated_responses.json',
        'Lama32/Output/group_generated_responses.json',
        'Lama32/Output/mix_generated_responses.json'
    ]
    
    # Analyze and save data
    analyzer.analyze_and_save_tsmp_data(input_files)
    
    # Check combinations
    analyzer.check_combinations_in_dataset()

    # Draw all molecules
    analyzer.draw_molecule_pairs()
    
    # # Draw selected molecules (example)
    # selected_indices = {
    #     'reactive': [0, 1, 2],  # First three reactive pairs
    #     'non_reactive': [0, 1, 2]  # First three non-reactive pairs
    # }
    # analyzer.draw_selected_pairs(selected_indices)



