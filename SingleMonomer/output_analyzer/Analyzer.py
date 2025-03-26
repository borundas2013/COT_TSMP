import json
import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import numpy as np

def calculate_similarities(smi1, smi2):
    """Calculate different similarity metrics between two SMILES strings"""
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)
    
    if mol1 is None or mol2 is None:
        return None, None, None
    
    # Morgan Fingerprints similarity (radius=2)
    fp1_morgan = AllChem.GetMorganFingerprintAsBitVect(mol1, 2)
    fp2_morgan = AllChem.GetMorganFingerprintAsBitVect(mol2, 2)
    morgan_sim = DataStructs.TanimotoSimilarity(fp1_morgan, fp2_morgan)
    
    # MACCS keys similarity
    fp1_maccs = MACCSkeys.GenMACCSKeys(mol1)
    fp2_maccs = MACCSkeys.GenMACCSKeys(mol2)
    maccs_sim = DataStructs.TanimotoSimilarity(fp1_maccs, fp2_maccs)
    
    # Topological fingerprints similarity
    fp1_topo = Chem.RDKFingerprint(mol1)
    fp2_topo = Chem.RDKFingerprint(mol2)
    topo_sim = DataStructs.TanimotoSimilarity(fp1_topo, fp2_topo)
    
    return morgan_sim, maccs_sim, topo_sim

def extract_json_data(json_file_path, similarity_threshold=0.5):
    # Read JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # Initialize lists to store extracted data
    all_data = []
    unique_data = []
    
    # Iterate through results
    for result in data['results']:
        input_smiles = result['input_smiles']
        
        # Keep track of unique generated SMILES for this input
        seen_smiles = set()
        
        # Process each generation for the current input SMILES
        for generation in result['generations']:
            temp = generation['temperature']
            generated_smiles = generation['generated_smiles']
            
            # Skip if generated SMILES is identical to input SMILES
            if generated_smiles == input_smiles:
                continue
            
            # Calculate similarities
            similarities = calculate_similarities(input_smiles, generated_smiles)
            
            # Skip if RDKit couldn't process the molecules
            if similarities is None:
                continue
                
            morgan_sim, maccs_sim, topo_sim = similarities
            if morgan_sim == None or maccs_sim == None or topo_sim == None:
                continue
            
            # Skip if any similarity is above threshold
            if (morgan_sim < similarity_threshold) and (maccs_sim < similarity_threshold) and (topo_sim < similarity_threshold):
                row_data = {
                    'input_smiles': input_smiles,
                    'temperature': temp,
                    'generated_smiles': generated_smiles,
                    'morgan_similarity': morgan_sim,
                    'maccs_similarity': maccs_sim,
                    'topo_similarity': topo_sim
                }
                all_data.append(row_data)
                
                # Add to unique_data only if not seen before
                if generated_smiles not in seen_smiles:
                    unique_row = row_data.copy()
                    unique_data.append(unique_row)
                    seen_smiles.add(generated_smiles)
        
    # Convert to pandas DataFrames
    df_all = pd.DataFrame(all_data)
    df_unique = pd.DataFrame(unique_data)
    
    return df_all, df_unique

def draw_molecule_comparison(input_smiles, generated_smiles, similarities=None, save_path=None):
    """
    Draw input and generated molecules side by side with similarity scores
    
    Parameters:
    - input_smiles: Input SMILES string
    - generated_smiles: Generated SMILES string
    - similarities: Tuple of (morgan_sim, maccs_sim, topo_sim)
    - save_path: Path to save the image (optional)
    """
    # Convert SMILES to molecules
    mol1 = Chem.MolFromSmiles(input_smiles)
    mol2 = Chem.MolFromSmiles(generated_smiles)
    
    if mol1 is None or mol2 is None:
        print("Error: Could not parse SMILES strings")
        return
    
    # Generate 2D coordinates for better visualization
    AllChem.Compute2DCoords(mol1)
    AllChem.Compute2DCoords(mol2)
    
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Draw molecules
    img1 = Draw.MolToImage(mol1)
    img2 = Draw.MolToImage(mol2)
    
    # Display molecules
    ax1.imshow(img1)
    ax1.set_title('Input Molecule')
    ax1.axis('off')
    
    ax2.imshow(img2)
    ax2.set_title('Generated Molecule')
    ax2.axis('off')
    
    # Add similarity scores if provided
    if similarities:
        morgan_sim, maccs_sim, topo_sim = similarities
        plt.figtext(0.5, 0.02, 
                   f'Similarities:\nMorgan: {morgan_sim:.3f} | MACCS: {maccs_sim:.3f} | Topo: {topo_sim:.3f}',
                   ha='center', bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def draw_multiple_comparisons(df, num_examples=5, save_dir=None):
    """
    Draw multiple random examples from the DataFrame
    
    Parameters:
    - df: DataFrame containing 'input_smiles', 'generated_smiles' and similarity columns
    - num_examples: Number of examples to draw
    - save_dir: Directory to save images (optional)
    """
    # Sample random rows
    samples = df.sample(min(num_examples, len(df)))
    
    for idx, row in samples.iterrows():
        similarities = (row['morgan_similarity'], 
                      row['maccs_similarity'], 
                      row['topo_similarity'])
        
        save_path = None
        if save_dir:
            save_path = f"{save_dir}/comparison_{idx}.png"
            
        draw_molecule_comparison(
            row['input_smiles'],
            row['generated_smiles'],
            similarities,
            save_path
        )

# Usage example:
file_path = 'output_analyzer/generation_results_rl_6.json'
df_all, df_unique = extract_json_data(file_path)

# Print statistics
print("All different generations DataFrame (similarity < 0.5):")
print(df_all)
print(f"\nTotal different generations: {len(df_all)}")

print("\nUnique different generations DataFrame (similarity < 0.5):")
print(df_unique)
print(f"\nTotal unique different generations: {len(df_unique)}")

# Analysis of similarity distributions
print("\nSimilarity Statistics for unique generations:")
for sim_type in ['morgan_similarity', 'maccs_similarity', 'topo_similarity']:
    print(f"\n{sim_type} statistics:")
    print(df_unique[sim_type].describe())

# Distribution of temperatures for successful generations
print("\nTemperature distribution for successful generations:")
print(df_unique['temperature'].value_counts().sort_index())

# Optionally, save to CSV filesr
df_all.to_csv('output_analyzer/rl_implemented_all_data_6.csv', index=False)
df_unique.to_csv('output_analyzer/rl_implemented_unique_6.csv', index=False)  


# draw_molecule_comparison(
#     df_unique.iloc[0]['input_smiles'],
#     df_unique.iloc[0]['generated_smiles'],
#     (df_unique.iloc[0]['morgan_similarity'],
#      df_unique.iloc[0]['maccs_similarity'],
#      df_unique.iloc[0]['topo_similarity'])
# )

# Or draw multiple random comparisons and save them
# import os
# save_dir = 'Code/json_output/molecule_comparisons_1'
# os.makedirs(save_dir, exist_ok=True)
# draw_multiple_comparisons(df_unique, num_examples=5, save_dir=save_dir)