import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import os


def read_and_print_results():
    # Read the good results CSV
    good_results_df = pd.read_csv('Two_Monomers_Group/output_analyzer/latest2/good_results_new_noise_pred.csv')
    print("\nColumns in good results:")
    print(good_results_df.columns)

    # Read the bad results CSV 
    good_smiles_results_df = pd.read_csv('Two_Monomers_Group/output_analyzer/latest2/good_results_new_noise_pred_S.csv')
    print("\nColumns in bad results:")
    print(good_smiles_results_df.columns)

def combine_results():
    """
    Combines the good results and good SMILES results CSVs into a single CSV file,
    keeping only selected columns and removing duplicates.
    """
    # Read both CSV files
    good_results = pd.read_csv('Two_Monomers_Group/output_analyzer/latest2/good_results_new_noise_pred.csv')
    good_smiles = pd.read_csv('Two_Monomers_Group/output_analyzer/latest2/good_results_new_noise_pred_S.csv')
    
    # Select only specified columns
    columns_to_keep = ['pred_monomer1', 'pred_monomer2', 'tanimoto_avg', 'complexity_ratio', 'From_noise']
    good_results = good_results[columns_to_keep]
    good_smiles = good_smiles[columns_to_keep]
    
    # Combine the dataframes and remove duplicates
    combined_df = pd.concat([good_results, good_smiles], axis=0, ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=['pred_monomer1', 'pred_monomer2'])
    
    # Add Serial No column
    combined_df.insert(0, 'Serial_No', range(1, len(combined_df) + 1))
    
    # Save combined results
    output_path = 'Two_Monomers_Group/output_analyzer/latest2/combined_good_results.csv'
    combined_df.to_csv(output_path, index=False)
    print(f"Combined results saved to: {output_path}")
    print(f"Total number of unique rows: {len(combined_df)}")

def analyze_noise_statistics():
    """
    Analyzes and prints statistics about the From_noise column in the combined results.
    """
    # Read the combined results
    combined_df = pd.read_csv('Two_Monomers_Group/output_analyzer/latest2/combined_good_results.csv')
    
    # Get value counts and percentages
    noise_counts = combined_df['From_noise'].value_counts()
    noise_percentages = combined_df['From_noise'].value_counts(normalize=True) * 100
    
    print("\nNoise Statistics:")
    print("-----------------")
    print("\nCounts:")
    print(noise_counts)
    print("\nPercentages:")
    for value, percentage in noise_percentages.items():
        print(f"{value}: {percentage:.2f}%")
    
    # Calculate average tanimoto scores for each category
    avg_tanimoto = combined_df.groupby('From_noise')['tanimoto_avg'].mean()
    print("\nAverage Tanimoto Scores by Category:")
    print(avg_tanimoto)

def draw_molecules():
    """
    Draws and saves molecular structures for each monomer pair in the combined results,
    naming files by their Serial_No.
    """
   
    # Create output directory if it doesn't exist
    output_dir = 'Two_Monomers_Group/output_analyzer/latest2/molecule_drawings'
    os.makedirs(output_dir, exist_ok=True)

    # Read the combined results
    combined_df = pd.read_csv('Two_Monomers_Group/output_analyzer/latest2/combined_good_results.csv')
    
    for _, row in combined_df.iterrows():
        try:
            # Convert SMILES to molecules
            mol1 = Chem.MolFromSmiles(row['pred_monomer1'])
            mol2 = Chem.MolFromSmiles(row['pred_monomer2'])
            
            if mol1 and mol2:
                # Draw molecules side by side
                img = Draw.MolsToImage([mol1, mol2], subImgSize=(1024, 1024), 
                                     legends=[f"Monomer 1", f"Monomer 2"])
                
                # Save the image with Serial_No as filename
                img.save(os.path.join(output_dir, f"SL_{row['Serial_No']}.png"))
                
        except Exception as e:
            print(f"Error processing Serial_No {row['Serial_No']}: {e}")
    
    print(f"Molecule drawings saved in: {output_dir}")

    

if __name__ == "__main__":
    #read_and_print_results()
    combine_results()
    analyze_noise_statistics()
    draw_molecules()
