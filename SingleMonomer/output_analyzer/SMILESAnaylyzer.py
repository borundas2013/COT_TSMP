import os
import csv
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem

class SMILESAnalyzer:
    def __init__(self, input_csv_path, complexity_ratio_threshold=0.3):
        self.input_csv_path = input_csv_path
        self.filtered_pairs = []
        self.complexity_ratio_threshold = complexity_ratio_threshold  # New parameter

    def is_valid_molecule(self, smiles):
        """Check if molecule meets complexity criteria"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            
            # Check number of atoms (exclude very small molecules)
            num_atoms = mol.GetNumAtoms()
            if num_atoms < 5:
                return False
            
            # Check molecular weight
            mol_weight = Descriptors.ExactMolWt(mol)
            if mol_weight < 50:
                return False
            
            # Check number of rings
            num_rings = Descriptors.RingCount(mol)
            if num_rings < 1:
                return False
            
           
            
            return True
        except:
            return False

    def calculate_complexity_score(self, smiles):
        """Calculate a complexity score for a molecule"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0
            
            # Combine multiple descriptors for complexity
            num_atoms = mol.GetNumAtoms()
            num_rings = Descriptors.RingCount(mol)
            num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
            num_aromatic_rings = Descriptors.NumAromaticRings(mol)
            
            complexity = (
                num_atoms * 1.0 +
                num_rings * 2.0 +
                num_rotatable_bonds * 1.5 +
                num_aromatic_rings * 2.5
            )
            
            return complexity
        except:
            return 0

    def is_complexity_similar(self, input_complexity, pred_complexity):
        """Check if complexities are within acceptable range"""
        if input_complexity == 0:
            return False
            
        ratio = pred_complexity / input_complexity
        print(f"Ratio: {ratio}")
        # Check if ratio is within range [1-threshold, 1+threshold]
        return (1 - self.complexity_ratio_threshold) <= ratio <= (1 + self.complexity_ratio_threshold)

    def filter_pairs(self):
        """Filter SMILES pairs based on multiple criteria"""
        if not os.path.exists(self.input_csv_path):
            raise FileNotFoundError(f"Input CSV file not found at {self.input_csv_path}")

        with open(self.input_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                input_smiles = row['Input_SMILES']
                pred_smiles = row['Predicted_SMILES']
                tanimoto = float(row['Tanimoto_Score'])
                
                # Skip identical molecules
                if tanimoto >= 0.95:
                    continue
                
                # Check if predicted molecule is valid and complex enough
                if not self.is_valid_molecule(pred_smiles):
                    continue
                
                # Calculate complexity scores
                input_complexity = self.calculate_complexity_score(input_smiles)
                pred_complexity = self.calculate_complexity_score(pred_smiles)
                
                # Skip if complexities are too different
                if not self.is_complexity_similar(input_complexity, pred_complexity):
                    continue
                
                # Ensure predicted molecule has reasonable complexity
                if pred_complexity < 10:
                    continue
                
                # Calculate complexity ratio for reference
                complexity_ratio = pred_complexity / input_complexity if input_complexity > 0 else 0
                
                self.filtered_pairs.append({
                    'Input_SMILES': input_smiles,
                    'Predicted_SMILES': pred_smiles,
                    'Tanimoto_Score': tanimoto,
                    'Input_Complexity': input_complexity,
                    'Pred_Complexity': pred_complexity,
                    'Complexity_Ratio': complexity_ratio
                })
    def save_filtered_pairs(self, output_file):
        """Save filtered pairs to a new CSV file"""
        fieldnames = [
            'Input_SMILES', 
            'Predicted_SMILES', 
            'Tanimoto_Score',
            'Input_Complexity', 
            'Pred_Complexity', 
            'Complexity_Ratio'
        ]
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for pair in self.filtered_pairs:
                writer.writerow(pair)
        
        print(f"\nSaved {len(self.filtered_pairs)} filtered pairs to {output_file}")

def main():
    input_file = "output_analyzer/valid_smiles_pairs.csv"
    output_file = "output_analyzer/filtered_smiles_pairs.csv"
    
    try:
        # Initialize with complexity ratio threshold
        analyzer = SMILESAnalyzer(input_file, complexity_ratio_threshold=0.3)
        analyzer.filter_pairs()
        analyzer.save_filtered_pairs(output_file)
        
        print(f"\nFiltered SMILES pairs have been saved to: {output_file}")
        print(f"Number of filtered pairs: {len(analyzer.filtered_pairs)}")
        
        # Print some examples
        if analyzer.filtered_pairs:
            print("\nExample filtered pairs:")
            for pair in analyzer.filtered_pairs[:5]:
                print(f"\nInput SMILES: {pair['Input_SMILES']}")
                print(f"Predicted SMILES: {pair['Predicted_SMILES']}")
                print(f"Tanimoto Score: {pair['Tanimoto_Score']:.3f}")
                print(f"Complexity Scores: Input={pair['Input_Complexity']:.1f}, "
                      f"Pred={pair['Pred_Complexity']:.1f}")
                print(f"Complexity Ratio: {pair['Complexity_Ratio']:.2f}")
                print("-" * 50)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()