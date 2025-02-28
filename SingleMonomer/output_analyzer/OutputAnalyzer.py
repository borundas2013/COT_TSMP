import os
import csv
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem

class OutputAnalyzer:
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.valid_pairs = []  # Will store tuples of (input_smiles, pred_smiles, tanimoto)
        self.invalid_smiles = []

    def calculate_tanimoto(self, smiles1, smiles2):
        """Calculate Tanimoto similarity between two SMILES strings"""
        try:
            mol1 = Chem.MolFromSmiles(smiles1)
            mol2 = Chem.MolFromSmiles(smiles2)
            if mol1 and mol2:
                fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2)
                fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2)
                return DataStructs.TanimotoSimilarity(fp1, fp2)
        except:
            pass
        return 0.0

    def analyze_predictions(self):
        """
        Reads the training prediction file and analyzes SMILES predictions
        """
        if not os.path.exists(self.log_file_path):
            raise FileNotFoundError(f"Log file not found at {self.log_file_path}")

        current_input = None
        with open(self.log_file_path, 'r') as file:
            for line in file:
                if line.startswith("Input SMILES:"):
                    current_input = line.split("Input SMILES:")[1].strip()
                elif line.startswith("Predicted SMILES:") and current_input:
                    pred_smiles = line.split("Predicted SMILES:")[1].strip()
                    
                    # Skip empty or None SMILES
                    if not pred_smiles or pred_smiles.lower() == 'none':
                        continue
                    
                    # Validate SMILES using RDKit
                    try:
                        mol = Chem.MolFromSmiles(pred_smiles)
                        if mol is not None:
                            # Calculate Tanimoto score
                            tanimoto = self.calculate_tanimoto(current_input, pred_smiles)
                            self.valid_pairs.append((current_input, pred_smiles, tanimoto))
                        else:
                            self.invalid_smiles.append(pred_smiles)
                    except:
                        self.invalid_smiles.append(pred_smiles)

    def save_valid_pairs(self, output_file):
        """
        Saves valid SMILES pairs and their Tanimoto scores to a CSV file
        
        Args:
            output_file (str): Path to the output file
        """
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Input_SMILES', 'Predicted_SMILES', 'Tanimoto_Score'])
            for input_smiles, pred_smiles, tanimoto in self.valid_pairs:
                writer.writerow([input_smiles, pred_smiles, f"{tanimoto:.4f}"])

    def get_statistics(self):
        """
        Returns statistics about the analyzed predictions
        """
        total = len(self.valid_pairs) + len(self.invalid_smiles)
        valid_percentage = (len(self.valid_pairs) / total * 100) if total > 0 else 0
        
        # Calculate average Tanimoto score
        avg_tanimoto = 0.0
        if self.valid_pairs:
            avg_tanimoto = sum(t for _, _, t in self.valid_pairs) / len(self.valid_pairs)
        
        return {
            'total_predictions': total,
            'valid_smiles': len(self.valid_pairs),
            'invalid_smiles': len(self.invalid_smiles),
            'valid_percentage': valid_percentage,
            'average_tanimoto': avg_tanimoto
        }

    def get_valid_pairs(self):
        """
        Returns the list of valid SMILES pairs
        """
        return self.valid_pairs

    def get_invalid_smiles(self):
        """
        Returns the list of invalid SMILES
        """
        return self.invalid_smiles
    
    

def main():
    # Example usage
    log_file_path = "output_analyzer/training_predictions4.log"
    output_file = "output_analyzer/valid_smiles_pairs4.csv"
    
    try:
        analyzer = OutputAnalyzer(log_file_path)
        analyzer.analyze_predictions()
        
        # Save valid SMILES pairs to CSV
        analyzer.save_valid_pairs(output_file)
        print(f"\nValid SMILES pairs have been saved to: {output_file}")
        
        # Get and print statistics
        stats = analyzer.get_statistics()
        print("\nAnalysis Results:")
        print("-" * 50)
        print(f"Total predictions analyzed: {stats['total_predictions']}")
        print(f"Valid SMILES found: {stats['valid_smiles']}")
        print(f"Invalid SMILES found: {stats['invalid_smiles']}")
        print(f"Percentage of valid SMILES: {stats['valid_percentage']:.2f}%")
        print(f"Average Tanimoto similarity: {stats['average_tanimoto']:.4f}")
        
        # Print some example valid pairs
        if analyzer.valid_pairs:
            print("\nFirst 5 valid SMILES pairs examples:")
            for i, (input_s, pred_s, tanimoto) in enumerate(analyzer.valid_pairs[:5]):
                print(f"{i+1}. Input: {input_s}")
                print(f"   Pred:  {pred_s}")
                print(f"   Tanimoto: {tanimoto:.4f}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()

