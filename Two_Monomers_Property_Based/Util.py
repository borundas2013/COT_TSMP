import pandas as pd

def get_unique_smiles(csv_path):
    try:
        # Read Excel file
        df = pd.read_excel(csv_path)
        
        # Check if required columns exist
        required_cols = ['SMILES', 'Er', 'Tg']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in Excel file")
        
        # Initialize lists for storing data
        smiles1_list = []
        smiles2_list = []
        er_list = []
        tg_list = []
        
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
            except:
                print(f"Skipping malformed SMILES pair: {row['SMILES']}")
                continue
                
        return smiles1_list, smiles2_list, er_list, tg_list
        
    except Exception as e:
        print(f"Error processing Excel file: {str(e)}")
        raise

if __name__ == "__main__":
    smiles1_list, smiles2_list, er_list, tg_list = get_unique_smiles("Two_Monomers_Property_Based/Dataset/unique_smiles_Er.xlsx")
    print(len(smiles1_list),len(smiles2_list),len(er_list),len(tg_list))
