import json
import pandas as pd
import os
from dual_smile_process import *
from template import *
import json

def load_dataset():
    """Load dataset from Data folder and return all columns"""
    data_path = os.path.join('LLM_Tuned_COT', 'Data', 'unique_smiles_Er.csv')
    data_path2 = os.path.join('LLM_Tuned_COT', 'Data', 'smiles.xlsx')
    
    try:
        monomer1, monomer2, er, tg = process_dual_monomer_data(data_path, data_path2)
        return monomer1, monomer2
    except FileNotFoundError:
        print(f"Error: Could not find dataset at {data_path}")
        return None
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None
    
def filter_samples():
    dataframe = pd.read_csv('LLM_Tuned_COT/Output/all_reactive_pairs_combined.csv')
    smiles1 = dataframe['SMILE1']
    smiles2 = dataframe['SMILE2']
    temperature = dataframe['Temperature']
    # Group by temperature and count occurrences
    temp_counts = temperature.value_counts()
    
    # Sort by count in descending order
    temp_counts_sorted = temp_counts.sort_values(ascending=False)
    
    print("\nTemperature distribution:")
    print("------------------------")
    for temp, count in temp_counts_sorted.items():
        print(f"Temperature {temp}°C: {count} samples")
    
    # Get the most common temperature
    most_common_temp = temp_counts_sorted.index[0]
    most_common_count = temp_counts_sorted.iloc[0]
    
    print(f"\nMost common temperature: {most_common_temp}°C with {most_common_count} samples")
    
    return temp_counts_sorted

    

def check_unique_samples():
    monomer1, monomer2 = load_dataset()
    dataframe = pd.read_csv('LLM_Tuned_COT/Output/all_reactive_pairs_combined.csv')
    smiles1 = dataframe['SMILE1']
    smiles2 = dataframe['SMILE2']
    found_count = 0
    unique_count = 0
    combined=zip(monomer1,monomer2)
    for i in range(len(smiles1)):
        if smiles1[i] in monomer1 and smiles2[i] in monomer2:
            found_count += 1
        elif smiles1[i] in monomer2 and smiles2[i] in monomer1:
            found_count += 1
        else:
            #print(f"Found {smiles1[i]} and {smiles2[i]}")
            unique_count += 1

    print(f"Found {found_count} samples out of {len(smiles1)}")
    print(f"Unique {unique_count} samples out of {len(smiles1)}")
    return found_count, unique_count

    
def combine_json_files(file1_path, file2_path, output_path):
    
    # Read and combine JSONL files line by line
    combined_data = []
    
    # Read first JSONL file
    with open(file1_path, 'r') as f1:
        for line in f1:
            combined_data.append(json.loads(line.strip()))
            
    # Read second JSONL file
    with open(file2_path, 'r') as f2:
        for line in f2:
            combined_data.append(json.loads(line.strip()))
    
    # Write combined data to output JSONL file
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in combined_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return combined_data

if __name__ == "__main__":
    #check_unique_samples()
    filter_samples()
    # file1_path = "LLM_Tuned_COT/Data/lat/training_conversations_large.jsonl"
    # file2_path = "LLM_Tuned_COT/Data/lat/training_polymer_large_conversational.jsonl"
    # output_path = "LLM_Tuned_COT/Data/lat/training_large_combined_conversational.jsonl"
    # combine_json_files(file1_path, file2_path, output_path)

    # file1_path = "LLM_Tuned_COT/Data/lat/validation_conversations_large.jsonl"
    # file2_path = "LLM_Tuned_COT/Data/lat/validation_polymer_large_conversational.jsonl"
    # output_path = "LLM_Tuned_COT/Data/lat/validation_large_combined_conversational.jsonl"
    # combine_json_files(file1_path, file2_path, output_path)
