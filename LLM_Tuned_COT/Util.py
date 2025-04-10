import json
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
    file1_path = "LLM_Tuned_COT/Data/lat/training_conversations_large.jsonl"
    file2_path = "LLM_Tuned_COT/Data/lat/training_polymer_large_conversational.jsonl"
    output_path = "LLM_Tuned_COT/Data/lat/training_large_combined_conversational.jsonl"
    combine_json_files(file1_path, file2_path, output_path)

    file1_path = "LLM_Tuned_COT/Data/lat/validation_conversations_large.jsonl"
    file2_path = "LLM_Tuned_COT/Data/lat/validation_polymer_large_conversational.jsonl"
    output_path = "LLM_Tuned_COT/Data/lat/validation_large_combined_conversational.jsonl"
    combine_json_files(file1_path, file2_path, output_path)
