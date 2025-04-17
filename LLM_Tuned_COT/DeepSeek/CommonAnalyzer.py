import pandas as pd

def combine_and_deduplicate_csv(file1_path, file2_path, output_path):
    try:
        # Load both CSV files
        df1 = pd.read_csv(file1_path)
        df2 = pd.read_csv(file2_path)
        
        # Combine the dataframes
        combined_df = pd.concat([df1, df2], ignore_index=True)
        
        # Remove duplicates based on SMILE1 and SMILE2 columns
        unique_df = combined_df.drop_duplicates(subset=['SMILE1', 'SMILE2'])
        
        # Reset index
        unique_df = unique_df.reset_index(drop=True)
        
        # Save to new CSV file
        unique_df.to_csv(output_path, index=False)
        
        print(f"\nDeduplication Statistics:")
        print(f"Records in first file: {len(df1)}")
        print(f"Records in second file: {len(df2)}")
        print(f"Total combined records: {len(combined_df)}")
        print(f"Records after deduplication: {len(unique_df)}")
        print(f"Number of duplicates removed: {len(combined_df) - len(unique_df)}")
        
        return unique_df
        
    except Exception as e:
        print(f"Error processing files: {str(e)}")
        return None
    

if __name__ == "__main__":
    combine_and_deduplicate_csv('DeepSeek/Output/small_model/all_reactive_pairs.csv', 'DeepSeek/Output/large_model/all_reactive_pairs.csv', 'DeepSeek/Output/all_reactive_pairs_combined.csv')

