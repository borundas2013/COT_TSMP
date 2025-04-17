import os
import pandas as pd

def combine_csv_files(folder_path, output_file):
    """
    Read all CSV files from a folder, combine them, and save to one CSV file
    
    Args:
        folder_path (str): Path to folder containing CSV files
        output_file (str): Path where combined CSV will be saved
    """
    # List to store all dataframes
    all_dfs = []
    
    try:
        # Iterate through all files in folder
        for filename in os.listdir(folder_path):
            if filename.endswith('.csv'):
                file_path = os.path.join(folder_path, filename)
                # Read CSV file
                df = pd.read_csv(file_path)
                all_dfs.append(df)
                print(f"Read {filename}")
        
        if not all_dfs:
            print("No CSV files found in the specified folder")
            return
            
        # Combine all dataframes
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Save combined dataframe
        combined_df.to_csv(output_file, index=False)
        
        print(f"\nCombination Statistics:")
        print(f"Number of files combined: {len(all_dfs)}")
        print(f"Total records in combined file: {len(combined_df)}")
        print(f"Combined CSV saved to: {output_file}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

def deduplicate_csv(input_file, output_file):
    """
    Read CSV file, remove duplicates based on SMILE1 and SMILE2 columns, and save to new file
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path where deduplicated CSV will be saved
    """
    try:
        # Read CSV file
        df = pd.read_csv(input_file)
        original_count = len(df)
        
        # Remove duplicates based on SMILE1 and SMILE2 columns
        deduplicated_df = df.drop_duplicates(subset=['SMILE1', 'SMILE2'])
        
        # Reset index
        deduplicated_df = deduplicated_df.reset_index(drop=True)
        
        # Save deduplicated dataframe
        deduplicated_df.to_csv(output_file, index=False)
        
        # Print statistics
        print(f"\nDeduplication Statistics for {input_file}:")
        print(f"Original records: {original_count}")
        print(f"Records after deduplication: {len(deduplicated_df)}")
        print(f"Number of duplicates removed: {original_count - len(deduplicated_df)}")
        print(f"Deduplicated CSV saved to: {output_file}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

def analyze_temperatures(input_file):
    """
    Analyze temperature distribution in the CSV file
    
    Args:
        input_file (str): Path to input CSV file
    """
    try:
        # Read CSV file
        df = pd.read_csv(input_file)
        
        # Get temperature counts
        temp_counts = df['Temperature'].value_counts()
        
        # Get most common temperature
        most_common_temp = temp_counts.index[0]
        most_common_count = temp_counts.iloc[0]
        
        # Print statistics
        print(f"\nTemperature Analysis for {input_file}:")
        print(f"Most common temperature: {most_common_temp}")
        print(f"Number of occurrences: {most_common_count}")
        print("\nTemperature distribution:")
        for temp, count in temp_counts.items():
            print(f"Temperature {temp}: {count} occurrences")
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")



if __name__ == "__main__":
    # Example usage
    # folder_path = "All_together/small/reactive"
    # output_file = "All_together/combined_small_reactive.csv"
    # combine_csv_files(folder_path, output_file)

    # folder_path = "All_together/large/reactive"
    # output_file = "All_together/combined_large_reactive.csv"
    # combine_csv_files(folder_path, output_file)

    # folder_path = "All_together/small/nonreactive"
    # output_file = "All_together/combined_small_non_reactive.csv"
    # combine_csv_files(folder_path, output_file)

    # folder_path = "All_together/large/nonreactive"
    # output_file = "All_together/combined_large_non_reactive.csv"
    # combine_csv_files(folder_path, output_file)

    # input_file = "All_together/combined_large_reactive.csv"
    # output_file = "All_together/deduplicated_large_reactive.csv"
    # deduplicate_csv(input_file, output_file)
    
    # input_file = "All_together/combined_large_non_reactive.csv"
    # output_file = "All_together/deduplicated_large_non_reactive.csv"
    # deduplicate_csv(input_file, output_file)

    # input_file = "All_together/combined_small_reactive.csv"
    # output_file = "All_together/deduplicated_small_reactive.csv"
    # deduplicate_csv(input_file, output_file)    
    
    # input_file = "All_together/combined_small_non_reactive.csv"
    # output_file = "All_together/deduplicated_small_non_reactive.csv"
    # deduplicate_csv(input_file, output_file)

    input_file = "All_together/combined_large_reactive.csv"
    analyze_temperatures(input_file)
    input_file = "All_together/combined_small_reactive.csv"
    analyze_temperatures(input_file)


  
