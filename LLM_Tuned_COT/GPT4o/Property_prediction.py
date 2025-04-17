import pandas as pd
import numpy as np
from dual_smile_process import process_dual_monomer_data
from Property_Prediction.predict import predict_properties

def load_data(file_path):
    df = pd.read_csv(file_path)
    print(df.columns)
    return df, df['SMILE1'], df['SMILE2']


def analyze_predictions_file(self):
    """
    Load predictions CSV file, filter based on Er > Tg criterion, and save results
    """
    try:
        # Load predictions
        predictions_df = pd.read_csv('LLM_Tuned_COT/Output/predictions.csv')
        
        # Create mask for Er > Tg
        valid_mask = predictions_df['Predicted_Er'] > predictions_df['Predicted_Tg']
        
        # Split into valid and invalid predictions
        valid_predictions = predictions_df[valid_mask].copy()
        invalid_predictions = predictions_df[~valid_mask].copy()
        
        # For valid predictions, keep only the max Er for each Serial_No
        max_er_predictions = valid_predictions.loc[
            valid_predictions.groupby('Serial_No')['Predicted_Er'].idxmax()
        ]
        
        # Sort by Serial_No
        max_er_predictions = max_er_predictions.sort_values('Serial_No')
        invalid_predictions = invalid_predictions.sort_values('Serial_No')
        
        # Save to separate CSV files
        max_er_predictions.to_csv('LLM_Tuned_COT/Output/valid_predictions_max_er.csv', index=False)
        invalid_predictions.to_csv('LLM_Tuned_COT/Output/invalid_predictions.csv', index=False)
        
        # Print statistics
        print("\nPrediction Analysis Statistics:")
        print(f"Total predictions: {len(predictions_df)}")
        print(f"Valid predictions (Er > Tg): {len(valid_predictions)}")
        print(f"Unique combinations with max Er: {len(max_er_predictions)}")
        print(f"Invalid predictions (Er ≤ Tg): {len(invalid_predictions)}")
        
        # Print sample of valid predictions
        print("\nSample of Valid Predictions (Max Er):")
        print(max_er_predictions[['Serial_No', 'SMILE1', 'SMILE2', 'Predicted_Er', 
                                'Predicted_Tg', 'Ratio_1', 'Ratio_2']].head())
        
        return max_er_predictions, invalid_predictions
        
    except Exception as e:
        print(f"Error processing predictions: {str(e)}")
        return None, None
    
def get_max_er_predictions():
    """
    For each SMILE1-SMILE2 combination, keep only the prediction with the highest Er value
    """
    try:
        # Load predictions
        predictions_df = pd.read_csv('LLM_Tuned_COT/Output/predictions_small_model.csv')
        
        # Group by SMILE1 and SMILE2, keep the row with maximum Er for each group
        max_er_predictions = predictions_df.loc[
            predictions_df.groupby(['SMILE1', 'SMILE2'])['Predicted_Er'].idxmax()
        ]
        
        # Sort by Serial_No for better readability
        max_er_predictions = max_er_predictions.sort_values('Serial_No')
        
        # Save results
        max_er_predictions.to_csv('LLM_Tuned_COT/Output/max_er_predictions_small_model.csv', index=False)
        
        # Print statistics
        print("\nMax Er Prediction Analysis:")
        print(f"Total unique SMILE combinations: {len(max_er_predictions)}")
        print(f"\nSummary Statistics:")
        print(f"Average Max Er: {max_er_predictions['Predicted_Er'].mean():.2f}")
        print(f"Average Tg at Max Er: {max_er_predictions['Predicted_Tg'].mean():.2f}")
        print(f"Average Ratio_1 for Max Er: {max_er_predictions['Ratio_1'].mean():.2f}")
        
        # Distribution of ratios that give maximum Er
        print("\nDistribution of Ratio_1 values giving maximum Er:")
        ratio_dist = max_er_predictions['Ratio_1'].value_counts().sort_index()
        for ratio, count in ratio_dist.items():
            print(f"Ratio_1 = {ratio:.1f}: {count} combinations")
        
        # Sample output
        print("\nSample of predictions (first 5 combinations):")
        print(max_er_predictions[['Serial_No', 'SMILE1', 'SMILE2', 
                                'Predicted_Er', 'Predicted_Tg', 
                                'Ratio_1', 'Ratio_2']].head().to_string())
        
        return max_er_predictions
        
    except Exception as e:
        print(f"Error processing predictions: {str(e)}")
        return None
    

    




if __name__ == "__main__":
    df, smiles1, smiles2 = load_data('LLM_Tuned_COT/Output/all_reactive_pairs_small_model.csv')
    # Create lists to store the predictions
    predictions = []
 
    for i in range(len(smiles1)):
        for ratio_1 in np.arange(0.1, 1.0, 0.1):
            ratio_2 = 1 - ratio_1
            print(f"ratio_1: {ratio_1:.1f}, ratio_2: {ratio_2:.1f}")
            er,tg = predict_properties(smiles1[i], smiles2[i], ratio_1, ratio_2)
            predictions.append({
            'Serial_No': i+1,
            'SMILE1': smiles1[i],
            'SMILE2': smiles2[i], 
            'Predicted_Tg': round(tg, 2),
            'Predicted_Er': round(er, 2),
            'Ratio_1': ratio_1,
            'Ratio_2': ratio_2
            })
    
    # Convert predictions to dataframe and save
    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv('LLM_Tuned_COT/Output/predictions_small_model.csv', index=False)

    #analyze_predictions_file('LLM_Tuned_COT/Output/predictions.csv')
    get_max_er_predictions()

