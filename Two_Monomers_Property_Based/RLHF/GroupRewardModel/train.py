
from data_processor import DataProcessor
from constants import Constants
import os
from pathlib import Path
from Group_Reward import GroupRewardScorePredictor
import pandas as pd
def main():
    # Get the root directory and setup paths
    root_dir = Path(__file__).parent  # Gets the directory containing train.py
    data_path = 'Two_Monomers_Group/data/' +  'data_with_score.csv'
    model_save_dir = root_dir / 'saved_models'
    
    # Create directories if they don't exist
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from: {data_path}")
    print(f"Models will be saved to: {model_save_dir}")
    
    try:
        # Load and process data
        print("Loading data...")
        data = DataProcessor.load_data(data_path)
        if data is None:
            raise ValueError("Failed to load data")
        
        print("Splitting data...")
        train_data, test_data = DataProcessor.split_data(data)
        
        # Initialize and train model
        print("Initializing model...")
        predictor = GroupRewardScorePredictor(model_path=model_save_dir)
        
        # print("Training model...")
        predictor.train(
            train_data,
            validation_split=Constants.VALIDATION_SPLIT,
            epochs=Constants.DEFAULT_EPOCHS
        )
        
        # Save model
        print("Saving model...")
        predictor.save_models(model_save_dir)
        # Load the saved models
        print("\nLoading saved models...")
        loaded_predictor = GroupRewardScorePredictor(model_path=model_save_dir)
        
        # Make predictions using loaded model
        print("\nTesting loaded model predictions...")
        for i in range(min(5, len(test_data['smiles1']))):
            score_pred = loaded_predictor.predict(
                test_data['smiles1'][i],
                test_data['smiles2'][i], 
                test_data['group1'][i],
                test_data['group2'][i]
            )
            print(f"\nLoaded model test pair {i+1}:")
            print(f"Monomer 1: {test_data['smiles1'][i]}")
            print(f"Monomer 2: {test_data['smiles2'][i]}")
            print(f"Actual score: {test_data['score'][i]:.2f}, Predicted score: {score_pred:.2f}")
            
        
        # Test predictions
        print("\nTesting predictions...")
        result_list = []
        for i in range(len(test_data['smiles1'])):
            score_pred = predictor.predict(
                test_data['smiles1'][i],
                test_data['smiles2'][i],
                test_data['group1'][i],
                test_data['group2'][i]
            )

            result_dict = {
            'Monomer1': test_data['smiles1'][i],
            'Monomer2': test_data['smiles2'][i],
            'Group1': test_data['group1'][i],
            'Group2': test_data['group2'][i],
            'Actual_Score': round(test_data['score'][i], 2),
            'Predicted_Score': round(score_pred, 2),
            'Score_Difference': round(abs(test_data['score'][i] - score_pred), 2)
        }
            result_list.append(result_dict)

        # Save results to CSV
        result_df = pd.DataFrame(result_list)
        result_df.to_csv('Two_Monomers_Group/data/result.csv', index=False)

            
            
            
        
    except Exception as e:
        print(f"Error during training: {str(e)}")

if __name__ == "__main__":
    main() 