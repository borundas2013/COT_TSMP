from property_prediction_model import PropertyPredictor
from data_processor import DataProcessor
from constants import Constants
import os
from pathlib import Path

def main():
    # Get the root directory and setup paths
    root_dir = Path(__file__).parent  # Gets the directory containing train.py
    data_path = root_dir / 'data' / 'unique_smiles_Er.xlsx'
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
        predictor = PropertyPredictor(model_path=model_save_dir)
        
        print("Training model...")
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
        loaded_predictor = PropertyPredictor(model_path=model_save_dir)
        
        # Make predictions using loaded model
        print("\nTesting loaded model predictions...")
        for i in range(min(5, len(test_data['smiles1']))):
            er_pred, tg_pred = loaded_predictor.predict(
                test_data['smiles1'][i],
                test_data['smiles2'][i], 
                test_data['ratio_1'][i],
                test_data['ratio_2'][i]
            )
            print(f"\nLoaded model test pair {i+1}:")
            print(f"Monomer 1: {test_data['smiles1'][i]}")
            print(f"Monomer 2: {test_data['smiles2'][i]}")
            print(f"Actual Er: {test_data['er'][i]:.2f}, Predicted Er: {er_pred:.2f}")
            print(f"Actual Tg: {test_data['tg'][i]:.2f}, Predicted Tg: {tg_pred:.2f}")
        
        # Test predictions
        print("\nTesting predictions...")
        for i in range(min(5, len(test_data['smiles1']))):
            er_pred, tg_pred = predictor.predict(
                test_data['smiles1'][i],
                test_data['smiles2'][i],
                test_data['ratio_1'][i],
                test_data['ratio_2'][i]
            )
            print(f"\nTest pair {i+1}:")
            print(f"Monomer 1: {test_data['smiles1'][i]}")
            print(f"Monomer 2: {test_data['smiles2'][i]}")
            print(f"Actual Er: {test_data['er'][i]:.2f}, Predicted Er: {er_pred:.2f}")
            print(f"Actual Tg: {test_data['tg'][i]:.2f}, Predicted Tg: {tg_pred:.2f}")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")

if __name__ == "__main__":
    main() 