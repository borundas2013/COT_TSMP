from .property_prediction_model import PropertyPredictor
from .constants import Constants
from pathlib import Path
from rdkit import Chem

def load_predictor():
    """Load the trained predictor model"""
    root_dir = Path(__file__).parent
    model_dir = root_dir / Constants.MODEL_DIR
    return PropertyPredictor(model_path=str(model_dir))

def predict_properties(smiles1: str, smiles2: str, ratio_1: float, ratio_2: float) -> tuple:
    """Predict Er and Tg for a given SMILES pair"""
    predictor = load_predictor()
    return predictor.predict(smiles1, smiles2, ratio_1, ratio_2)


def reward_score(smiles1, smiles2, actual_tg, actual_er):
    
    # Get predictions
    try:
        if Chem.MolFromSmiles(smiles1) is None or Chem.MolFromSmiles(smiles2) is None:
            return 0.0, 0.0, 0.0
        pred_tg, pred_er = predict_properties(smiles1, smiles2,0.5,0.5)
    except Exception as e:
        print(f"Reward_Score Error in prediction: {e}")
        return 0.0, 0.0, 0.0
    
    # Calculate relative errors
    tg_error = abs(pred_tg - actual_tg) / abs(actual_tg)
    er_error = abs(pred_er - actual_er) / abs(actual_er)
    
    # Calculate individual scores (0 to 1)
    tg_score = max(0, 1 - tg_error)
    er_score = max(0, 1 - er_error)
    
    final_score = (tg_score + er_score)
    

    
    # Print detailed scores for debugging
    print(f"Predictions - Tg: {pred_tg:.2f}, Er: {pred_er:.2f}")
    print(f"Actuals    - Tg: {actual_tg:.2f}, Er: {actual_er:.2f}")
    print(f"Scores     - Tg: {tg_score:.3f}, Er: {er_score:.3f}")
    print(f"Final Score: {final_score:.3f}")
    
    return final_score, tg_score, er_score

if __name__ == "__main__":
    # Example usage
    smiles1 = 'CN(C)Cc1ccc(-c2ccc3cnc(Nc4ccc(C5CCN(CC(N)=O)CC5)cc4)nn23)cc1'
    smiles2 = 'CCCNC1OC1'
    actual_tg = 250.0  # example value
    actual_er = 300.0
    
    final_score,er_pred, tg_pred = reward_score(smiles1, smiles2, actual_tg, actual_er)
    print(f"\nPredictions for test pair:")
    print(f"Monomer 1: {smiles1}")
    print(f"Monomer 2: {smiles2}")
    print(f"ER reward score: {er_pred:.2f}")
    print(f"TG reward score: {tg_pred:.2f}") 