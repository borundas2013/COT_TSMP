import os
import json
from datetime import datetime
from property_based_model import *

def save_model(model, model_params, save_dir=""):
    """Save the model weights and parameters"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save weights with correct extension
    weights_path = os.path.join(save_dir, "weights_model.weights.h5")
    model.save_weights(weights_path)
    
    # Save parameters
    params_path = os.path.join(save_dir, "params_model.json")
    with open(params_path, 'w') as f:
        json.dump(model_params, f)
    
    return weights_path, params_path

def load_model(weights_path, params_path, pretrained_model):
    """Load saved weights into a new model instance"""
    try:
        # Load parameters
        with open(params_path, 'r') as f:
            params = json.load(f)
        
        # Create new model instance
        new_model = create_group_relationship_model(
            pretrained_model=pretrained_model,
            max_length=params['max_length'],
            vocab_size=params['vocab_size']
        )
        
        # Load weights
        new_model.load_weights(weights_path)
        
        print(f"Weights loaded from: {weights_path}")
        print(f"Parameters loaded from: {params_path}")
        
        return new_model, params
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None
