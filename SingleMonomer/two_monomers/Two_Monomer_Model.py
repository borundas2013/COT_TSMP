import tensorflow as tf
import json
from Two_Monomer_Generation import *
from Data_Process_with_prevocab import *
from tensorflow.keras.losses import Loss

@tf.function
def functional_group_loss(y_true, y_pred, monomer1, monomer2):
    """
    Calculate loss based on functional group requirements
    """
    # Standard cross-entropy loss
    base_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    
    # Group SMARTS patterns
    epoxy_smarts = "[OX2]1[CX3][CX3]1"
    imine_smarts = "[NX2]=[CX3]"
    vinyl_smarts = "C=C"
    thiol_smarts = "CCS"
    acryl_smarts = "C=C(C=O)"
    
    # Calculate group presence (1 if group appears >=2 times, 0 otherwise)
    epoxy_m1 = calculate_group_presence(monomer1, epoxy_smarts)
    imine_m1 = calculate_group_presence(monomer1, imine_smarts)
    vinyl_m1 = calculate_group_presence(monomer1, vinyl_smarts)
    thiol_m1 = calculate_group_presence(monomer1, thiol_smarts)
    acryl_m1 = calculate_group_presence(monomer1, acryl_smarts)
    
    epoxy_m2 = calculate_group_presence(monomer2, epoxy_smarts)
    imine_m2 = calculate_group_presence(monomer2, imine_smarts)
    vinyl_m2 = calculate_group_presence(monomer2, vinyl_smarts)
    thiol_m2 = calculate_group_presence(monomer2, thiol_smarts)
    acryl_m2 = calculate_group_presence(monomer2, acryl_smarts)
    
    # Check valid combinations
    valid_combination = tf.cast(
        # Epoxy-Imine combination
        ((epoxy_m1 and imine_m2) or (epoxy_m2 and imine_m1)) or
        # Vinyl-Thiol combination
        ((vinyl_m1 and thiol_m2) or (vinyl_m2 and thiol_m1)) or
        # Vinyl-Vinyl combination
        (vinyl_m1 and vinyl_m2) or
        # Vinyl-Acrylic combination
        ((vinyl_m1 and acryl_m2) or (vinyl_m2 and acryl_m1)),
        tf.float32
    )
    
    # Penalty for invalid combinations
    group_penalty = 1.0 - valid_combination
    
    # Combine losses with weighting
    alpha = 0.7  # Weight for base loss
    beta = 0.3   # Weight for group penalty
    
    total_loss = alpha * base_loss + beta * group_penalty
    
    return total_loss

class CustomMonomerLoss(Loss):
    def __init__(self, name="custom_monomer_loss"):
        super().__init__(name=name)
    
    def call(self, y_true, y_pred):
        # Extract monomer sequences from y_true and y_pred
        # This needs to be adapted based on your exact model architecture
        monomer1 = y_true[:, :, 0]  # Adjust slicing based on your data structure
        monomer2 = y_pred[:, :, 0]  # Adjust slicing based on your data structure
        
        return functional_group_loss(y_true, y_pred, monomer1, monomer2)
    
