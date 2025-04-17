class Constants:
    # Model parameters
    FEATURE_SIZE = 200
    FINGERPRINT_RADIUS = 2
    
    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    DEFAULT_EPOCHS = 100
    VALIDATION_SPLIT = 0.2
    
    # Model architecture
    LAYER_SIZES = [512, 256, 128]
    DROPOUT_RATE = 0.2
    
    # File paths
    MODEL_DIR = 'saved_models'
    ER_MODEL_NAME = 'er_model'
    TG_MODEL_NAME = 'tg_model'
    FEATURE_SCALER_NAME = 'feature_scaler.pkl'
    ER_SCALER_NAME = 'er_scaler.pkl'
    TG_SCALER_NAME = 'tg_scaler.pkl' 