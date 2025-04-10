WEIGHT_PATH = "Two_Monomers_Group/RLHF/models/weights"
WEIGHT_PATH_NAME = "best_model.weights.h5"
CONFIG_PATH = "Two_Monomers_Group/RLHF/models/weights"
DATA_PATH = "Two_Monomers_Group/data"

FEEDBACK_COLLECT_EPOCH = 10
PRETRAINED_MODEL_PATH = "Two_Monomers_Group/pretrained_model"
PRETRAINED_MODEL_NAME = "saved_models_rl_gpu_3"
DATA_FILE_NAME = "smiles_orginal.xlsx"
GENERATOR_MODEL_NAME = "group_based_rl_model_noise_s"
GENERATOR_MODEL_PATH = "Two_Monomers_Group/RLHF/Generator/models/"
FEEDBACK_COLLECTOR_PATH = "Two_Monomers_Group/RLHF/FeedbackCollector/feedback_data"
FEEDBACK_COLLECTOR_FILE_NAME = "feedback_data.json"

DATA_WITH_SCORE = "Two_Monomers_Group/data/data_with_score.json"

GENERATED_MOLECULES_PATH = "Two_Monomers_Group/RLHF/generated_molecules"
GENERATED_MOLECULES_FILE_NAME = "valid_generations.json"


SAMPLES_WITH_REWARD_PATH = "Two_Monomers_Group/RLHF/samples_with_reward"
SAMPLES_WITH_REWARD_FILE_NAME = "samples_with_reward.json"

# PPO Training Constants
# PPO_EPOCHS = 50
# PPO_STEPS_PER_EPOCH = 50
# PPO_BATCH_SIZE = 32
# PPO_NUM_SAMPLES = 10#1024
# PPO_MAX_ATTEMPT =  5
# PPO_SAVE_FREQ = 5


#DEMO Constants
PPO_EPOCHS = 1
PPO_BATCH_SIZE = 2
PPO_SAVE_FREQ = 1
NUM_SAMPLES = 1

# PPO Optimization Hyperparameters
PPO_LEARNING_RATE = 1e-5
PPO_CLIP_EPSILON = 0.2
PPO_VALUE_COEF = 0.5
PPO_ENTROPY_COEF = 0.01

TEST_SAMPLES = [
    {
        'smiles1': 'CC(=O)NCCC(=O)O',  # N-acetyl-beta-alanine
        'smiles2': 'CC(N)CC(=O)O',      # beta-aminobutyric acid
        'group1': 'COOH',
        'group2': 'NH2',
        'input_smiles1': 'CC(=O)NCCC(=O)O',
        'input_smiles2': 'CC(N)CC(=O)O'
    },
    {
        'smiles1': 'NCCC(=O)O',         # beta-alanine
        'smiles2': 'NCC(=O)O',          # glycine
        'group1': 'COOH',
        'group2': 'NH2',
        'input_smiles1': 'NCCC(=O)O',
        'input_smiles2': 'NCC(=O)O'
    },
    {
        'smiles1': 'CC(O)C(=O)O',       # lactic acid
        'smiles2': 'OCC(=O)O',          # glycolic acid
        'group1': 'COOH',
        'group2': 'OH',
        'input_smiles1': 'CC(O)C(=O)O',
        'input_smiles2': 'OCC(=O)O'
    },
    {
        'smiles1': 'CC(N)C(=O)O',       # alanine
        'smiles2': 'NCC(=O)O',          # glycine
        'group1': 'COOH',
        'group2': 'NH2',
        'input_smiles1': 'CC(N)C(=O)O',
        'input_smiles2': 'NCC(=O)O'
    },
    {
        'smiles1': 'OCC(O)C(=O)O',      # glyceric acid
        'smiles2': 'CC(O)C(=O)O',       # lactic acid
        'group1': 'COOH',
        'group2': 'OH',
        'input_smiles1': 'OCC(O)C(=O)O',
        'input_smiles2': 'CC(O)C(=O)O'
    },
    {
        'smiles1': 'CC(=O)O',           # acetic acid
        'smiles2': 'CCC(=O)O',          # propionic acid
        'group1': 'COOH',
        'group2': 'COOH',
        'input_smiles1': 'CC(=O)O',
        'input_smiles2': 'CCC(=O)O'
    },
    {
        'smiles1': 'NCCCCC(=O)O',       # 5-aminopentanoic acid
        'smiles2': 'NCCCC(=O)O',        # 4-aminobutanoic acid
        'group1': 'COOH',
        'group2': 'NH2',
        'input_smiles1': 'NCCCCC(=O)O',
        'input_smiles2': 'NCCCC(=O)O'
    },
    {
        'smiles1': 'CC(N)CC(=O)O',      # beta-aminobutyric acid
        'smiles2': 'CC(N)C(=O)O',       # alanine
        'group1': 'COOH',
        'group2': 'NH2',
        'input_smiles1': 'CC(N)CC(=O)O',
        'input_smiles2': 'CC(N)C(=O)O'
    },
    {
        'smiles1': 'OCC(N)C(=O)O',      # serine
        'smiles2': 'CC(O)C(=O)O',       # lactic acid
        'group1': 'COOH',
        'group2': 'OH',
        'input_smiles1': 'OCC(N)C(=O)O',
        'input_smiles2': 'CC(O)C(=O)O'
    },
    {
        'smiles1': 'CC(O)CC(=O)O',      # beta-hydroxybutyric acid
        'smiles2': 'OCC(O)C(=O)O',      # glyceric acid
        'group1': 'COOH',
        'group2': 'OH',
        'input_smiles1': 'CC(O)CC(=O)O',
        'input_smiles2': 'OCC(O)C(=O)O'
    }
    ]
TEMPERATURES = [0.4,0.6]#][0.8, 1.0, 1.2]




