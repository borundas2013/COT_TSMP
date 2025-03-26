import os
import tensorflow as tf
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from Constants import *
from Data_Process_with_prevocab import *
from LoadPreTrainedModel import *
from pretrained_weights import *
from saveandload import *
from dual_smile_process import *
from validation_prediction import *
from group_based_model import create_group_relationship_model
from pathlib import Path
from sample_generator import *
from losswithReward import CombinedLoss
from NewModelApp1 import *
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
print(tf.config.list_physical_devices('GPU'))

def get_project_root():
    """Get the path to the project root directory"""
    current_file = Path(__file__).resolve()
    return current_file.parent



if __name__ == "__main__":
    root_dir = get_project_root()
    
    # Setup paths
    save_dir_abs = os.path.join(root_dir, "pretrained_model", "saved_models_rl_gpu_3")
    file_path = os.path.join(root_dir, 'Data', "smiles.xlsx")
    try:
        pretrained_model, smiles_vocab, model_params = load_and_retrain(save_dir=save_dir_abs)

        prediction_model = create_model(model_params['max_length'], len(smiles_vocab),pretrained_model)
        #weights_path = os.path.join(root_dir, "group_based_rl_model", "weights_model.weights.h5")
        prediction_model.load_weights('Two_Monomers_Group/Model_prediction_testing/group_based_rl_model_n1/weights_model.weights.h5')
            
            # Generate example predictions
        monomer1_list, monomer2_list = process_dual_monomer_data(file_path)
        group_combinations = [["C=C", "C=C(C=O)"], ["C=C", "CCS"], ["C1OC1", "NC"], ["C=C", "OH"]]
        selected_groups = random.choice(group_combinations)

        smiles_1 = []
        smiles_2 = []

        for index in range(len(monomer1_list)):
            mol = Chem.MolFromSmiles(monomer1_list[index])
            mol2 = Chem.MolFromSmiles(monomer2_list[index])
            pattern = Chem.MolFromSmarts(selected_groups[0])
            pattern2 = Chem.MolFromSmarts(selected_groups[1])
            if mol is not None and mol2 is not None:
                if len(mol.GetSubstructMatches(pattern)) >= 2 and len(mol2.GetSubstructMatches(pattern2)) >= 2:
                    smiles_1.append(monomer1_list[index])
                    smiles_2.append(monomer2_list[index])

        print('Selected groups: ',selected_groups)
        print('Number of smiles 1: ',len(smiles_1))
        print('Number of smiles 2: ',len(smiles_2))

        # smiles1 = random.choice(smiles_1)
        # smiles2 = random.choice(smiles_2)

        for i in range(len(smiles_1)):
            smiles1 = smiles_1[i]
            smiles2 = smiles_2[i]
            valid_pairs = generate_monomer_pair_with_temperature(
            model=prediction_model,
            input_smiles=[smiles1, smiles2],
            desired_groups=selected_groups,
            vocab=smiles_vocab,
            max_length=model_params['max_length'],
            temperatures=[1.5, 1.2, 1.0, 0.8,0.6,0.4],
            group_smarts1=selected_groups[0],
            group_smarts2=selected_groups[1],
            add_noise=False
            )

            valid_pairs_noise = generate_monomer_pair_with_temperature(
            model=prediction_model,
            input_smiles=[smiles1, smiles2],
            desired_groups=selected_groups,
            vocab=smiles_vocab,
            max_length=model_params['max_length'],
            temperatures=[1.5, 1.2, 1.0, 0.8,0.6,0.4],
            group_smarts1=selected_groups[0],
            group_smarts2=selected_groups[1],
            add_noise=True
            )

            for pair in valid_pairs:
                print(f"Monomer 1: {pair[0]}")
                print(f"Monomer 2: {pair[1]}")
                print("-" * 40)
            print("-" * 40)
            for pair in valid_pairs_noise:
                print(f"Monomer 1: {pair[0]}")
                print(f"Monomer 2: {pair[1]}")
                print("-" * 40)
    
    except Exception as e:
        print(f"Error in model training: {str(e)}")
        raise