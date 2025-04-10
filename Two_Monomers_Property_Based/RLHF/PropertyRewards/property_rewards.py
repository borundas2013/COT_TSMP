import os
import sys

# Get the absolute path of the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

# Add RLHF directory to Python path
rlhf_dir = os.path.dirname(current_dir)
sys.path.append(rlhf_dir)

# Add Generator directory to Python path 
generator_dir = current_dir
sys.path.append(generator_dir)

from Property_Prediction.predict import predict_property, predict_properties_batch

class PropertyRewardCalculator:
    def __init__(self, tg_weight=0.5, er_weight=0.5):
        """
        Initialize the PropertyRewardCalculator
        
        Args:
            tg_weight (float): Weight for Tg reward (default: 0.5)
            er_weight (float): Weight for Er reward (default: 0.5)
        """
        self.tg_weight = tg_weight
        self.er_weight = er_weight

    def calculate_single_reward(self, sample):
        """
        Calculate reward for a single sample
        """
        smiles1 = sample["smiles1"]
        smiles2 = sample["smiles2"]
        actual_tg = sample["actual_tg"]
        actual_er = sample["actual_er"]
        try:
            scores = predict_property(smiles1, smiles2)
            tg_score = scores["tg_score"]
            er_score = scores["er_score"]
            
            print(f"TG Score: {tg_score:.3f}")
            print(f"ER Score: {er_score:.3f}")
            print(f"Actual TG: {actual_tg:.3f}")
            print(f"Actual ER: {actual_er:.3f}")
            
            tg_error = abs(tg_score - actual_tg) / abs(actual_tg)
            er_error = abs(er_score - actual_er) / abs(actual_er)
            
            tg_reward = max(0, 1 - tg_error)
            er_reward = max(0, 1 - er_error)
            total_reward = self.tg_weight * tg_reward + self.er_weight * er_reward
            
            return {
                "total_reward": total_reward,
                "tg_reward": tg_reward,
                "er_reward": er_reward
            }
        except Exception as e:
            print(f"Error in property reward calculation: {e}")
            return {
                "total_reward": 0.0,
                "tg_reward": 0.0,
                "er_reward": 0.0
            }
    def calculate_batch_reward(self, sample_list):
        """
        Calculate rewards for a batch of samples
        """
        total_score = []
        er_score = []
        tg_score = []
        
        for sample in sample_list:
            reward_dict = self.calculate_single_reward(sample)
            total_score.append(reward_dict["total_reward"])
            tg_score.append(reward_dict["tg_reward"])
            er_score.append(reward_dict["er_reward"])
            
        return {
            "total_score": total_score,
            "tg_score": tg_score,
            "er_score": er_score
        }

if __name__ == "__main__":
    # Create calculator instance
    calculator = PropertyRewardCalculator()
    
    # Test single sample
    smiles1 = 'CN(C)Cc1ccc(-c2ccc3cnc(Nc4ccc(C5CCN(CC(N)=O)CC5)cc4)nn23)cc1'
    smiles2 = 'CCCNC1OC1'
    actual_tg = 250.0  # example value
    actual_er = 100  # example value
    
    reward_dict = calculator.calculate_single_reward(smiles1, smiles2, actual_tg, actual_er)
    print("\nSingle-Sample Reward Results:")
    print("Total Reward: {:.3f}".format(reward_dict["total_reward"]))
    print("TG Reward: {:.3f}".format(reward_dict["tg_reward"]))
    print("ER Reward: {:.3f}".format(reward_dict["er_reward"]))

    # Test batch
    sample_list = [
        {"smiles1": smiles1, "smiles2": smiles2, "actual_tg": actual_tg, "actual_er": actual_er}
    ]
    reward_dict = calculator.calculate_batch_reward(sample_list)
    print("\nBatch Reward Results:")
    print("Total Reward: {:.3f}".format(reward_dict["total_reward"]))
    print("TG Reward: {:.3f}".format(reward_dict["tg_reward"]))
    print("ER Reward: {:.3f}".format(reward_dict["er_reward"]))



