
from Property_based_reward import property_based_reward_scoring
from Synthesis_based_reward import synthesis_reward_scoring
from Functional_group_based_reward import functional_group_reward_score
from Functional_group_based_reward_2 import compatibility_based_reward_score
from Database_similarity_reward import similarity_reward_scoring
from Reaction_based_reward import reaction_reward_scoring


def min_max_normalize(score, min_value, max_value):
    if max_value == min_value:
        return 0.0
    return (score - min_value) / (max_value - min_value)


def calculate_composite_reward(generated_smiles, w1=0.2, w2=0.2, w3=0.2, w4=0.2, w5=0.2):
    property_min, property_max = 0.0, 1.0
    fg_min, fg_max = 0.0, 1.0
    sa_min, sa_max = 0.0, 1.0
    react_min,react_max=0.0,1.0
    similarity_min, similarity_max = 0.0, 1.0

    property_score = property_based_reward_scoring([generated_smiles])
    synthesis_score = synthesis_reward_scoring(generated_smiles)
    functional_group_score = compatibility_based_reward_score(generated_smiles)
    reaction_score = reaction_reward_scoring(generated_smiles)
    similarity_score = similarity_reward_scoring(generated_smiles)
    print("property_score",property_score)
    print("synthesis_score : ",synthesis_score)
    print("functional_group_score : ", functional_group_score)
    print("reaction_score : ", reaction_score)
    print("similarity_score : ", similarity_score)

    # Normalize the scores
    norm_property_score = min_max_normalize(property_score, property_min, property_max)
    norm_fg_score = min_max_normalize(functional_group_score, fg_min, fg_max)
    norm_sa_score = min_max_normalize(synthesis_score, sa_min, sa_max)
    norm_reaction_score=min_max_normalize(reaction_score, react_min,react_max)
    norm_similarity_score = min_max_normalize(similarity_score, similarity_min, similarity_max)


    # Combine scores using the assigned weights
    total_reward_score = ( w1 * norm_property_score+
                          w2 * norm_fg_score +
                          w3 * norm_sa_score +
                          w4 * norm_reaction_score +
                          w5 * norm_similarity_score)

    return total_reward_score


import pandas as pd
data = pd.read_excel('../Data/smiles.xlsx')
smiles=data['SMILES'][189].split(',')
print(smiles)
reward= calculate_composite_reward(smiles)
print('Reward: ', reward)
