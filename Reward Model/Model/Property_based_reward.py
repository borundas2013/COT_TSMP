
from pred.Prediction_Model import predict_property
def tg_based_scoring(tg):
    if tg >= 50 and tg <= 100:
        return round(1 - (abs(tg-75)/25),2)
    return 0.0

def er_based_scoring(er):
    if er >150 and er <300:
        return round(1 - (abs(er-225)/75),2)
    return 0.0

def property_based_reward_scoring(generated_smiles):
    er, tg=  predict_property( generated_smiles)
    print(er, tg)

    tg_score=tg_based_scoring(tg)
    er_score=er_based_scoring(er)
    print(tg_score,er_score)
    return round((tg_score+er_score)/2,2)

monomer1_smiles = 'c3cc(N(CC1CC1)CC2CO2)ccc3OCC4CO4'#'C1COC1Cl'  # Epichlorohydrin (epoxy monomer)
monomer2_smiles = 'NCCNCCN(CCNCCC(CN)CCN)CCN(CCNCCN)CCN(CCN)CCN'#NCCN'
generated_smiles = [[monomer1_smiles,monomer2_smiles]]
print(property_based_reward_scoring(generated_smiles))