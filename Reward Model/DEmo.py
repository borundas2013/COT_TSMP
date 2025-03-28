from rdkit import Chem
from rdkit.Chem import QED

def is_valid_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    return mol is not None

def score_two_monomer_sample(monomer1, monomer2, target_tg, target_er, pred_tg=None, pred_er=None):
    m1_valid = is_valid_smiles(monomer1)
    m2_valid = is_valid_smiles(monomer2)

    if not (m1_valid and m2_valid):
        return 1.0

    if pred_tg == target_tg and pred_er == target_er:
        return 5.0

    mol1 = Chem.MolFromSmiles(monomer1)
    mol2 = Chem.MolFromSmiles(monomer2)
    qed_score = (QED.qed(mol1) + QED.qed(mol2)) / 2

    if pred_tg is not None and pred_er is not None:
        tg_diff = abs(pred_tg - target_tg)
        er_diff = abs(pred_er - target_er)
        prop_score = max(0, 1 - 0.01 * tg_diff - 0.01 * er_diff)
    else:
        prop_score = 1.0

    total_score =0.8 * prop_score + 0.2 * qed_score
    return round(total_score * 5, 2)


# Example usage
score = score_two_monomer_sample(
    monomer1="CCOC(=O)C1=CC=CC=C1",
    monomer2="C(COC(=O)C)OC",
    target_tg=150,
    target_er=90,
    pred_tg=149,
    pred_er=90
)

print("Score:", score)
