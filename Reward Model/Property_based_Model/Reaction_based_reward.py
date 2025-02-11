
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from sascorer import calculateScore
functional_groups = {
    'Hydroxyl': '[OX2H]',  # Alcohol -OH group
    'Carboxyl': '[CX3](=O)[OX2H1]',  # Carboxylic acid -COOH group
    'Amine': '[NX3;H2,H1;!$(NC=O)]',  # Primary or secondary amine -NH2 or -NH-
    'Isocyanate': '[NX1]=[CX2]=[OX1]',  # Isocyanate -N=C=O group
    'Vinyl': '[CX3]=[CX2]',  # Vinyl group -CH=CH2
    # 'Epoxide': '[OX2r3]',  # Epoxy group (three-membered ring with oxygen)
    'Epoxy': '[OX2r3][CX4][CX4]',  # Epoxy group (three-membered ring with oxygen and two carbons)
    'Aldehyde': '[CX3H1](=O)[#6]',  # Aldehyde -CHO group
    'Ketone': '[#6][CX3](=O)[#6]',  # Ketone -C(=O)- group
    'Ester': '[CX3](=O)[OX2][#6]',  # Ester -COOR group
    'Thiol': '[#16X2H]',  # Thiol -SH group
    'Thioester': '[CX3](=O)[SX2][#6]',  # Thioester -COSR group
    'Ether': '[OD2]([#6])[#6]',  # Ether -O- group
    'Anhydride': '[CX3](=O)O[CX3](=O)',  # Anhydride -CO-O-CO- group
    'Acyl Chloride': '[CX3](=O)Cl',  # Acyl chloride -COCl group
    'Sulfonic Acid': '[SX4](=O)(=O)([OX2H])O',  # Sulfonic acid -SO3H group
    'Sulfonamide': '[NX3][SX4](=O)(=O)[#6]',  # Sulfonamide -SO2NR group
    'Phosphate': '[PX4](=O)([OX1-])(O[H1,O])',  # Phosphate group -PO4
    'Nitrile': '[CX2]#N',  # Nitrile -C≡N group
    'Phenol': '[cX3][OX2H]',  # Phenol -ArOH group
    'Azo': '[NX2]=[NX2]',  # Azo -N=N- group
    'Azide': '[NX3]([NX1]=[NX1])',  # Azide -N3 group
    'Amide': '[NX3][CX3](=O)[#6]',  # Amide -CONH- group
    'Imide': '[NX3]([CX3](=O))[CX3](=O)',  # Imide -CONHCO- group
    'Peroxide': '[OX2][OX2]',  # Peroxide -O-O- group
    'Carbonate': '[CX3](=O)[OX2][CX3](=O)',  # Carbonate -O(CO)O- group
    'Isothiocyanate': '[NX1]=[CX2]=[SX1]',  # Isothiocyanate -N=C=S group
    'Carbamate': '[NX3][CX3](=O)[OX2][#6]',  # Carbamate -OC(=O)NR- group
    'Boronate': '[BX3]([OX2])[OX2]',  # Boronate group -B(OR)2
    'Triazole': '[nX2]1[nX2][nX2]1',  # Triazole group (five-membered ring with three nitrogens)
    'Tetrazole': '[nX2]1[nX2][nX2][nX2]1',  # Tetrazole group (five-membered ring with four nitrogens)
    'Phosphonate': '[PX3](=O)(O[H1,O])([#6])',  # Phosphonate group -PO(OR)2
    # Add more as needed
}

reactivity_pairs = [
    ('Hydroxyl', 'Carboxyl', 'Esterification', 'Ester (-COOR)'),
    ('Hydroxyl', 'Isocyanate', 'Urethane Formation', 'Urethane (-NHCOO-)'),
    ('Hydroxyl', 'Acyl Chloride', 'Esterification', 'Ester (-COOR)'),
    ('Hydroxyl', 'Anhydride', 'Esterification and Acid Formation', 'Ester (-COOR) + Carboxylic Acid'),
    ('Hydroxyl', 'Epoxy', 'Nucleophilic Ring Opening', 'Beta-Hydroxy Ether'),
    ('Hydroxyl', 'Isothiocyanate', 'Thiocarbamate Formation', 'Thiocarbamate'),
    ('Hydroxyl', 'Aldehyde', 'Hemiacetal Formation', 'Hemiacetal (-C(OH)(OR)-)'),
    ('Amine', 'Carboxyl', 'Amidation', 'Amide (-CONH-)'),
    ('Amine', 'Acyl Chloride', 'Amidation', 'Amide (-CONH-)'),
    ('Amine', 'Isocyanate', 'Urea Formation', 'Urea (-NHCONH-)'),
    ('Amine', 'Anhydride', 'Amidation and Acid Formation', 'Amide (-CONH-) + Carboxylic Acid'),
    ('Amine', 'Epoxy', 'Nucleophilic Ring Opening', 'Beta-Amino Alcohol'),
    ('Amine', 'Aldehyde', 'Schiff Base Formation', 'Imine (=N-)'),
    ('Amine', 'Ketone', 'Imine Formation', 'Imine (=N-)'),
    ('Amine', 'Isothiocyanate', 'Thiourea Formation', 'Thiourea (-NHCSNH-)'),
    ('Carboxyl', 'Hydroxyl', 'Esterification', 'Ester (-COOR)'),
    ('Carboxyl', 'Amine', 'Amidation', 'Amide (-CONH-)'),
    ('Carboxyl', 'Epoxy', 'Nucleophilic Ring Opening', 'Beta-Hydroxy Ester'),
    ('Carboxyl', 'Isocyanate', 'Carbamic Acid Formation', 'Carbamate (-COONH-)'),
    ('Carboxyl', 'Acyl Chloride', 'Acid Chloride Formation', 'Acid Chloride (-COCl)'),
    ('Carboxyl', 'Anhydride', 'Formation of Acids', 'Carboxylic Acid'),
    ('Isocyanate', 'Hydroxyl', 'Urethane Formation', 'Urethane (-NHCOO-)'),
    ('Isocyanate', 'Amine', 'Urea Formation', 'Urea (-NHCONH-)'),
    ('Isocyanate', 'Water', 'Hydrolysis', 'Amine (-NH2) and CO2'),
    ('Isocyanate', 'Epoxy', 'Epoxy-Isocyanate Reaction', 'Isocyanate Polymer'),
    ('Epoxy', 'Amine', 'Nucleophilic Ring Opening', 'Beta-Amino Alcohol'),
    ('Epoxy', 'Carboxyl', 'Nucleophilic Ring Opening', 'Beta-Hydroxy Ester'),
    ('Epoxy', 'Hydroxyl', 'Nucleophilic Ring Opening', 'Beta-Hydroxy Ether'),
    ('Epoxy', 'Thiol', 'Thiol-Epoxy Reaction', 'Thioether or Beta-Mercapto Alcohol'),
    ('Epoxy', 'Acyl Chloride', 'Nucleophilic Ring Opening', 'Halohydrin'),
    ('Aldehyde', 'Hydroxyl', 'Acetal/Ketal Formation', 'Acetal (C(OR)2) or Hemiacetal'),
    ('Aldehyde', 'Amine', 'Schiff Base Formation', 'Imine (=N-)'),
    ('Aldehyde', 'Cyanide', 'Cyanohydrin Formation', 'Cyanohydrin (-C(OH)CN)'),
    ('Aldehyde', 'Nucleophiles (e.g., RMgX)', 'Nucleophilic Addition', 'Alcohol (-OH)'),
    ('Ketone', 'Amine', 'Imine Formation', 'Imine (=N-)'),
    ('Ketone', 'Hydroxyl', 'Hemiacetal/Ketal Formation', 'Hemiacetal or Ketal'),
    ('Ketone', 'Nucleophiles (e.g., RMgX)', 'Nucleophilic Addition', 'Alcohol (-OH)'),
    ('Thiol', 'Epoxy', 'Thiol-Epoxy Reaction (Ring Opening)', 'Thioether or Beta-Mercapto Alcohol'),
    ('Thiol', 'Vinyl', 'Thiol-Ene Click Chemistry', 'Thioether'),
    ('Thiol', 'Disulfide', 'Disulfide Exchange Reaction', 'New Disulfide Bonds'),
    ('Vinyl', 'Radical Initiators', 'Radical Polymerization', 'Polyvinyl Chain'),
    ('Vinyl', 'Diene (Conjugated Diene)', 'Diels-Alder Cycloaddition', 'Cyclohexene Derivative'),
    ('Acyl Chloride', 'Hydroxyl', 'Esterification', 'Ester (-COOR)'),
    ('Acyl Chloride', 'Amine', 'Amidation', 'Amide (-CONH-)'),
    ('Anhydride', 'Hydroxyl', 'Esterification and Acid Formation', 'Ester and Carboxylic Acid'),
    ('Anhydride', 'Amine', 'Amidation and Acid Formation', 'Amide and Carboxylic Acid'),
    ('Nitrile', 'Water', 'Hydrolysis', 'Amide or Carboxylic Acid'),
    ('Nitrile', 'Amine', 'Nucleophilic Addition', 'Amidine'),
    ('Phenol', 'Acyl Chloride', 'Esterification', 'Ester (-COOR)'),
    ('Azo', 'Reducing Agents', 'Reduction', 'Amines (-NH2)'),
    ('Triazole', 'Alkynes', 'Azide-Alkyne Cycloaddition (Click Reaction)', '1,2,3-Triazole'),
    ('Carbamate', 'Hydroxyl', 'Transesterification', 'Carbamate'),
    ('Isothiocyanate', 'Amine', 'Thiourea Formation', 'Thiourea (-NHCSNH-)'),
    ('Boronate', 'Hydroxyl', 'Boronic Ester Formation', 'Boronic Ester (-B(OR)3)'),
    ('Boronate', 'Amine', 'Boron-Nitrogen Complex Formation', 'Boronate Complex'),
]

reaction_weights = {
    'Urethane Formation': 2.0,
    'Epoxy Ring Opening': 2.5,
    'Thiol-Epoxy Reaction': 2.5,
    'Amidation': 1.5,
    'Anhydride Reaction with Hydroxyl or Amine': 1.8,
    'Esterification': 1.2,
    'Isocyanate Reactions with Water': 1.0,
    'Acyl Chloride Reactions with Amine or Hydroxyl': 1.3,
    'Diels-Alder Cycloaddition': 2.0,
    'Schiff Base Formation': 1.5,
    'Disulfide Exchange': 2.0,
    'Triazole Formation (Click Chemistry)': 2.2,
    'Cyanate Ester Formation': 2.5,
    'Nucleophilic Ring Opening': 3.0
}

def reaction_reward_scoring(generated_smiles):
    monomer1_smiles = generated_smiles[0]
    monomer2_smiles = generated_smiles[1]
    # monomer1_smiles = 'OCC(O)(COC1CO1)COC2CO2'  # 'NCCN'  # Example monomer with a vinyl and carboxyl group
    # monomer2_smiles = 'CCNc1ccc(cc1)Cc2ccc(N=C=O)cc2'  # 'C2OC2CCOC1OC1'  # Example monomer with an alcohol group

    # Convert SMILES to RDKit Molecule objects
    monomer1 = Chem.MolFromSmiles(monomer1_smiles)
    monomer2 = Chem.MolFromSmiles(monomer2_smiles)

    monomer1_groups = find_functional_groups(monomer1, functional_groups)
    monomer2_groups = find_functional_groups(monomer2, functional_groups)

    # Print results
    #print(f'Monomer 1 Functional Groups: {monomer1_groups}')
    #print(f'Monomer 2 Functional Groups: {monomer2_groups}')
    reward= calcualte_reward(monomer1_groups,monomer2_groups,reactivity_pairs,reaction_weights)

    #print(f"Reward score for the generated monomers: {reward:.2f}")
    return reward


# Updated function to check for functional groups in a monomer
def find_functional_groups(molecule, fg_dict):
    # found_groups = {}
    found_groups = []
    for name, smarts in fg_dict.items():
        pattern = Chem.MolFromSmarts(smarts)
        if molecule.HasSubstructMatch(pattern):
            # found_groups[name] = True
            found_groups.append(name)
    return found_groups

# Function to check for potential reactions between two monomers
def check_potential_reactions(monomer1_groups, monomer2_groups, reactivity_pairs):
    potential_reactions = []
    for fg1, fg2, reaction, product in reactivity_pairs:
        if fg1 in monomer1_groups and fg2 in monomer2_groups:
            potential_reactions.append((fg1, fg2, reaction, product))
        elif fg2 in monomer1_groups and fg1 in monomer2_groups:
            potential_reactions.append((fg2, fg1, reaction, product))

        # if monomer1_groups.get(fg1, False) and monomer2_groups.get(fg2, False):
        #     potential_reactions.append((fg1, fg2, reaction, product))
        # elif monomer1_groups.get(fg2, False) and monomer2_groups.get(fg1, False):
        #     potential_reactions.append((fg2, fg1, reaction, product))
    return set(potential_reactions)


# Example usage


# Function to calculate a reward score based on potential reactions
def calcualte_reward(monomer1_groups, monomer2_groups, reactivity_pairs, weight_dict=None):
    potential_reactions = check_potential_reactions(monomer1_groups, monomer2_groups, reactivity_pairs)
    print(potential_reactions)

    reward_score = 0.0

    for fg1, fg2, reaction, product in potential_reactions:
        # Add the weight for each detected reaction
        reward_score += weight_dict.get(reaction, 1.0)

    # Calculate the maximum possible score for the given monomers
    max_possible_score = reaction_weights.get(max(reaction_weights, key=reaction_weights.get))
    print(max_possible_score)

    # Normalize the score
    if max_possible_score > 0:
        normalized_reward = reward_score / max_possible_score
    else:
        normalized_reward = 0.0

    return normalized_reward





