import pandas as pd
import numpy as np
from rdkit import Chem
from Model import Constants
import tensorflow as tf


def data_load(file_path):
    df = pd.read_excel(file_path)
    max_len =max(df['SMILES'].str.len())
    df = data_filterize(df)
    return max_len,df


def data_filterize(df):
    df_smiles = []
    for idx in range(len(df['SMILES'])):
        smiles = df["SMILES"][idx].split(',')
        if len(smiles) == 2:
            df_smiles.append(smiles)
    return df_smiles[:50]


 # 512  # 435 #600  # Size of the latent space

#MAX_MOLSIZE, df_smiles = data_load(Constants.DATA_FILE_PATH)


def convert_smiles_to_graph(smiles):
    adjecenies_array = []
    features_array = []
    for i in range(2):
        molecule = Chem.MolFromSmiles(smiles[i])
        adjacency = np.zeros((Constants.NUM_ATOMS, Constants.NUM_ATOMS), 'float32')
        features = np.zeros((Constants.NUM_ATOMS, Constants.ATOM_DIM), 'float32')
        for atom in molecule.GetAtoms():
            i = atom.GetIdx()
            atom_type = Constants.ATOM_MAPPING[atom.GetSymbol()]
            features[i] = np.eye(Constants.ATOM_DIM)[atom_type]
            for neighbor in atom.GetNeighbors():
                j = neighbor.GetIdx()

                bond = molecule.GetBondBetweenAtoms(i, j)
                bond_type_idx = Constants.BOND_MAPPING[bond.GetBondType().name]
                # adjacency[[i,j],[j,i]] = bond_type_idx+1
                adjacency[i, j] = bond_type_idx + 1
        features[np.where(np.sum(features, axis=1) == 0)[0], -1] = 1
        # adjacency[-1,np.sum(adjacency, axis=0)== 0] = 1
        adjecenies_array.append(adjacency)
        features_array.append(features)
    return adjecenies_array[0], features_array[0], adjecenies_array[1], features_array[1]


def create_mol_from_graph(adjacency, features):
    molecule = Chem.RWMol()
    keep_idx = np.where(
        (np.argmax(features, axis=1) != Constants.ATOM_DIM - 1))[0]
    features = features[keep_idx]
    adjacency = adjacency[keep_idx, :][:, keep_idx]
    for atom_type_idx in np.argmax(features, axis=1):
        atom = Chem.Atom(Constants.ATOM_MAPPING[atom_type_idx])
        _ = molecule.AddAtom(atom)

    atoms_i, atoms_j = np.where(np.triu(adjacency) > 0)
    for atom_i, atom_j in zip(atoms_i, atoms_j):
        if atom_i == atom_j:
            continue
        bond_idx = adjacency[atom_i, atom_j] - 1
        if bond_idx > 3 or bond_idx < 0:
            bond_idx = 0
        bond_type = Constants.BOND_MAPPING[int(bond_idx)]
        molecule.AddBond(int(atom_i), int(atom_j), bond_type)
    flag = Chem.SanitizeMol(molecule, catchErrors=True)
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
        return None
    return molecule


def convert_graph_to_mol(graphs):
    adjecenies_array_0, features_array_0, adjecenies_array_1, features_array_1 = graphs
   # print(len(adjecenies_array_0))
    all_mols = []
    for i in range(len(adjecenies_array_0)):
        adjacency_0, features_0, adjacency_1, features_1 = adjecenies_array_0[i], features_array_0[i], \
                                                           adjecenies_array_1[i], features_array_1[i]
        mols = [create_mol_from_graph(adjacency_0, features_0), create_mol_from_graph(adjacency_1, features_1)]
        all_mols.append(mols)
    return all_mols

def convert_graph_to_mol_2(graphs):
    adjecenies_array_0, features_array_0, adjecenies_array_1, features_array_1 = graphs
    #print(len(adjecenies_array_0[0]))
    all_mols = []
    for i in range(len(adjecenies_array_0)):
        adjacency_0, features_0, adjacency_1, features_1 = adjecenies_array_0[i], features_array_0[i], \
                                                           adjecenies_array_1[i], features_array_1[i]
        print(adjacency_0.shape)
        mols = [create_mol_from_graph(adjacency_0.numpy(), features_0.numpy()), create_mol_from_graph(adjacency_1.numpy(), features_1.numpy())]
        all_mols.append(mols)
    return all_mols


# s =["C=CC(Cl)=O","NCC(=O)O"]
# print(s)
# a0, f0, a1, f1 = convert_smiles_to_graph(s)
# graph = [[a0], [f0], [a1], [f1]]
# mols = convert_graph_to_mol(graph)
# print(Chem.MolToSmiles(mols[0][0]), Chem.MolToSmiles(mols[0][1]))


def prepare_graph_features(df_smiles):
    adjacency_0, features_0, adjacency_1, features_1 = [], [], [], []

    for idx in range(len(df_smiles)):
        if idx == 78:
            continue
        smiles = df_smiles[idx]
        adj_0, feat_0, adj_1, feat_1 = convert_smiles_to_graph(smiles)
        adjacency_0.append(adj_0)
        features_0.append(feat_0)
        adjacency_1.append(adj_1)
        features_1.append(feat_1)
    # graph = [adjacency_0, features_0, adjacency_1, features_1]
    #
    # mols_all = convert_graph_to_mol(graph)

    adjacency_0_tensor = np.array(adjacency_0, dtype='float32')
    adjacency_1_tensor = np.array(adjacency_1, dtype='float32')
    feature_0_tensor = np.array(features_0, dtype='float32')
    feature_1_tensor = np.array(features_1, dtype='float32')
    return adjacency_0_tensor,feature_0_tensor,adjacency_1_tensor,feature_1_tensor

#adjacency_0_tensor,adjacency_1_tensor,feature_0_tensor,feature_1_tensor=prepare_graph_features(df_smiles)



## Conditional Features Started here

def hasEpoxyGroup(smile):
    mol = Chem.MolFromSmiles(smile)
    substructure = Chem.MolFromSmarts('C1OC1')
    matches = []
    if mol.HasSubstructMatch(substructure):
        matches = mol.GetSubstructMatches(substructure)
    else:
        return (False, 0, 0, False)
    return (len(matches) >= 2, len(matches), matches, False)


def has_imine(smiles):
    imine_pattern_1 = Chem.MolFromSmarts('NC')
    imine_pattern_2 = Chem.MolFromSmarts('Nc')
    capital_C = False
    mol = Chem.MolFromSmiles(smiles)
    matches = []
    if mol.HasSubstructMatch(imine_pattern_1):
        matches = mol.GetSubstructMatches(imine_pattern_1)
        capital_C = True
    elif mol.HasSubstructMatch(imine_pattern_2):
        matches = mol.GetSubstructMatches(imine_pattern_2)
        capital_C = False
    else:
        return (False, 0, 0, False)
    return (len(matches) >= 2, len(matches), matches, capital_C)


def has_vinyl_group(smiles):
    vinyl_pattern = Chem.MolFromSmarts('C=C')
    mol = Chem.MolFromSmiles(smiles)

    if mol is not None and vinyl_pattern is not None:
        matches = mol.GetSubstructMatches(vinyl_pattern)
        return (len(matches) >= 2, len(matches), matches, False)
    else:
        return (False, 0, 0, False)


def has_thiol_group(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        thiol_substructure = Chem.MolFromSmiles('CCS')
        matches = mol.GetSubstructMatches(thiol_substructure)
        return (len(matches) >= 2, len(matches), matches)
    else:
        return (False, 0, 0, False)


def has_acrylate_group(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol:
        acrylate_substructure = Chem.MolFromSmiles('C=C(C=O)')
        matches = mol.GetSubstructMatches(acrylate_substructure)
        return (len(matches) >= 2, len(matches), matches)
    else:
        return (False, 0, 0, False)


def indexize(indices, array_data, value):
    # array_data[:] = 1
    for i in range(len(indices)):
        idxs = indices[i]
        for j in range(len(idxs) - 1):
            # array_data[0,idxs[j]]=value
            if value == 1:
                array_data[idxs[0], idxs[len(idxs) - 1]] = value
            array_data[idxs[j], idxs[j + 1]] = value
            array_data[idxs[j + 1], idxs[j]] = value
    return array_data


def checkFunctionalGroups(smile1, smile2):
    result_epoxy = hasEpoxyGroup(smile1)
    result_imine = has_imine(smile1)
    result_vinyl = has_vinyl_group(smile1)
    result_thiol = has_thiol_group(smile1)
    result_acrylates = has_acrylate_group(smile1)

    result_epoxy_1 = hasEpoxyGroup(smile2)
    result_imine_1 = has_imine(smile2)
    result_vinyl_1 = has_vinyl_group(smile2)
    result_thiol_1 = has_thiol_group(smile2)
    result_acrylates_1 = has_acrylate_group(smile2)

    if result_epoxy[0] and result_imine_1[0]:
        return (result_epoxy[2], result_imine_1[2], Constants.EPOXY_GROUP, False)
    elif result_imine[0] and result_epoxy_1[0]:
        return (result_imine[2], result_epoxy_1[2], Constants.IMINE_GROUP, result_imine[3])
    elif result_vinyl[0] and result_vinyl_1[0]:
        return (result_vinyl[2], result_vinyl_1[2], Constants.VINYL_GROUP, False)
    elif result_vinyl[0] and result_thiol_1[0]:
        return (result_vinyl[2], result_thiol_1[2], Constants.THIOL_GROUP, False)
    elif result_thiol[0] and result_vinyl_1[0]:
        return (result_thiol[2], result_vinyl_1[2], Constants.ACRYLATE_GROUP, False)
    else:
        return None


def make_condtion(indices_1, indices_2, condtion, is_capital_C, imine_features_cond_1, imine_features_cond_2):
    if condtion == Constants.EPOXY_GROUP:
        imine_features_cond_1 = indexize(indices_1, imine_features_cond_1, 1)
        imine_features_cond_2 = indexize(indices_2, imine_features_cond_2, 1)
    if condtion == Constants.IMINE_GROUP:
        imine_features_cond_1 = indexize(indices_1, imine_features_cond_1, 1)
        imine_features_cond_2 = indexize(indices_2, imine_features_cond_2, 1)
    if condtion == Constants.VINYL_GROUP:
        imine_features_cond_1 = indexize(indices_1, imine_features_cond_1, 2)
        imine_features_cond_2 = indexize(indices_2, imine_features_cond_2, 2)
    if condtion == Constants.THIOL_GROUP:
        imine_features_cond_1 = indexize(indices_1, imine_features_cond_1, 2)
        imine_features_cond_2 = indexize(indices_2, imine_features_cond_2, 1)
    if condtion == Constants.ACRYLATE_GROUP:
        imine_features_cond_1 = indexize(indices_1, imine_features_cond_1, 1)
        imine_features_cond_2 = indexize(indices_2, imine_features_cond_2, 2)

    return imine_features_cond_1, imine_features_cond_2




def prepare_condition(sample):
    s1, s2 = sample[0], sample[1]
    imine_features_cond_1 = np.zeros((Constants.NUM_ATOMS, Constants.NUM_ATOMS), 'float32')
    imine_features_cond_2 = np.zeros((Constants.NUM_ATOMS, Constants.NUM_ATOMS), 'float32')
    if checkFunctionalGroups(s1, s2) != None:
        indices_1, indices_2, condtion, is_capital_C = checkFunctionalGroups(s1, s2)
        imine_features_cond_1, imine_features_cond_2 = make_condtion(indices_1, indices_2, condtion,
                                                                     is_capital_C, imine_features_cond_1,
                                                                     imine_features_cond_2)
        # condition_3.append(imine_features_cond_1)
    return imine_features_cond_1, imine_features_cond_2


def get_condtions(df_smiles):
    condition_1 = []
    condition_2 = []

    for i in range(len(df_smiles)):
        if i == 78:
            continue
        imine_features_cond_1, imine_features_cond_2 = prepare_condition(df_smiles[i])
        condition_1.append(imine_features_cond_1)
        condition_2.append(imine_features_cond_2)
        condition_1_array = np.array(condition_1, 'float32')
        condition_2_array= np.array(condition_2, 'float32')
    return condition_1_array, condition_2_array


#condition_1, condition_2=get_condtions(df_smiles)




# print(prepare_condition(
#     ['CC(C)(c2ccc(OCC1CO1)cc2)c4ccc(OCC3OC3)cc4', 'CC(C)(c3ccc2OCN(c1ccccc1)Cc2c3)c6ccc5OCN(c4ccccc4)Cc5c6']))
def get_imine_feature_list(df_smiles):
    imine_list = []
    for i in range(len(df_smiles)):
        if i == 78:
            continue
        val = 0

        array_160_1 = np.zeros((Constants.NUM_ATOMS, 1))
        imine_list.append(array_160_1 + val)

    imine_features_array = np.array(imine_list, 'float32')
    return imine_features_array

def has_benzene_with_extra_double_bonds(molecule):
    if molecule is None:
        return False
    benzene_pattern = Chem.MolFromSmarts("c1ccccc1")
    matches = molecule.GetSubstructMatches(benzene_pattern)
    return len(matches) > 0


# def graph_to_molecule2_d1(graphs):
#     adjecenies_array_0, features_array_0, adjecenies_array_1, features_array_1 = graphs
#
#     all_mols = []
#     for i in range(len(adjecenies_array_0)):
#         adjacency_0 = tf.linalg.set_diag(adjecenies_array_0[i],
#                                          tf.zeros(tf.shape(adjecenies_array_0[i])[:-1]))
#         adjacency_0 = adjacency_0.numpy()
#         adjacency_0 = abs(adjacency_0.astype(int))
#         features_0 = tf.argmax(features_array_0[i], axis=1)
#         features_0 = tf.one_hot(features_0, depth=Constants.ATOM_DIM, axis=1)
#
#         adjacency_1 = tf.linalg.set_diag(adjecenies_array_1[i],
#                                          tf.zeros(tf.shape(adjecenies_array_1[i])[:-1]))
#         adjacency_1 = adjacency_1.numpy()
#         adjacency_1 = abs(adjacency_1.astype(int))
#         features_1 = tf.argmax(features_array_1[i], axis=1)
#         features_1 = tf.one_hot(features_1, depth=Constants.ATOM_DIM, axis=1)
#         mols = [create_mol_from_graph(adjacency_0, features_0.numpy()), create_mol_from_graph(adjacency_1, features_1.numpy())]
#         all_mols.append(mols)
#
#     return all_mols
#
#
# def loss_BenzeneRing(adjacency_0_gen, features_0_gen, adjacency_1_gen, features_1_gen, total_loss):
#     mols = []
#     for i in range(adjacency_0_gen.shape[0]):
#         mols.append(graph_to_molecule2_d1([[adjacency_0_gen[i]], [features_0_gen[i]],
#                                            [adjacency_1_gen[i]], [features_1_gen[i]]]))
#     count_issue = 0
#     count_no_issue = 0
#     for mls in mols:
#         if mls[0][0] != None:
#             if has_benzene_with_extra_double_bonds(mls[0][0]):
#                 count_issue += 1
#             else:
#                 count_no_issue += 1
#         else:
#             count_issue += 0
#         if mls[0][1] != None:
#             if has_benzene_with_extra_double_bonds(mls[0][1]):
#                 count_issue += 1
#             else:
#                 count_no_issue += 1
#         else:
#             count_issue += 0
#
#     if count_no_issue < 0.5 * count_issue:
#         total_loss = np.log(count_issue) + total_loss
#     return total_loss