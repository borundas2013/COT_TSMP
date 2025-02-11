from rdkit import Chem
from rdkit.Chem import rdmolops
import numpy as np
import deepchem as dc
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# Constants
DEFAULT_ATOM_TYPE_SET = [
    "C", "N", "O", "F", "P", "S", "Cl", "Br"
]
DEFAULT_HYBRIDIZATION_SET = ["SP", "SP2", "SP3"]
DEFAULT_TOTAL_NUM_Hs_SET = [0, 1, 2, 3, 4]
DEFAULT_FORMAL_CHARGE_SET = [-2, -1, 0, 1, 2]
DEFAULT_TOTAL_DEGREE_SET = [0, 1, 2, 3, 4, 5]
DEFAULT_RING_SIZE_SET = [3, 4, 5, 6, 7, 8]
DEFAULT_BOND_TYPE_SET = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
DEFAULT_BOND_STEREO_SET = ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"]
DEFAULT_GRAPH_DISTANCE_SET = [1, 2, 3, 4, 5, 6, 7]
DEFAULT_ATOM_IMPLICIT_VALENCE_SET = [0, 1, 2, 3, 4, 5, 6]
DEFAULT_ATOM_EXPLICIT_VALENCE_SET = [1, 2, 3, 4, 5, 6]

# Core feature processing functions
def pad_features(node_features, edge_features, edge_index, max_nodes=280, max_edges=280):
    """
    Pad node features, edge features, and edge index to fixed sizes.
    
    Args:
        node_features: Node feature matrix
        edge_features: Edge feature matrix
        edge_index: Edge index matrix
        max_nodes: Maximum number of nodes
        max_edges: Maximum number of edges
        
    Returns:
        Tuple of (padded_nodes, padded_edges, padded_edge_index)
    """
    # Pad node features
    feature_dim = node_features.shape[1]
    padded_nodes = np.zeros((max_nodes, feature_dim))
    padded_nodes[:node_features.shape[0], :] = node_features

    # Pad edge features
    edge_dim = edge_features.shape[1]
    padded_edges = np.zeros((max_edges, edge_dim))
    padded_edges[:edge_features.shape[0], :] = edge_features

    # Pad edge index
    padded_edge_index = np.full((2, max_edges), -1)
    padded_edge_index[:, :edge_index.shape[1]] = edge_index

    return padded_nodes, padded_edges, padded_edge_index

def defeaturize_to_smiles(node_features, edge_features, edge_index):
    """Convert molecular graph features back to SMILES string."""
    mol = Chem.RWMol()  
    atom_mapping = []  

    # Add atoms
    for node in node_features:
        atom_type = DEFAULT_ATOM_TYPE_SET[np.argmax(node[:len(DEFAULT_ATOM_TYPE_SET)])]
        if atom_type is None:
            continue
        atom = Chem.Atom(atom_type)
        
        formal_charge = int(node[10]) if 10 < len(node) else 0
        hybridization_type = DEFAULT_HYBRIDIZATION_SET[np.argmax(node[11:14])]
        aromatic = bool(node[16])

        atom.SetFormalCharge(formal_charge)
        atom.SetHybridization(getattr(Chem.rdchem.HybridizationType, hybridization_type.upper(), 
                                    Chem.rdchem.HybridizationType.UNSPECIFIED))
        atom.SetIsAromatic(aromatic)
        
        atom_idx = mol.AddAtom(atom)
        atom_mapping.append(atom_idx)  

    # Add bonds
    for edge, (start, end) in zip(edge_features, edge_index.T):
        if mol.GetBondBetweenAtoms(atom_mapping[start], atom_mapping[end]) is None:
            bond_type_str = DEFAULT_BOND_TYPE_SET[np.argmax(edge[:4])]
            bond_type = getattr(Chem.rdchem.BondType, bond_type_str.upper(), 
                              Chem.rdchem.BondType.UNSPECIFIED)
            mol.AddBond(atom_mapping[start], atom_mapping[end], bond_type)

    rdmolops.SanitizeMol(mol)
    return Chem.MolToSmiles(mol)

def features_to_smiles(node_features, edge_features, edge_index, max_nodes=74, max_edges=152):
    """Convert padded features back to SMILES string."""
    # Convert TensorFlow tensors to numpy if needed
    if tf.is_tensor(node_features):
        node_features = node_features.numpy()
    if tf.is_tensor(edge_features):
        edge_features = edge_features.numpy()
    if tf.is_tensor(edge_index):
        edge_index = edge_index.numpy()
    
    # Remove padding
    valid_node_mask = ~np.all(node_features == 0, axis=1)
    valid_edge_mask = ~np.all(edge_features == 0, axis=1)
    valid_edge_index_mask = edge_index[0] != -1
    
    valid_node_features = node_features[valid_node_mask]
    valid_edge_features = edge_features[valid_edge_mask]
    valid_edge_index = edge_index[:, valid_edge_index_mask]
    
    try:
        return defeaturize_to_smiles(valid_node_features, valid_edge_features, valid_edge_index)
    except:
        return None

# Data processing functions
def extract_and_pad_features(smiles_file):
    """Extract and pad features from SMILES file."""
    df = read_smiles_file(smiles_file)

    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)

    padded_node_features_1 = []
    padded_edge_features_1 = [] 
    padded_node_features_2 = []
    padded_edge_features_2 = []
    edge_index_1 = []
    edge_index_2 = []

    for _, row in df.iterrows():
        smiles = row['Smiles']
 
        # Process first molecule
        features_1 = featurizer.featurize([smiles[0]])[0]
        node_features_1, edge_features_1 = features_1.node_features, features_1.edge_features
            
        padded_nodes_1, padded_edges_1, padded_edge_index_1 = pad_features(
                node_features_1, 
                edge_features_1,
                features_1.edge_index
        )
        padded_node_features_1.append(padded_nodes_1)
        padded_edge_features_1.append(padded_edges_1)
        edge_index_1.append(padded_edge_index_1)

        # Process second molecule
        features_2 = featurizer.featurize([smiles[1]])[0]
        node_features_2, edge_features_2 = features_2.node_features, features_2.edge_features
            
        padded_nodes_2, padded_edges_2, padded_edge_index_2 = pad_features(
                node_features_2, 
                edge_features_2,
                features_2.edge_index
            )
        padded_node_features_2.append(padded_nodes_2)
        padded_edge_features_2.append(padded_edges_2)
        edge_index_2.append(padded_edge_index_2)

    return (
        np.array(padded_node_features_1),
        np.array(padded_edge_features_1),
        np.array(edge_index_1),
        np.array(padded_node_features_2),
        np.array(padded_edge_features_2),
        np.array(edge_index_2)
    )


def get_property_matrix(smiles_file):
    df_smiles = read_smiles_file(smiles_file)
   
    data=df_smiles[['Er','Tg']]
    scaler = MinMaxScaler()
    normalized_properties = scaler.fit_transform(data)
    property_matrix = np.array(normalized_properties)
    print("property_matrix.shape",property_matrix.shape)
    return property_matrix

def read_smiles_file(file_path):
    df = pd.read_csv(file_path)[:50]
    valid_rows = []
    for _, row in df.iterrows():
        smiles_str = row['Smiles'][1:-1].replace("'", "").split(',')
        if len(smiles_str) == 2:
            row['Smiles'] = [s.strip() for s in smiles_str]
            valid_rows.append(row)
    return pd.DataFrame(valid_rows)
   

# Example usage and testing
if __name__ == "__main__":

    # Test feature extraction and padding
    smiles_file = 'unique_smiles_Er.csv'
    results = extract_and_pad_features(smiles_file)
    padded_nodes_1, padded_edges_1, edge_index_1, padded_nodes_2, padded_edges_2, edge_index_2 = results
    print(padded_nodes_1.shape,padded_edges_1.shape,edge_index_1.shape,padded_nodes_2.shape,padded_edges_2.shape,edge_index_2.shape)
    
    # Test reconstruction
    smiles = features_to_smiles(padded_nodes_2[0], padded_edges_2[0], edge_index_2[0])
    print("Reconstructed SMILES:", smiles)

