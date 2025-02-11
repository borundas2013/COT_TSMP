import pandas as pd
import numpy as np
from rdkit import Chem
import Constants
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from transformers import PreTrainedTokenizerFast
#FILENAME='chembl_5thresh_Dataset_valid'
FILENAME='kaggle_SMILE_Dataset_valid'
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

def read_kaggle_smiles():
    try:
        # Read the processed Kaggle dataset
        df = pd.read_csv(f'Processed_Data/{FILENAME}.csv')
        
        # Extract SMILES column as a list
        smiles_list = df['Smiles'].tolist()
        
        print(f"Successfully loaded {len(smiles_list)} SMILES strings from Kaggle dataset")
        return smiles_list
        
        
    except Exception as e:
        print(f"Error reading Kaggle SMILES data: {e}")
        return None
    
def hasEpoxyGroup(smile):
    mol = Chem.MolFromSmiles(smile)
    substructure = Chem.MolFromSmarts('C1OC1')
    matches = []
    if mol is not None and mol.HasSubstructMatch(substructure):
        matches = mol.GetSubstructMatches(substructure)
    else:
        return None
    return  'C1OC1'

def has_imine(smiles):
    imine_pattern_1 = Chem.MolFromSmarts('NC')
    imine_pattern_2 = Chem.MolFromSmarts('Nc')
    capital_C = False
    mol = Chem.MolFromSmiles(smiles)
    matches = []
    if mol is not None and mol.HasSubstructMatch(imine_pattern_1):
        matches = mol.GetSubstructMatches(imine_pattern_1)
        capital_C = True
    elif mol is not None and mol.HasSubstructMatch(imine_pattern_2):
        matches = mol.GetSubstructMatches(imine_pattern_2)
        capital_C = False
    else:
        return None
    return 'NC' if capital_C else 'Nc'


def has_vinyl_group(smiles):
    vinyl_pattern = Chem.MolFromSmarts('C=C')
    mol = Chem.MolFromSmiles(smiles)

    if mol is not None  and mol.HasSubstructMatch(vinyl_pattern):
        matches = mol.GetSubstructMatches(vinyl_pattern)
        return 'C=C'
    else:
        return None


def has_thiol_group(smiles):
    thiol_substructure = Chem.MolFromSmarts('CCS')
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None  and mol.HasSubstructMatch(thiol_substructure):
        thiol_substructure = Chem.MolFromSmiles('CCS')
        matches = mol.GetSubstructMatches(thiol_substructure)
        return 'CCS'
    else:
        return None


def has_acrylate_group(smiles):
    mol = Chem.MolFromSmiles(smiles)
    acrylate_substructure = Chem.MolFromSmarts('C=C(C=O)')

    if mol is not None  and mol.HasSubstructMatch(acrylate_substructure):
        acrylate_substructure = Chem.MolFromSmiles('C=C(C=O)')
        matches = mol.GetSubstructMatches(acrylate_substructure)
        return 'C=C(C=O)'
    else:
        return None
    
def has_benzene_ring(smiles):
    # Aromatic notation pattern
    aromatic_pattern = Chem.MolFromSmarts('c1ccccc1')
    # Kekulé notation pattern
    kekule_pattern = Chem.MolFromSmarts('C1=CC=CC=C1')
    
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is not None:
        if mol.HasSubstructMatch(aromatic_pattern):
            return 'c1ccccc1'  # Aromatic notation
        elif mol.HasSubstructMatch(kekule_pattern):
            return 'C1=CC=CC=C1'  # Kekulé notation
    return None

def extract_group_smarts(smile):
    group_smarts = []
    epoxy = hasEpoxyGroup(smile)
    imine = has_imine(smile)
    vinyl = has_vinyl_group(smile)
    thiol = has_thiol_group(smile)
    acrylate = has_acrylate_group(smile)
    benzene = has_benzene_ring(smile)
    
    if epoxy:
        group_smarts.append(epoxy)
    if imine:
        group_smarts.append(imine)
    if vinyl:
        group_smarts.append(vinyl)
    if thiol:
        group_smarts.append(thiol)
    if acrylate:
        group_smarts.append(acrylate)
    if benzene:
        group_smarts.append(benzene)
    if not (epoxy or acrylate or imine or thiol or vinyl or benzene):
        group_smarts.append('No group')
    return group_smarts
def extract_groups(smiles_list):
    groups = []
    epoxy_count = 0
    imine_count = 0
    vinyl_count = 0
    thiol_count = 0
    acrylate_count = 0
    benzene_count = 0
    no_group_count = 0
    
    for smile in smiles_list:
        found_groups = []
        epoxy = hasEpoxyGroup(smile)
        imine = has_imine(smile) 
        vinyl = has_vinyl_group(smile)
        thiol = has_thiol_group(smile)
        acrylate = has_acrylate_group(smile)
        benzene = has_benzene_ring(smile)
        
        if epoxy:
            found_groups.append(epoxy)
            epoxy_count += 1
        if acrylate:
            found_groups.append(acrylate)
            acrylate_count += 1
        if imine:
            found_groups.append(imine)
            imine_count += 1
        if thiol:
            found_groups.append(thiol)
            thiol_count += 1
        if vinyl:
            found_groups.append(vinyl)
            vinyl_count += 1
        if benzene:
            found_groups.append(benzene)
            benzene_count += 1
        if not (epoxy or acrylate or imine or thiol or vinyl or benzene):
            found_groups.append('No group')
            no_group_count += 1
        groups.append(found_groups)
    print(f'Epoxy count: {epoxy_count}')
    print(f'Imine count: {imine_count}')
    print(f'Vinyl count: {vinyl_count}')
    print(f'Thiol count: {thiol_count}')
    print(f'Acrylate count: {acrylate_count}')
    print(f'Benzene count: {benzene_count}')
    print(f'No group count: {no_group_count}')
    encoded_groups = [encode_groups(groups, Constants.GROUP_VOCAB) for groups in groups]
        
    return smiles_list, encoded_groups

def encode_groups(groups, vocab):
    encoded = np.zeros(Constants.GROUP_SIZE, dtype=int)
    for group in groups:
        if group in vocab:
            encoded[vocab[group]] = 1
    return encoded

def extract_vocab(filename):
    # Read vocabulary from file
    smiles_vocab = {}
    try:
        print(f"Attempting to read vocabulary from: {filename}")
        with open(filename, 'r') as f:
            for line in f:
                token, idx = line.strip().split()
                smiles_vocab[token] = int(idx)
    except FileNotFoundError as e:
        print(f"Vocabulary file not found: {filename}")
        print("Please ensure the file exists and the path is correct")
        print("Current working directory:", os.getcwd())
        return None, None
    except Exception as e:
        print(f"Error reading vocabulary file: {e}")
        return None, None
    
    # Verify all required special tokens are present
    required_tokens = {"<start>": 0, "<end>": 1, "<pad>": 2}
    for token, idx in required_tokens.items():
        if token not in smiles_vocab:
            smiles_vocab[token] = idx
    
    vocab_size = len(smiles_vocab)
    return smiles_vocab, vocab_size

def extract_tokens(smiles):
    # Initialize tokenizer from saved file
    tokenizer = PreTrainedTokenizerFast.from_pretrained("../vocab/smiles_tokenizer")
    tokens = tokenizer.encode(smiles, add_special_tokens=True)
    return tokens

def decode_smiles(tokens):
   tokenizer = PreTrainedTokenizerFast.from_pretrained("../vocab/smiles_tokenizer")
   decoded = tokenizer.decode(tokens, skip_special_tokens=True).replace(" ","")
   smiles = ''.join(decoded)
   return smiles

def tokenize_smiles(smiles):
    tokenizer = PreTrainedTokenizerFast.from_pretrained("../vocab/smiles_tokenizer")
    tokens = []
    for smile in smiles:
        tokens.append(tokenizer.encode(smile, add_special_tokens=True))
    return tokens

def pad_token(tokens,max_length,vocab):
    padded_tokens = pad_sequences(tokens, maxlen=max_length, padding='post', value=vocab["<pad>"])
    return padded_tokens

def pad_tokens(tokens,vocab):
    # Find max length including start/end tokens
    max_length = max(len(token) for token in tokens)
    padded_tokens = pad_sequences(tokens, maxlen=max_length, padding='post', value=vocab["<pad>"])
    return padded_tokens, max_length

def make_target(padded_tokens):
    decoder_input = np.array([np.concatenate(([0], seq[:-1])) for seq in padded_tokens])
    decoder_output = np.expand_dims(padded_tokens, axis=-1)
    print("Decoder Output Shape:", decoder_output.shape)
    return decoder_input,decoder_output

def make_training_data(smiles_list):
    smiles_list, groups = extract_groups(smiles_list)
    smiles_vocab, vocab_size = extract_vocab('code/vocab/word_vocab.txt')
    tokens = tokenize_smiles(smiles_list)
    padded_tokens, max_length = pad_tokens(tokens,smiles_vocab)
    decoder_input,decoder_output = make_target(padded_tokens)

    x_smiles = np.array(padded_tokens)
    x_groups = np.array(groups)
    y = np.array(decoder_output)
    print(x_smiles.shape, x_groups.shape, decoder_input.shape, y.shape)
    return x_smiles, x_groups, decoder_input, y, vocab_size, max_length,smiles_vocab







if __name__ == "__main__":
    smiles_list = read_kaggle_smiles()
    smiles_vocab, vocab_size = extract_vocab('code/vocab/word_vocab.txt')
    tokens = tokenize_smiles(smiles_list)
    padded_tokens, max_length = pad_tokens(tokens,smiles_vocab)
    decoder_input,decoder_output = make_target(padded_tokens)
    x_smiles, x_groups, decoder_input, y, vocab_size, max_length,smiles_vocab=make_training_data(smiles_list)
    # smiles_vocab, vocab_size = extract_vocab('code/vocab/word_vocab.txt')
    # print(smiles_list[50])
    # tokens = extract_tokens(smiles_list[50], smiles_vocab)
    # print(tokens)
    # decoded_smiles = decode_smiles(tokens, smiles_vocab)
    # print(decoded_smiles)
    

#     # Test the feature extraction pipeline
#     print(has_thiol_group('CN(C)Cc1ccc(-c2ccc3cnc(Nc4ccc(C5CCN(CC(N)=O)CC5)cc4)nn23)cc1'))
#     print(has_acrylate_group('CN(C)Cc1ccc(-c2ccc3cnc(Nc4ccc(C5CCN(CC(N)=O)CC5)cc4)nn23)cc1'))
#     print(has_vinyl_group('CN(C)Cc1ccc(-c2ccc3cnc(Nc4ccc(C5CCN(CC(N)=O)CC5)cc4)nn23)cc1'))
#     print(has_imine('CN(C)Cc1ccc(-c2ccc3cnc(Nc4ccc(C5CCN(CC(N)=O)CC5)cc4)nn23)cc1'))
#     print(hasEpoxyGroup('CN(C)Cc1ccc(-c2ccc3cnc(Nc4ccc(C5CCN(CC(N)=O)CC5)cc4)nn23)cc1'))
#     print(has_benzene_ring('CN(C)Cc1ccc(-c2ccc3cnc(Nc4ccc(C5CCN(CC(N)=O)CC5)cc4)nn23)cc1'))


