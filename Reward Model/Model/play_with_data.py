import ast

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import BondType
from rdkit.Chem.Draw import MolsToGridImage

import os
import math






RDLogger.DisableLog("rdApp.*")

df = pd.read_excel('/home/C00521897/Fall 22/New_Monomer_generation/Data/smiles.xlsx')
df.head()

SMILE_CHARSET = '["C", "B", "F", "I", "H", "O", "N", "S", "P", "Cl", "Br","Si"]'


bond_mapping = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}
bond_mapping.update(
    {0: BondType.SINGLE, 1: BondType.DOUBLE, 2: BondType.TRIPLE, 3: BondType.AROMATIC}
)
SMILE_CHARSET = ast.literal_eval(SMILE_CHARSET)
print(SMILE_CHARSET)
MAX_MOLSIZE = max(df["SMILES"].str.len())
SMILE_to_index = dict((c, i) for i, c in enumerate(SMILE_CHARSET))
index_to_SMILE = dict((i, c) for i, c in enumerate(SMILE_CHARSET))
atom_mapping = dict(SMILE_to_index)
print(atom_mapping)
atom_mapping.update(index_to_SMILE)
print(atom_mapping)


BATCH_SIZE = 32
EPOCHS = 10

VAE_LR = 5e-4
NUM_ATOMS = 160  # Maximum number of atoms

ATOM_DIM = len(SMILE_CHARSET)  # Number of atom types
BOND_DIM = 5  # Number of bond types
LATENT_DIM = 512 #435 #600  # Size of the latent space

df_smiles=[]
for idx in range(len(df['SMILES'])):
    smiles = df["SMILES"][idx].split(',')
    if len(smiles) == 2:
        df_smiles.append(smiles)
print(len(df_smiles))

def smiles_to_graph2(smiles):
    adjecenies_array = []
    features_array = []
    for i in range (2):
        #print(i)
        molecule=Chem.MolFromSmiles(smiles[i])
        adjacency = np.zeros((BOND_DIM,NUM_ATOMS,NUM_ATOMS), 'float32')
        features =  np.zeros((NUM_ATOMS,ATOM_DIM), 'float32')
        for atom in molecule.GetAtoms():
            #print(atom.GetSymbol())
            i = atom.GetIdx()
            atom_type=atom_mapping[atom.GetSymbol()]
            features[i] =  np.eye(ATOM_DIM)[atom_type]
            for neighbor in atom.GetNeighbors():
                j = neighbor.GetIdx()
                bond = molecule.GetBondBetweenAtoms(i,j)
                bond_type_idx =  bond_mapping[bond.GetBondType().name]
                adjacency[bond_type_idx,[i,j],[j,i]] = 1
        adjacency[-1,np.sum(adjacency, axis=0)== 0] = 1
        features[np.where(np.sum(features,axis=1) == 0 )[0],-1] = 1
        adjecenies_array.append(adjacency)
        features_array.append(features)
    return adjecenies_array[0],features_array[0],adjecenies_array[1],features_array[1]

def formMol(adjacency, features):
    molecule = Chem.RWMol()
    keep_idx = np.where (
        (np.argmax(features,axis=1) != ATOM_DIM - 1) &
        (np.sum(adjacency[:-1],axis=(0,1)) !=0 )
    )[0]
    
    features = features[keep_idx]
    adjacency = adjacency[:,keep_idx,:][:,:,keep_idx]

    for atom_type_idx in np.argmax(features, axis=1):
        atom = Chem.Atom(atom_mapping[atom_type_idx])
        #print("fpr", atom.GetSymbol())
        _ = molecule.AddAtom(atom)
    (bonds_ij, atoms_i, atoms_j) = np.where(np.triu(adjacency) == 1)
    for (bond_ij,atom_i,atom_j) in zip(bonds_ij,atoms_i,atoms_j):
        if atom_i == atom_j or bond_ij == BOND_DIM - 1:
            continue
        bond_type = bond_mapping[bond_ij]
        molecule.AddBond(int(atom_i), int(atom_j),bond_type)
        #print(atom_i,atom_j,bond_type)

    flag = Chem.SanitizeMol(molecule,catchErrors=True)
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
        return None
    return molecule

def graph_to_molecule2(graphs):
    adjecenies_array_0, features_array_0,adjecenies_array_1, features_array_1=graphs
    all_mols=[]
    for i in range(len(adjecenies_array_0)):
       
        adjacency_0, features_0,adjacency_1, features_1 = adjecenies_array_0[i],features_array_0[i],adjecenies_array_1[i],features_array_1[i]
        mols=[formMol(adjacency_0, features_0),formMol(adjacency_1, features_1)]
        all_mols.append(mols)
    return all_mols

adjacenices_array = []
features_array = []

adjacency_0, features_0,adjacency_1,features_1 = [],[],[],[]
for idx in range(len(df['SMILES'])):
    smiles = df["SMILES"][idx].split(',')
    if len(smiles) == 2:
        adj_0, feat_0,adj_1,feat_1 = smiles_to_graph2(smiles)
        adjacency_0.append(adj_0)
        features_0.append(feat_0)
        adjacency_1.append(adj_1)
        features_1.append(feat_1)

graph= [adjacency_0, features_0,adjacency_1,features_1]


mols_all = graph_to_molecule2(graph)

#train_df = df.sample(frac=1, random_state=42)
#train_df.reset_index(drop=True,inplace=True)
adjacency_0_tensor = np.array(adjacency_0)
adjacency_1_tensor = np.array(adjacency_1)
feature_0_tensor = np.array(features_0)
feature_1_tensor = np.array(features_1)
# #feature_tensor = np.array(features_array)
# print(adjacency_0_tensor.shape,adjacency_1_tensor.shape,feature_0_tensor.shape,feature_1_tensor.shape)
#print(feature_tensor.shape)

print(len(adjacency_0_tensor))
print(df_smiles[1250])
print(Chem.MolToSmiles(mols_all[1250][0]),Chem.MolToSmiles(mols_all[1250][1]))

class RelationalGraphConvLayer(keras.layers.Layer):
  def __init__(self, units=128,activation='relu',use_bias=False,kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None, **kwargs):
    super().__init__( **kwargs)
    self.units = units
    self.activation = keras.activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = keras.initializers.get(kernel_initializer)
    self.bias_initializer = keras.initializers.get(bias_initializer)
    self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
    self.bias_regularizer = keras.regularizers.get(bias_regularizer)

  def build(self,input_shape):
    bond_dim = input_shape[0][1]
    atom_dim = input_shape[1][2]
    self.kernel = self.add_weight(shape=(bond_dim, atom_dim, self.units),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  trainable=True,
                                  name="W",
                                  dtype=tf.float32,
                                  )

    if self.use_bias:
      self.bias = self.add_weight(shape = (bond_dim, 1, self.units),
                                  initializer=self.bias_initializer,
                                  regularizer=self.bias_regularizer,
                                  trainable=True,
                                  name="b",
                                  dtype=tf.float32)
      self.built = True

  def call(self,inputs,training=False):
    adjacency_0, features_0,adjacency_1, features_1 = inputs
    x1 = tf.matmul(adjacency_0,features_0[:,None,:,:])
    x2 = tf.matmul(adjacency_1,features_1[:,None,:,:])
    x=tf.matmul(x1,self.kernel)
    y= tf.matmul(x2, self.kernel)
   
    if self.use_bias:
      x += self.bias
      y += self.bias

    x_reduced = tf.reduce_sum(x, axis=1)
    y_reduced = tf.reduce_sum(y,axis=1)
    return self.activation(x_reduced) , self.activation(y_reduced)

def get_encoder(gconv_units,latent_dim,adjacency_shape,feature_shape,dense_units,dropout_rate):
  adjacency_0 = keras.layers.Input(shape=adjacency_shape)
  features_0 = keras.layers.Input(shape=feature_shape)
  adjacency_1 = keras.layers.Input(shape=adjacency_shape)
  features_1 = keras.layers.Input(shape=feature_shape)

  features_transformed_0=  features_0
  features_transformed_1 =  features_1
 
  for units in gconv_units:
    features_transformed_0,features_transformed_1 = RelationalGraphConvLayer(units) ([adjacency_0,features_transformed_0,adjacency_1,features_transformed_1])

  x = keras.layers.GlobalAveragePooling1D()(features_transformed_0)
  y = keras.layers.GlobalAveragePooling1D()(features_transformed_1)
  for units in dense_units:
    x = layers.Dense(units,activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    y = layers.Dense(units,activation='relu')(y)
    y = layers.Dropout(dropout_rate)(y)
  z_mean = layers.Dense(latent_dim,dtype='float32',name='z_mean')(x)
  log_var = layers.Dense(latent_dim, dtype='float32',name='log_var')(x)
  z_mean_1 = layers.Dense(latent_dim,dtype='float32',name='z_mean_1')(y)
  log_var_1 = layers.Dense(latent_dim, dtype='float32',name='log_var_1')(y)
  encoder= keras.Model([adjacency_0,features_0,adjacency_1,features_1],[z_mean,log_var,z_mean_1,log_var_1], name='encoder')

  return encoder

def get_decoder(dense_units, dropout_rate, latent_dim, adjacency_shape, feature_shape):
    latent_inputs = keras.Input(shape=(latent_dim,))

    x = latent_inputs
    for units in dense_units:
        x = keras.layers.Dense(units, activation="tanh")(x)
        x = keras.layers.Dropout(dropout_rate)(x)

    # Map outputs of previous layer (x) to [continuous] adjacency tensors (x_adjacency)
    x_0_adjacency = keras.layers.Dense(tf.math.reduce_prod(adjacency_shape))(x)
    x_0_adjacency = keras.layers.Reshape(adjacency_shape)(x_0_adjacency)
    # Symmetrify tensors in the last two dimensions
    x_0_adjacency = (x_0_adjacency + tf.transpose(x_0_adjacency, (0, 1, 3, 2))) / 2
    x_0_adjacency = keras.layers.Softmax(axis=1)(x_0_adjacency)

    # Map outputs of previous layer (x) to [continuous] feature tensors (x_features)
    x_0_features = keras.layers.Dense(tf.math.reduce_prod(feature_shape))(x)
    x_0_features = keras.layers.Reshape(feature_shape)(x_0_features)
    x_0_features = keras.layers.Softmax(axis=2)(x_0_features)
    
    
    # Map outputs of previous layer (x) to [continuous] adjacency tensors (x_adjacency)
    x_1_adjacency = keras.layers.Dense(tf.math.reduce_prod(adjacency_shape))(x)
    x_1_adjacency = keras.layers.Reshape(adjacency_shape)(x_1_adjacency)
    # Symmetrify tensors in the last two dimensions
    x_1_adjacency = (x_1_adjacency + tf.transpose(x_1_adjacency, (0, 1, 3, 2))) / 2
    x_1_adjacency = keras.layers.Softmax(axis=1)(x_1_adjacency)

    # Map outputs of previous layer (x) to [continuous] feature tensors (x_features)
    x_1_features = keras.layers.Dense(tf.math.reduce_prod(feature_shape))(x)
    x_1_features = keras.layers.Reshape(feature_shape)(x_1_features)
    x_1_features = keras.layers.Softmax(axis=2)(x_1_features)

    decoder = keras.Model(
        latent_inputs, outputs=[x_0_adjacency, x_0_features,x_1_adjacency, x_1_features], name="decoder"
    )

    return decoder
    
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var,z_mean_1, z_log_var_1 = inputs
        batch = tf.shape(z_log_var)[0]
        dim = tf.shape(z_log_var)[1]
        print(batch)
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean +z_mean_1 + tf.exp(0.5 * z_log_var) * epsilon +  tf.exp(0.5 * z_log_var_1) * epsilon

class MoleculeGenerator(keras.Model):
  def __init__(self,encoder,decoder,max_len,**kwargs):
    super().__init__(**kwargs)
    self.encoder = encoder
    self.decoder = decoder
    self.max_len = max_len

    self.train_total_loss_tracker = keras.metrics.Mean(name="train_total_loss")
    self.val_total_loss_tracker = keras.metrics.Mean(name="val_total_loss")

  def _gradient_penalty(self, graph_real, graph_generated):
    # Unpack graphs
    adjacency_0_real, features_0_real,adjacency_1_real, features_1_real = graph_real
    adjacency_0_generated, features_0_generated,adjacency_1_generated, features_1_generated = graph_generated
   
    # Generate interpolated graphs (adjacency_interp and features_interp)
    alpha = tf.random.uniform([self.batch_size])
    alpha = tf.reshape(alpha, (self.batch_size, 1, 1, 1))
    print(alpha.shape, adjacency_0_real.shape)
    adjacency_0_interp = (adjacency_0_real * alpha) + (1 - alpha) * adjacency_0_generated
    adjacency_1_interp = (adjacency_1_real * alpha) + (1 - alpha) * adjacency_1_generated
    alpha = tf.reshape(alpha, (self.batch_size, 1, 1))
    features_0_interp = (features_0_real * alpha) + (1 - alpha) * features_0_generated
    features_1_interp = (features_1_real * alpha) + (1 - alpha) * features_1_generated

    # Compute the logits of interpolated graphs
    with tf.GradientTape() as tape:
        tape.watch(adjacency_0_interp)
        tape.watch(features_0_interp)
        tape.watch(adjacency_1_interp)
        tape.watch(features_1_interp)
        print(self(
            [adjacency_0_interp, features_0_interp,adjacency_1_interp,features_1_interp], training=True
        ))
        _,_,_,_,_,_,_,_=self(
            [adjacency_0_interp, features_0_interp,adjacency_1_interp,features_1_interp], training=True
        )

    # Compute the gradients with respect to the interpolated graphs
    grads = tape.gradient(0, [adjacency_0_interp, features_0_interp,adjacency_1_interp,features_1_interp])
    
    # Compute the gradient penalty
    grads_adjacency_0_penalty = (1 - tf.norm(grads[0], axis=1)) ** 2
    grads_features_0_penalty = (1 - tf.norm(grads[1], axis=2)) ** 2
    grads_adjacency_1_penalty = (1 - tf.norm(grads[2], axis=1)) ** 2
    grads_features_1_penalty = (1 - tf.norm(grads[3], axis=2)) ** 2
    return tf.reduce_mean(
        tf.reduce_mean(grads_adjacency_0_penalty, axis=(-2, -1))
        + tf.reduce_mean(grads_features_0_penalty, axis=(-1)) 
        +  tf.reduce_mean(grads_adjacency_1_penalty, axis=(-2, -1)) # need to check
        +  tf.reduce_mean(grads_features_1_penalty, axis=(-1)) # need to check
    )

  def train_step (self,data) :
    adjacency_0_tensor, feature_0_tensor,adjacency_1_tensor, feature_1_tensor = data[0]
    graph_real = [adjacency_0_tensor, feature_0_tensor,adjacency_1_tensor, feature_1_tensor]
    self.batch_size = tf.shape(adjacency_0_tensor)[0]

    with tf.GradientTape() as tape:
      z_mean,z_log_var, z_mean_1,z_log_var_1,gen_0_adjacency,gen_0_features,gen_1_adjacency,gen_1_features = self(graph_real,training=True)
      graph_generated=[gen_0_adjacency,gen_0_features,gen_1_adjacency,gen_1_features]
      total_loss = self._compute_loss(
          z_log_var,z_mean,z_mean_1,z_log_var_1,graph_real,graph_generated
      )
    grads = tape.gradient(total_loss, self.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

    self.train_total_loss_tracker.update_state(total_loss)
    return {'loss' : self.train_total_loss_tracker.result()}
  def cosine_similarity_loss(self,y_true, y_pred):
    y_true = tf.nn.l2_normalize(y_true, axis=-1)
    y_pred = tf.nn.l2_normalize(y_pred, axis=-1)
    return -tf.reduce_mean(tf.reduce_sum(y_true * y_pred, axis=-1))

  def _compute_loss(self, z_log_var, z_mean,z_mean_1,z_log_var_1, graph_real, graph_generated) :
    alpha = 0.01
    adjacency_0_real, features_0_real,adjacency_1_real, features_1_real = graph_real
    adjacency_0_gen, features_0_gen,adjacency_1_gen, features_1_gen = graph_generated
  

    adjacency_0_loss = tf.reduce_mean(
        tf.reduce_sum(
                keras.losses.categorical_crossentropy(adjacency_0_real, adjacency_0_gen),
                axis=(1, 2),
            )
        )
    adj_0_similarity= self.cosine_similarity_loss(adjacency_0_real,adjacency_0_gen)
    adjacency_0_loss = adjacency_0_loss + alpha*adj_0_similarity
    adjacency_1_loss = tf.reduce_mean(
        tf.reduce_sum(
                keras.losses.categorical_crossentropy(adjacency_1_real, adjacency_1_gen),
                axis=(1, 2),
            )
        )
    adj_1_similarity= self.cosine_similarity_loss(adjacency_1_real,adjacency_1_gen)
    adjacency_1_loss = adjacency_1_loss + alpha*adj_1_similarity
    features_0_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.categorical_crossentropy(features_0_real, features_0_gen),
                axis=(1),
            )
        )
    feat_0_similarity= self.cosine_similarity_loss(features_0_real,features_0_gen)
    features_0_loss = features_0_loss + alpha*feat_0_similarity
    features_1_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.categorical_crossentropy(features_1_real, features_1_gen),
                axis=(1),
            )
        )
    feat_1_similarity= self.cosine_similarity_loss(features_1_real,features_1_gen)
    features_1_loss = features_1_loss + alpha*feat_1_similarity
    
    kl_loss_0 = -0.5 * tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), 1
        )
    kl_loss_0 = tf.reduce_mean(kl_loss_0)
    kl_loss_1 = -0.5 * tf.reduce_sum(
            1 + z_log_var_1 - tf.square(z_mean_1) - tf.exp(z_log_var_1), 1
        )
    kl_loss_1 = tf.reduce_mean(kl_loss_1)
    #graph_loss = self._gradient_penalty(graph_real, graph_generated) 
    total_loss = kl_loss_0 +kl_loss_1 +  adjacency_0_loss + features_0_loss +adjacency_1_loss + features_1_loss#+graph_loss
    
    # if checkFunctionalGroups([[adjacency_0_gen], [features_0_gen],[adjacency_1_gen], [features_1_gen]]):
    #     total_loss= total_loss ** 0.5
    return total_loss 



  def inference(self,batch_size,sample) :
    ad_0,feat_0,ad_1,feat_1=smiles_to_graph2(sample)
    z_mean, log_var,z_mean_1, log_var_1=model.encoder([np.array([ad_0]),np.array([feat_0]),np.array([ad_1]),np.array([feat_1])])
    z1= Sampling()([z_mean,log_var,z_mean_1, log_var_1])
    all_mols=[]
    for i in range(batch_size):
        z = tf.random.normal((1,LATENT_DIM))
        z2= np.multiply((z1 * (-1)) , z) 
        reconstruction_adjacnency_0, recontstruction_features_0,reconstruction_adjacnency_1, recontstruction_features_1 = model.decoder.predict(z2)
        adjacency_0 = tf.argmax(reconstruction_adjacnency_0,axis=1)
        adjacency_0 =  tf.one_hot(adjacency_0,depth=BOND_DIM,axis=1)
        adjacency_0 = tf.linalg.set_diag(adjacency_0, tf.zeros(tf.shape(adjacency_0)[:-1]))
        features_0 = tf.argmax(recontstruction_features_0,axis=2)
        features_0 = tf.one_hot(features_0, depth=ATOM_DIM,axis=2)
    
    
        adjacency_1 = tf.argmax(reconstruction_adjacnency_1,axis=1)
        adjacency_1 =  tf.one_hot(adjacency_1,depth=BOND_DIM,axis=1)
        adjacency_1 = tf.linalg.set_diag(adjacency_1, tf.zeros(tf.shape(adjacency_1)[:-1]))
        features_1 = tf.argmax(recontstruction_features_1,axis=2)
        features_1 = tf.one_hot(features_1, depth=ATOM_DIM,axis=2)
        graph2=[[adjacency_0[0].numpy()],[features_0[0].numpy()],[adjacency_1[0].numpy()],[features_1[0].numpy()]]
        all_mols.append(graph_to_molecule2(graph2))
    return all_mols


  def call(self,inputs):
    z_mean, log_var,z_mean_1, log_var_1 = self.encoder(inputs)
    z= Sampling()([z_mean,log_var,z_mean_1, log_var_1])
    gen_adjacency_0, gen_features_0,gen_adjacency_1, gen_features_1 = self.decoder(z)

    return z_mean, log_var,z_mean_1,log_var_1,gen_adjacency_0, gen_features_0,gen_adjacency_1, gen_features_1

vae_optimizer = tf.keras.optimizers.Adam(learning_rate=VAE_LR)

encoder = get_encoder(
    gconv_units=[9],
    adjacency_shape=(BOND_DIM, NUM_ATOMS, NUM_ATOMS),
    feature_shape=(NUM_ATOMS, ATOM_DIM),
    latent_dim=LATENT_DIM,
    dense_units=[128,256,512],
    dropout_rate=0.2,
)
decoder = get_decoder(
    dense_units=[128,256, 512,512,1024],
    dropout_rate=0.2,
    latent_dim=LATENT_DIM,
    adjacency_shape=(BOND_DIM, NUM_ATOMS, NUM_ATOMS),
    feature_shape=(NUM_ATOMS, ATOM_DIM),
)




model = MoleculeGenerator(encoder, decoder, MAX_MOLSIZE)


model.compile(vae_optimizer)

checkpoint_path = "/home/C00521897/Fall 22/New_Monomer_generation/checkpoints/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)



def check_mistmatch(original, generated):
    mismatches =[[]]
    generated=generated[0].numpy()
    for i in range(len(original)):
        m=[]
        for j in range(len(original[0])):
            if original[i][j]!=generated[i][j]:
                generated[i][j] = original[i][j]
                m.append(j)
        mismatches.append(m)
    return generated

def check_mistmatch2(original, generated):
    mismatches =[[]]
    generated=generated[0].numpy()
    for k in range(5):
         for i in range(NUM_ATOMS):
            m=[]
            for j in range(NUM_ATOMS):
                if original[k][i][j]!=generated[k][i][j]:
                    generated[k][i][j]=original[k][i][j]
                    
            mismatches.append(m)
    return generated

def test_model():
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    print(latest)
    model.load_weights(latest)
    smi=['c2cc(OCC1CO1)ccc2Cc4ccc(OCC3CO3)cc4', 'NCCOCCOCCN']
    print(smi)
    a0,f0,a1,f1=smiles_to_graph2(smi)
    graph =[[a0],[f0],[a1],[f1]]
    mols=graph_to_molecule2(graph)
    print("Original: ",Chem.MolToSmiles(mols[0][0]),Chem.MolToSmiles(mols[0][1]))
    z_mean, log_var,z_mean_1, log_var_1=model.encoder([np.array([a0]),np.array([f0]),np.array([a1]),np.array([f1])])
    z1= Sampling()([z_mean,log_var,z_mean_1, log_var_1])
    reconstruction_adjacnency_0, recontstruction_features_0,reconstruction_adjacnency_1, recontstruction_features_1 = model.decoder.predict(z1)
    adjacency_0 = tf.argmax(reconstruction_adjacnency_0,axis=1)
    adjacency_0 =  tf.one_hot(adjacency_0,depth=BOND_DIM,axis=1)
    adjacency_0 = tf.linalg.set_diag(adjacency_0, tf.zeros(tf.shape(adjacency_0)[:-1]))
    features_0 = tf.argmax(recontstruction_features_0,axis=2)
    features_0 = tf.one_hot(features_0, depth=ATOM_DIM,axis=2)
    
    
    adjacency_1 = tf.argmax(reconstruction_adjacnency_1,axis=1)
    adjacency_1 =  tf.one_hot(adjacency_1,depth=BOND_DIM,axis=1)
    adjacency_1 = tf.linalg.set_diag(adjacency_1, tf.zeros(tf.shape(adjacency_1)[:-1]))
    features_1 = tf.argmax(recontstruction_features_1,axis=2)
    features_1 = tf.one_hot(features_1, depth=ATOM_DIM,axis=2)
    graph2=[[adjacency_0[0].numpy()],[features_0[0].numpy()],[adjacency_1[0].numpy()],[features_1[0].numpy()]]
    all_mols=graph_to_molecule2(graph2)
    print(all_mols)
    #print("Predicted: ",Chem.MolToSmiles(all_mols[0][0]),Chem.MolToSmiles(all_mols[0][1]))
    
    feats_0 = check_mistmatch(f0,features_0)
    feats_1 = check_mistmatch(f1,features_1)
    a_0=check_mistmatch2(a0,adjacency_0)
    a_1=check_mistmatch2(a1,adjacency_1)
    graph3=[[a_0],[feats_0],[a_1],[feats_1]]
    all_mols2=graph_to_molecule2(graph3)
    #print(all_mols2)
    print("Predicted: ",Chem.MolToSmiles(all_mols2[0][0]),Chem.MolToSmiles(all_mols2[0][1]))
                
                
    
    #print("Mismatched Items:", mismatches)
test_model()
