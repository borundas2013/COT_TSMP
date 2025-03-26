# from itertools import combinations
# import numpy as np

# import csv
# from tensorflow.keras.models import Sequential,load_model
# import os
# from tensorflow.keras.layers import Conv1D
# from tensorflow.keras.layers import GlobalMaxPool1D,Dropout,BatchNormalization,Dense
# import tensorflow.keras.backend as K
# import pandas as pd
# import keras
# from rdkit import Chem

# charset=['-', 'F', 'S', '9', 'N', '(', 'l', 'P', 'L', 'T', 'p', 'r', 'A', 'K', 't', ']', '1', 'X', 'R', 'o', '!', 'c', '#', 'C', '+', 'B', 's', 'a', 'H', '8', 'n', '6', '4', '[', '3', ')', '0', '%', 'i', '.', '=', 'g', 'O', 'Z', 'E', '/', '@', 'e', '\\', 'I', 'b', '7', '2', 'M', '5']
# char_to_int = dict((c,i) for i,c in enumerate(charset))
# int_to_char = dict((i,c) for i,c in enumerate(charset))
# latent_dim=256;embed=205

# combined_vetor_all=np.zeros((1,1,latent_dim))

# smiles_to_latent_model=keras.layers.TFSMLayer("Two_Monomers_Property_Based/prediction_model_based/Blog_simple_smi2lat8_150",call_endpoint='serving_default')
# latent_to_states_model=keras.layers.TFSMLayer("Two_Monomers_Property_Based/prediction_model_based/Blog_simple_latstate8_150",call_endpoint='serving_default')
# sample_model=keras.layers.TFSMLayer("Two_Monomers_Property_Based/prediction_model_based/Blog_simple_samplemodel8_150",call_endpoint='serving_default')

# def vector_to_smiles(X):
#     X = X.reshape(1, X.shape[0], X.shape[1], 1)
#     x_latent = smiles_to_latent_model(X)
    
#     # Get the actual tensor from the dictionary output
#     if isinstance(x_latent, dict):
#         x_latent = x_latent['dense_1']  # or whatever the correct key is
    
#     # Get states from latent
#     states = latent_to_states_model(x_latent)
    
#     # Handle dictionary output for states
#     if isinstance(states, dict):
#         state_h = states['dense_2']
#         state_c = states['dense_3']
#         states = [state_h, state_c]
    
#     startidx = char_to_int["!"]
#     samplevec = np.zeros((1,1,len(charset)))
#     samplevec[0,0,startidx] = 1
#     smiles = ""
    
#     for i in range(205):
#         o = sample_model(samplevec)
#         if isinstance(o, dict):
#             o = list(o.values())[0]  # Get the actual output tensor
#         sampleidx = np.argmax(o)
#         samplechar = int_to_char[sampleidx]
#         if samplechar != "E":
#             smiles = smiles + int_to_char[sampleidx]
#             samplevec = np.zeros((1,1,len(charset)))
#             samplevec[0,0,sampleidx] = 1
#         else:
#             break
#     return x_latent
# def split1(word):
#     return [char for char in word]
# def vectorize1(smiles):
#         smiles=split1(smiles)
#         one_hot =  np.zeros(( embed-1 , len(charset)),dtype=np.float32)
#         # for i,smile in enumerate(smiles):
#             #encode the startchar
#         one_hot[0,char_to_int["!"]] = 1
#             #encode the rest of the chars
#         for j,c in enumerate(smiles):
#                 one_hot[j+1,char_to_int[c]] = 1
#             #Encode endchar
#         one_hot[len(smiles)+1:,char_to_int["E"]] = 1
#         #Return two, one for input and the other for output
#         return one_hot#[:,0:-1,:]#, one_hot[:,1:,:]


# def create_neutral_model():
#     Neutral_model = Sequential()
#     Neutral_model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(256,1)))
#     Neutral_model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
#     Neutral_model.add(GlobalMaxPool1D())
#     Neutral_model.add(BatchNormalization())
#     Neutral_model.add(Dropout(0.4))
#     Neutral_model.add(Dense(256, activation='relu'))
#     Neutral_model.add(Dense(64, activation="relu"))
#     Neutral_model.add(Dense(64, activation="relu"))
#     Neutral_model.add(Dense(64, activation="relu"))
#     Neutral_model.add(Dense(32, activation="relu"))
#     Neutral_model.add(Dense(32, activation="relu"))
#     Neutral_model.add(Dense(1, activation="linear"))

#     def root_mean_squared_error(y_true, y_pred):
#         return K.mean(K.abs(y_pred - y_true)/K.abs(y_true))
#     Neutral_model.compile(loss="mae", optimizer='adam',metrics=[root_mean_squared_error])
#     return Neutral_model
     

# def predict_properties(smiles1, smiles2, molar_ratio=0.1):
    
#     # Convert SMILES to vectors
#     vec1 = vectorize1(smiles1)
#     vec2 = vectorize1(smiles2)
#     latent_v1=vector_to_smiles(vec1)
#     combined_vetor=latent_v1*molar_ratio
            
#     latent_v2=vector_to_smiles(vec2)
#     combined_vetor=latent_v1*molar_ratio+latent_v2*(1-molar_ratio)
#     combined_vetor_all[0]=combined_vetor

#     Neutral_model=create_neutral_model()
#     Neutral_model.load_weights('Two_Monomers_Property_Based/prediction_model_based/conv1d_model1_Tg245_3.h5')

#     to_predict=combined_vetor_all[0].reshape(1,256,1)
#     pred_Tg=Neutral_model.predict(to_predict)

#     to_predict=combined_vetor_all[0].reshape(1,256,1)
#     Neutral_model=create_neutral_model()
#     Neutral_model.load_weights('Two_Monomers_Property_Based/prediction_model_based/conv1d_model1_Er245_2.h5')
#     pred_Er=Neutral_model.predict(to_predict)

#     return pred_Tg[0][0], pred_Er[0][0]

# def reward_score(smiles1, smiles2, actual_tg, actual_er):
    
#     # Get predictions
#     try:
#         if Chem.MolFromSmiles(smiles1) is None or Chem.MolFromSmiles(smiles2) is None:
#             return 0.0, 0.0, 0.0
#         pred_tg, pred_er = predict_properties(smiles1, smiles2)
#     except Exception as e:
#         print(f"Reward_Score Error in prediction: {e}")
#         return 0.0, 0.0, 0.0
    
#     # Calculate relative errors
#     tg_error = abs(pred_tg - actual_tg) / abs(actual_tg)
#     er_error = abs(pred_er - actual_er) / abs(actual_er)
    
#     # Calculate individual scores (0 to 1)
#     tg_score = max(0, 1 - tg_error)
#     er_score = max(0, 1 - er_error)
    
#     # Add penalties for extreme deviations (more than 50% error)
#     tg_penalty = max(0, (tg_error - 0.5)) if tg_error > 0.5 else 0
#     er_penalty = max(0, (er_error - 0.5)) if er_error > 0.5 else 0
    
#     # Calculate weighted combined score
#     # Giving slightly more weight to Tg (0.6) than Er (0.4)
#     base_score = (0.5 * tg_score + 0.5 * er_score)
    
#     # Apply penalties
#     penalty_factor = 1.0 - (tg_penalty + er_penalty)
#     final_score = base_score * max(0, penalty_factor)
    
#     # Ensure score is between 0 and 1
#     final_score = max(0, min(1, final_score))
    
#     # Print detailed scores for debugging
#     print(f"Predictions - Tg: {pred_tg:.2f}, Er: {pred_er:.2f}")
#     print(f"Actuals    - Tg: {actual_tg:.2f}, Er: {actual_er:.2f}")
#     print(f"Scores     - Tg: {tg_score:.3f}, Er: {er_score:.3f}")
#     print(f"Final Score: {final_score:.3f}")
    
#     return final_score, tg_score, er_score


# # def reward_score2(smiles1, smiles2, actual_tg, actual_er, weight_tg=0.5, weight_er=0.5):
   
  
# #     # Predict Tg and Er values
# #     predicted_tg, predicted_er = predict_properties(smiles1, smiles2)

# #     # Compute Tg and Er scores with normalization
# #     tg_score = np.exp(-abs(predicted_tg - actual_tg) / actual_tg)  # Exponential decay for smoothness
# #     er_score = np.exp(-abs(predicted_er - actual_er) / actual_er)  # Exponential decay for smoothness

# #     # Weighted sum of both scores
# #     total_reward = (weight_tg * tg_score) + (weight_er * er_score)

# #     return total_reward


# # Example usage:
# if __name__ == "__main__":
#     smiles1 = 'CN(C)Cc1ccc(-c2ccc3cnc(Nc4ccc(C5CCN(CC(N)=O)CC5)cc4)nn23)cc1'
#     smiles2 = 'CCCNC1OC1'
#     actual_tg = 250.0  # example value
#     actual_er = 300.0  # example value
    
#     final_score, tg_score, er_score = reward_score(smiles1, smiles2, actual_tg, actual_er)
#     print(f"\nSummary:")
#     print(f"Overall Score: {final_score:.3f}")
#     print(f"Tg Score: {tg_score:.3f}")
#     print(f"Er Score: {er_score:.3f}")

#     # total_score = reward_score2(smiles1, smiles2, actual_tg, actual_er)
#     # print(f"\nSummary:")
#     # print(f"Total Score: {total_score:.3f}")
    





      


# # import numpy as np
# # from tensorflow import keras
# # from tensorflow.keras.models import Sequential
# # from tensorflow.keras.layers import Conv1D, GlobalMaxPool1D, BatchNormalization, Dense, Dropout
# # import tensorflow.keras.backend as K

# # # Constants
# # charset = ['-', 'F', 'S', '9', 'N', '(', 'l', 'P', 'L', 'T', 'p', 'r', 'A', 'K', 't', ']', '1', 'X', 'R', 'o', '!', 
# #           'c', '#', 'C', '+', 'B', 's', 'a', 'H', '8', 'n', '6', '4', '[', '3', ')', '0', '%', 'i', '.', '=', 'g', 
# #           'O', 'Z', 'E', '/', '@', 'e', '\\', 'I', 'b', '7', '2', 'M', '5']
# # char_to_int = dict((c,i) for i,c in enumerate(charset))
# # int_to_char = dict((i,c) for i,c in enumerate(charset))
# # latent_dim = 256
# # embed = 205

# # # Load models
# # smiles_to_latent_model = keras.layers.TFSMLayer("Two_Monomers_Property_Based/prediction_model_based/Blog_simple_smi2lat8_150", call_endpoint='serving_default')
# # latent_to_states_model = keras.layers.TFSMLayer("Two_Monomers_Property_Based/prediction_model_based/Blog_simple_latstate8_150", call_endpoint='serving_default')
# # sample_model = keras.layers.TFSMLayer("Two_Monomers_Property_Based/prediction_model_based/Blog_simple_samplemodel8_150", call_endpoint='serving_default')

# # def create_neutral_model():
# #     model = Sequential([
# #         Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(256,1)),
# #         Conv1D(filters=64, kernel_size=3, activation='relu'),
# #         GlobalMaxPool1D(),
# #         BatchNormalization(),
# #         Dense(256, activation='relu'),
# #         Dense(64, activation="relu"),
# #         Dense(64, activation="relu"),
# #         Dense(64, activation="relu"),
# #         Dense(32, activation="relu"),
# #         Dense(32, activation="relu"),
# #         Dense(1, activation="linear")
# #     ])
# #     model.compile(loss="mae", optimizer='adam')
# #     return model

# # def vectorize1(smiles):
# #     one_hot = np.zeros((embed-1, len(charset)), dtype=np.float32)
# #     smiles_chars = [char for char in smiles]
# #     one_hot[0, char_to_int["!"]] = 1
# #     for j, c in enumerate(smiles_chars):
# #         one_hot[j+1, char_to_int[c]] = 1
# #     one_hot[len(smiles_chars)+1:, char_to_int["E"]] = 1
# #     return one_hot

# # def vector_to_smiles(X):
# #     X = X.reshape(1, X.shape[0], X.shape[1], 1)
# #     x_latent = smiles_to_latent_model(X)
# #     if isinstance(x_latent, dict):
# #         x_latent = x_latent['dense_1']
# #     return x_latent

# # def predict_properties(smiles1, smiles2, molar_ratio=0.1):
# #     """
# #     Predict properties for two SMILES strings with given molar ratio.
    
# #     Args:
# #         smiles1 (str): First SMILES string
# #         smiles2 (str): Second SMILES string
# #         molar_ratio (float): Molar ratio between 0 and 1 (default: 0.1)
        
# #     Returns:
# #         tuple: (Tg prediction, Er prediction)
# #     """
# #     # Convert SMILES to vectors
# #     vec1 = vectorize1(smiles1)
# #     vec2 = vectorize1(smiles2)
    
# #     # Get latent vectors
# #     latent_v1 = vector_to_smiles(vec1)
# #     latent_v2 = vector_to_smiles(vec2)
    
# #     # Convert to numpy if needed
# #     if hasattr(latent_v1, 'numpy'):
# #         latent_v1 = latent_v1.numpy()
# #     if hasattr(latent_v2, 'numpy'):
# #         latent_v2 = latent_v2.numpy()
    
# #     # Combine vectors based on molar ratio
# #     combined_vector = latent_v1 * molar_ratio + latent_v2 * (1 - molar_ratio)
    
# #     # Reshape using tf.reshape or after converting to numpy
# #     to_predict = np.array(combined_vector).reshape(1, 256, 1)
    
# #     # Create and load models for prediction
# #     neutral_model = create_neutral_model()
    
# #     # Predict Tg
# #     neutral_model.load_weights('Two_Monomers_Property_Based/prediction_model_based/conv1d_model1_Tg245_3.h5')
# #     pred_tg = neutral_model.predict(to_predict, verbose=0)
    
# #     # Predict Er
# #     neutral_model.load_weights('Two_Monomers_Property_Based/prediction_model_based/conv1d_model1_Er245_2.h5')
# #     pred_er = neutral_model.predict(to_predict, verbose=0)
    
# #     return pred_tg[0][0], pred_er[0][0]

# # # Example usage:
# # if __name__ == "__main__":
# #     smiles1 = 'CN(C)Cc1ccc(-c2ccc3cnc(Nc4ccc(C5CCN(CC(N)=O)CC5)cc4)nn23)cc1'
# #     smiles2 = 'CCCNC1OC1'
# #     tg, er = predict_properties(smiles1, smiles2, molar_ratio=0.1)
# #     print(f"Predicted Tg: {tg:.2f}")
# #     print(f"Predicted Er: {er:.2f}")