import json
import tensorflow as tf
import keras
from keras.layers import Layer
class ExpandDimsLayer(Layer):
    def call(self, inputs):
        return tf.expand_dims(inputs, axis=1)

class TileLayer(Layer):
    def __init__(self, max_length, **kwargs):
        super(TileLayer, self).__init__(**kwargs)
        self.max_length = max_length

    def call(self, inputs):
        return tf.tile(inputs, [1, self.max_length, 1])

def load_and_retrain(save_dir=""):

    keras.utils.get_custom_objects()["ExpandDimsLayer"] = ExpandDimsLayer
    keras.utils.get_custom_objects()["TileLayer"] = TileLayer
       # Load the vocabulary
    with open(f"{save_dir}/smiles_vocab.json", "r") as f:
        smiles_vocab = json.load(f)
    
    # Load model parameters
    with open(f"{save_dir}/model_params.json", "r") as f:
        model_params = json.load(f)
        print(save_dir)
    
    # Load the model with updated path
    model = tf.keras.models.load_model(f"{save_dir}/model.keras",custom_objects={"ExpandDimsLayer": ExpandDimsLayer,"TileLayer": TileLayer})
    
    return model, smiles_vocab, model_params
