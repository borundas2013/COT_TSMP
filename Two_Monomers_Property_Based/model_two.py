from Constants import *
from metrics import *
from loss import *
from groupAwareLayer import *
from LoadPreTrainedModel import *
from pretrained_weights import *

def create_group_relationship_model(pretrained_model, max_length, vocab_size):
    """
    Create model with explicit group relationship handling
    """
    inputs = [
        tf.keras.layers.Input(shape=(max_length,), name='monomer1_input'),
        tf.keras.layers.Input(shape=(max_length,), name='monomer2_input'),
        tf.keras.layers.Input(shape=(len(Constants.GROUP_VOCAB),), name='group_input'),
        tf.keras.layers.Input(shape=(max_length,), name='decoder_input1'),
        tf.keras.layers.Input(shape=(max_length,), name='decoder_input2')
    ]
    
    monomer1_input, monomer2_input, group_input,decoder_input1,decoder_input2 = inputs
    
    # Extract pretrained features
    pretrained_embedding = pretrained_model.get_layer('embedding')
    # pretrained_encoder = [layer for layer in pretrained_model.layers 
    #                      if 'encoder' in layer.name]
    pretrained_encoder = pretrained_model.get_layer('gru')
    pretrained_decoder_gru = pretrained_model.get_layer('gru_1')
    pretrained_decoder_dense = pretrained_model.get_layer('dense')
    
    # Freeze pretrained layers
    pretrained_embedding.trainable = False
    # for layer in pretrained_encoder:
    #     layer.trainable = False
    pretrained_decoder_gru.trainable = False
    pretrained_decoder_dense.trainable = False
    pretrained_encoder.trainable = False
    
    # Apply pretrained embedding
    monomer1_emb = pretrained_embedding(monomer1_input)
    monomer2_emb = pretrained_embedding(monomer2_input)
    decoder_input1_emb = pretrained_embedding(decoder_input1)
    decoder_input2_emb = pretrained_embedding(decoder_input2)
    print(pretrained_encoder)
    
    # Get chemical features from pretrained model
    def get_chemical_features(x):
        return PretrainedEncoder(max_length=max_length,pretrained_encoder=pretrained_encoder)(x)
    
    monomer1_features = get_chemical_features(monomer1_emb)
    monomer2_features = get_chemical_features(monomer2_emb)
    
    # Create group-aware features
    def create_group_aware_features(features, group_vec):
        """Combine features with group information"""
        group_aware_layer = GroupAwareLayer()
        return group_aware_layer([features, group_vec])
    
    # Apply group-aware features
    group_aware1 = create_group_aware_features(monomer1_features, group_input)
    group_aware2 = create_group_aware_features(monomer2_features, group_input)
    
    # Relationship modeling
    def create_relationship_block():
        return tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(0.1)
        ])
    
    relationship_block = create_relationship_block()
    
    # Process pairs with attention to their relationship
    combined_features = tf.keras.layers.Concatenate()(
        [group_aware1, group_aware2]
    )
    relationship_output = relationship_block(combined_features)
    
    
    
    # Create decoders using pretrained components
    # In create_group_relationship_model function:
    decoder1 = PretrainedDecoder(max_length=max_length,
                                 pretrained_decoder_gru=pretrained_decoder_gru,
                                 pretrained_decoder_dense=pretrained_decoder_dense, name='decoder1')(
        [monomer1_features, decoder_input1_emb, relationship_output]
    )
    decoder2 = PretrainedDecoder(max_length=max_length,
                                 pretrained_decoder_gru=pretrained_decoder_gru,
                                 pretrained_decoder_dense=pretrained_decoder_dense, name='decoder2')(
        [monomer2_features, decoder_input2_emb, relationship_output]
    )
    
    # Create model
    model = tf.keras.Model(
        inputs={
            'monomer1_input': monomer1_input,
            'monomer2_input': monomer2_input,
            'group_input': group_input,
            'decoder_input1': decoder_input1,
            'decoder_input2': decoder_input2
        },
        outputs=[decoder1, decoder2]
    )
    
    # # Compile with loss and metrics
    # model.compile(
    #     optimizer='adam',
    #     loss=CombinedLoss(),
    #    metrics={
    #         'decoder1': [SMILESQualityMetrics(name='monomer1_metrics')],
    #         'decoder2': [SMILESQualityMetrics(name='monomer2_metrics')]
    #     }
    # )
    model.compile(
        optimizer='adam',
        loss={
            'decoder1': CombinedLoss(decoder_name='decoder1'),
            'decoder2': CombinedLoss(decoder_name='decoder2')
        },
        metrics={
            'decoder1': [SMILESQualityMetrics(name='monomer1_metrics')],
            'decoder2': [SMILESQualityMetrics(name='monomer2_metrics')]
        }
    )
    
    return model
