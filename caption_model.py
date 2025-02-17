import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Dense, Embedding, LSTM, Add, Attention, LayerNormalization, Layer
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from config import vocab_size, embedding_dim, OUTPUT_DIM, max_length
"""
main:

from caption_model import caption_model

model = caption_model()
"""

class ExpandDims(Layer):
    
    def __init__(self, axis, **kwargs):
        
        super().__init__(**kwargs)
        self.axis = axis
        
    def call(self, inputs):
        
        return tf.expand_dims(inputs, self.axis)
    
def caption_model(wordtoidx, embeddings_matrix):
    
    # Step 1: Input layers - 保持原有形狀
    inputs1 = Input(shape=(OUTPUT_DIM,), name="image_input")
    inputs2 = Input(shape=(max_length,), name="sequence_input")

    # Step 2: Image feature extraction layers
    # 增加 L2 正則化，調整 dropout rate
    fe1 = Dropout(0.2, name="image_dropout")(inputs1)  # 降低dropout率到0.3
    fe2 = Dense(
        256, 
        activation='relu',
        kernel_regularizer=regularizers.l2(0.001),
        name="image_dense"
        )(fe1)

    # Step 3: Sequence processing layers
    se1 = Embedding(
        vocab_size,
        embedding_dim,
        mask_zero=True,
        name="sequence_embedding"
        )(inputs2)
    se2 = Dropout(0.2, name="sequence_dropout")(se1)  # 降低dropout率到0.3

    # 改進LSTM層
    se3 = LSTM(
        256,
        return_sequences=False,
        recurrent_dropout=0.1,  # 添加recurrent dropout
        name="sequence_lstm_1"
        )(se2)

    # Step 4: Attention mechanism
    # 使用您的自定義ExpandDims層
    fe2_expanded = ExpandDims(axis=1, name="expand_fe2")(fe2)
    se3_expanded = ExpandDims(axis=1, name="expand_se3")(se3)

    # 改進注意力機制
    attention_output = Attention(
        use_scale=True,  # 添加scale
        dropout=0.05,     # 添加attention dropout
        name="attention_layer"
        )([fe2_expanded, se3_expanded])

    attention_output = LayerNormalization(
        epsilon=1e-6,  # 添加epsilon參數
        name="attention_norm"
        )(attention_output)

    attention_output = tf.keras.layers.Lambda(
        lambda x: tf.squeeze(x, axis=1),
        name="squeeze_layer"
        )(attention_output)

    # Step 5: Combine image features with sequence
    combined_features = Add(name="add_image_seq")([fe2, attention_output])

    # Step 6: Decoder
    decoder2 = Dense(
        256,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.001),
        name="decoder_dense"
        )(combined_features)

    decoder2 = Dropout(0.3, name="decoder_dropout")(decoder2)  # 降低dropout率到0.3

    # Step 7: Output layer
    outputs = Dense(
        vocab_size,
        activation='softmax',
        name="output"
        )(decoder2)

    # Step 8: Build the model
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)

    return model

if __name__ == "__main__":
    
    model = caption_model()
    model.summary()