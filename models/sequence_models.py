import sys
sys.path.append('.')

import tensorflow as tf
from tensorflow.keras import layers
from tf2crf import CRF, ModelWithCRFLoss
from preprocessing.sequences import SequencePreprocessor
from layers.attention import AttentionLayer



class LstmCrf(tf.keras.Model):

    def __init__(self, processor:SequencePreprocessor, embedding_matrix, lstm_units=128, dropout=0.4, embedding_dim=100) -> None:
        super(LstmCrf, self).__init__()
        self.embedded_sequences = layers.Embedding(processor.token_num, embedding_dim, weights=[embedding_matrix], trainable=True)
        self.bidirectional = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))
        self.dropout = layers.Dropout(dropout)
        self.crf = CRF(units=processor.tag_num)
    
    def call(self, inputs):
        x = self.embedded_sequences(inputs)
        x = self.bidirectional(x)
        x = self.dropout(x)
        return self.crf(x)

class OpenTag(tf.keras.Model):

    def __init__(self, processor : SequencePreprocessor, embedding_matrix, lstm_units=128, dropout=0.4, embedding_dim=100) -> None:
        super(OpenTag, self).__init__()
        self.embedded_sequences = layers.Embedding(processor.token_num, embedding_dim, weights=[embedding_matrix], trainable=True)
        self.bidirectional = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))
        self.attention = AttentionLayer()
        self.dropout = layers.Dropout(dropout)
        self.crf = CRF(units=processor.tag_num)
    
    def call(self, inputs):
        x = self.embedded_sequences(inputs)
        x = self.bidirectional(x)
        x = self.attention(x)
        x = self.dropout(x)
        return self.crf(x)

class OpenBrandCNN(tf.keras.Model):
    
    def __init__(self, processor: SequencePreprocessor, embedding_matrix, lstm_units=128, dropout=0.4, embedding_dim=100, char_embed_size=30, 
    window_size=3, filters=30, char_lstm_units=25) -> None:
        super(OpenBrandCNN, self).__init__()
        self.word_embedding = layers.Embedding(processor.token_num, 100, input_length=processor.max_len, weights=[embedding_matrix], trainable=True)
        self.char_embedding = layers.TimeDistributed(layers.Embedding(len(processor.char2idx), char_embed_size, input_length=processor.char_max_len))
        self.conv = layers.TimeDistributed(layers.Conv1D(filters, window_size, padding='same'))
        self.max_pool = layers.TimeDistributed(layers.GlobalMaxPooling1D())
        self.spatial_dropout = layers.SpatialDropout1D(dropout)
        self.bidirectional = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True, recurrent_dropout=0.6))
        self.dropout = layers.Dropout(dropout)
        self.attention = AttentionLayer()
        self.crf = CRF(units=processor.tag_num)

    def call(self, inputs):
        word_input = inputs[0]
        char_input = inputs[1]

        x1 = self.word_embedding(word_input)
        x2 = self.char_embedding(char_input)
        x2 = self.conv(x2)
        x2 = self.max_pool(x2)
        x = layers.concatenate([x1, x2])
        x = self.spatial_dropout(x)
        x = self.bidirectional(x)
        x = self.dropout(x)
        x = self.attention(x)
        return self.crf(x)
    


class OpenBrandLSTM(tf.keras.Model):
    
    def __init__(self, processor: SequencePreprocessor, embedding_matrix, lstm_units=128, dropout=0.4, embedding_dim=100, char_embed_size=30, char_lstm_units=25) -> None:
        super(OpenBrandCNN, self).__init__()
        self.word_embedding = layers.Embedding(processor.token_num, 100, input_length=processor.max_len, weights=[embedding_matrix], trainable=True)
        self.char_embedding = layers.TimeDistributed(layers.Embedding(len(processor.char2idx), char_embed_size, input_length=processor.char_max_len))
        self.char_bidirectional = layers.TimeDistributed(layers.Bidirectional(layers.LSTM(units=char_lstm_units, return_sequences=True, recurrent_dropout=0.5), merge_mode='concat'))
        self.flatten = layers.TimeDistributed(layers.Flatten())
        self.spatial_dropout = layers.SpatialDropout1D(dropout)
        self.bidirectional = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True, recurrent_dropout=0.6))
        self.dropout = layers.Dropout(dropout)
        self.attention = AttentionLayer()
        self.crf = CRF(units=processor.tag_num)

    def call(self, inputs):
        word_input = inputs[0]
        char_input = inputs[1]

        x1 = self.word_embedding(word_input)
        x2 = self.char_embedding(char_input)
        x2 = self.char_bidirectional(x2)
        x2 = self.flatten(x2)
        x = layers.concatenate([x1, x2])
        x = self.spatial_dropout(x)
        x = self.bidirectional(x)
        x = self.dropout(x)
        x = self.attention(x)
        return self.crf(x)





    
    

