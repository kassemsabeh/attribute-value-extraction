import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer


class GloveEmbeddings():

    def __init__(self, dim=100) -> None:
        self.dim = dim
    
    def __create_glove_embeddings(self):
        name = "embeddings/glove/glove.6B." + str(self.dim) + "d.txt"
        embedding_dict = {}
        with open(name, "r") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")

                embedding_dict[word] = vector
        
        return embedding_dict
    
    def create_embeddings(self, token_num : int, word_tokenizer : Tokenizer):
        embedding_dict = self.__create_glove_embeddings()
        embedding_matrix = np.zeros((token_num, self.dim))
        for word, i in word_tokenizer.word_index.items():
            embedding_vector = embedding_dict.get(word)

            if (embedding_vector is not None):
                embedding_matrix[i] = embedding_vector
        
        return embedding_matrix
            
