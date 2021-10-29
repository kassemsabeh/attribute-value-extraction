from typing import Sequence
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class SequencePreprocessor():

    def __init__(self) -> None:
        pass

    def __train_test_split(self, sentences, tags):
        self.train_sentences, self.test_sentences, self.train_tags, self.test_tags = train_test_split(sentences, tags, 
        test_size=0.3, random_state=42)
        self.dev_sentences, self.test_sentences, self.dev_tags, self.test_tags = train_test_split(self.test_sentences, self.test_tags, 
        test_size=0.5, random_state=42)
        self.max_len = max(len(s) for s in self.train_sentences)

    def process_word_sequences(self, sentences, tags):
        self.__train_test_split(sentences, tags)
        self.word_tokenizer = Tokenizer(filters=[], lower=True, oov_token='<unk>')
        self.word_tokenizer.fit_on_texts([" ".join(s) for s in self.train_sentences])
        self.word_tokenizer.word_index['<pad>'] = 0
        self.word_tokenizer.index_word[0] = '<pad>'
        # Transform text to integers
        train_seqs = self.word_tokenizer.texts_to_sequences([" ".join(s) for s in self.train_sentences])
        dev_seqs = self.word_tokenizer.texts_to_sequences([" ".join(s) for s in self.dev_sentences])
        test_seqs = self.word_tokenizer.texts_to_sequences([" ".join(s) for s in self.test_sentences])
        # Pad all sequences to same length
        self.X_train = pad_sequences(train_seqs, maxlen=self.max_len)
        self.X_dev = pad_sequences(dev_seqs, maxlen=self.max_len)
        self.X_test = pad_sequences(test_seqs, maxlen=self.max_len)

        self.token_num = len(self.word_tokenizer.word_index)

        return self.X_train, self.X_dev, self.X_test
    
    def process_tag_sequences(self, sentences, tags):
        self.__train_test_split(sentences, tags)
        self.tag_tokenizer = Tokenizer(filters='', oov_token='<unk>', lower=False)
        self.tag_tokenizer.fit_on_texts([" ".join(s) for s in self.train_tags])
        self.tag_tokenizer.word_index['<pad>'] = 0
        self.tag_tokenizer.index_word[0] = '<pad>' 

        self.index_tag = {i:w for w, i in self.tag_tokenizer.word_index.items()}
        self.index_tag_wo_padding = dict(self.index_tag)
        self.index_tag_wo_padding[self.tag_tokenizer.word_index['<pad>']] = '0'
        # Transform tags to integers
        train_tags_seqs = self.tag_tokenizer.texts_to_sequences([" ".join(s) for s in self.train_tags])
        dev_tags_seqs = self.tag_tokenizer.texts_to_sequences([" ".join(s) for s in self.dev_tags])
        test_tags_seqs = self.tag_tokenizer.texts_to_sequences([" ".join(s) for s in self.test_tags])
        # Pad all sequences to same length
        self.y_train = pad_sequences(train_tags_seqs, maxlen=self.max_len)
        self.y_dev = pad_sequences(dev_tags_seqs, maxlen=self.max_len)
        self.y_test = pad_sequences(test_tags_seqs, maxlen=self.max_len)

        self.tag_num = len(self.tag_tokenizer.word_index)

        return self.y_train, self.y_dev, self.y_test
    
    def __extract_character_sequences(self, data):
        X_char = []
        for sentence in data:
            sent_seq = []
            for word in sentence:
                word_seq = []
                if word == 0:
                    word_seq.append(self.char2idx.get("<pad>"))
                    sent_seq.append(word_seq)
                    continue
                else:
                    actual_word =  self.word_tokenizer.index_word.get(word)
                    for char in actual_word:
                        value = self.char2idx.get(char)
                        if (value is None):
                            word_seq.append(self.char2idx.get('<unk>'))
                        else:
                            word_seq.append(value)
                sent_seq.append(word_seq)
            sent_seq = pad_sequences(sent_seq, maxlen=self.char_max_len)
            X_char.append(sent_seq)
        return X_char

    def process_characters(self, char_max_len=10):
        self.char_max_len = char_max_len
        words = set([word.lower() for sentence in self.train_sentences for word in sentence])
        chars = set([char for word in words for char in word])

        self.char2idx = {c: i+2 for i, c in enumerate(chars)}
        self.char2idx["<unk>"] = 1
        self.char2idx["<pad>"] = 0
        
        self.X_char_train = np.array(self.__extract_character_sequences(self.X_train))
        self.X_char_dev = np.array(self.__extract_character_sequences(self.X_dev))
        self.X_char_test = np.array(self.__extract_character_sequences(self.X_test))

        return self.X_char_train, self.X_char_dev, self.X_char_test
        

        


