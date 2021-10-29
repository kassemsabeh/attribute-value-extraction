import pandas as pd
from nltk.tokenize import word_tokenize

class Tagger():

    def __init__(self) -> None:
        pass

    def __fill_dict(self, attribute, name):
        for i, word in enumerate(attribute):
            if i == 0:
                self.my_dict[word] = 'B-' + name
            else:
                self.my_dict[word] = 'I-' + name


    def __create_dictionary(self, index, row):
        self.my_dict = {}
        try:
            brand = row['Brand Name'].split()
            self.__fill_dict(brand, 'Brand')
        except:
            pass
        try:
            material = row['Material'].split()
            self.__fill_dict(material, 'Material')
        except:
            pass
        try:
            color = row['Color'].split()
            self.__fill_dict(color, 'Color')
        except:
            pass
        try:
            category = row['Category'].split()
            self.__fill_dict(category, 'Category')
        except:
            pass



    def bio_tag(self, df: pd.DataFrame):
        sentences = []
        tags = []

        for index, row in df.iterrows():
            current_sentence = word_tokenize(row['Title'])
            current_tag = []
            self.__create_dictionary(index, row)
            for word in current_sentence:
                try:
                    tag = self.my_dict[word]
                    if tag.startswith('B-'):
                        tag_started = True
                        current_tag.append(tag)
                    elif tag.startswith('I-') and tag_started:
                        current_tag.append(tag)
                    else:
                        current_tag.append('O')
                        tag_started = False
                except:
                    current_tag.append('O')
                    tag_started = False
            
            sentences.append(current_sentence)
            tags.append(current_tag)

        return sentences, tags


