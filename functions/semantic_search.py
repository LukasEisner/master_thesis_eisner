# Imports
import os
import pandas as pd

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors

# Functions

def convert_glove(glove_filepath: str, word2vec_filepath: str):
    """This function converts GloVe vectors in text format into the word2vec text format

        Args:
            glove_filepath (str): path to the glove file
            word2vec_filepath (str): path where the word2vec file will be stored

        Return:
            glove file: Write a .txt file with the converted vectors

        """
    #check if the converted file already exists
    if not os.path.isfile(word2vec_filepath):

        # use glove2word2vec to convert GloVe vectors in text format into the word2vec text format
        glove2word2vec(glove_input_file=glove_filepath, word2vec_output_file=word2vec_filepath)
    else:
        print('The file has already been converted!')

    return


def search_similar_words(word2vec_filepath: str, my_word: str, amount_of_words: int = 100):
    """This function looks for semantically similar words to the input word

    Args:
        word2vec_filepath (str): path where the word2vec file will be stored
        my_word (str): the word for which to look for similarities
        amount_of_words (int): how many of the most similar words should be returned

    Return:
        wordlist: List of similar words

        """

    glove_model = KeyedVectors.load_word2vec_format(word2vec_filepath, binary=False)

    wordlist = glove_model.similar_by_word(my_word, topn=amount_of_words)

    return wordlist
