# Imports

import pickle

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# Functions

def create_keras_tokenizer(df_tweets, max_features: int = 5000, max_len: int = 40):
    """This function creates a keras tokenizer for a certain max_features and max_len and fits it on df_tweets

    Args:
        df_tweets (pandas.core.frame.DataFrame):  Data frame with labelled tweets
        max_features (int): Maximum features for the tokenizer. Defaults to 5000
        max_len (int): Maximum length of tokens. Defaults to 40

    Return:
        tokenizer_maxlen (int): Maximum length of tokens for the chosen tokenizer
        tokenizer (keras_preprocessing.text.Tokenizer): The created tokenizer

    """

    # Two hyperparameters for the tokenizer
    tokenizer_max_features = max_features
    tokenizer_maxlen = max_len

    # Create the tokenizer
    tokenizer = Tokenizer(num_words=tokenizer_max_features, split=' ')
    tokenizer.fit_on_texts(df_tweets['text'].values)

    return tokenizer_maxlen, tokenizer


def apply_keras_tokenizer(df_tweets, tokenizer, tokenizer_maxlen: int = 40):
    """This function applies a keras tokenizer to the tweets in the df_tweets data frame.

    Args:
        df_tweets (pandas.core.frame.DataFrame):  Data frame with labelled tweets
        tokenizer (keras_preprocessing.text.Tokenizer): The fitted tokenizer
        tokenizer_maxlen (int): Maximum length of tokens. Defaults to 40

    Return:
        tokenized_tweets (numpy.ndarray): Tokenized tweets

    """

    # Use the tokenizer
    tokenized_tweets = tokenizer.texts_to_sequences(df_tweets['text'].values)
    tokenized_tweets = pad_sequences(tokenized_tweets, maxlen=tokenizer_maxlen)

    return tokenized_tweets

def save_model_and_tokenizer (filepath_model: str, filepath_tokenizer: str, my_model, my_tokenizer):
    """This function saves a model and a tokenizer

    Args:
        filepath_model (str): The location and name of the model to be stored
        filepath_tokenizer (str): The location and name of the tokenizer to be stored
        my_model: The model to be stored
        my_tokenizer: The tokenizer to be stored

    Return:
        model file: Write a .sav file where it stores the model
        tokenizer file: Writes a .sav file where it stores the tokenizer
    """

    # Save the model
    pickle.dump(my_model, open(filepath_model, "wb"))

    # Save the tokenizer
    with open(filepath_tokenizer, "wb") as handle:
        pickle.dump(my_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Model and Tokenizer saved successfully!")
    return


def load_model_and_tokenizer (filepath_model: str, filepath_tokenizer: str):
    """This function loads a model and a tokenizer

    Args:
        filepath_model (str): The location and name of the model to be loaded
        filepath_tokenizer (str): The location and name of the tokenizer to be loaded

    Return:
        my_model: The loaded model
        my_tokenizer: The loaded tokenizer
    """

    # load a model
    my_model = pickle.load(open(filepath_model, "rb"))

    # Load a tokenizer
    with open(filepath_tokenizer, "rb") as handle:
        my_tokenizer = pickle.load(handle)

    return my_model, my_tokenizer
