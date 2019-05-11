# Imports

# Functions


def apply_model(my_tweets, my_tokenized_tweets, my_model):
    """This function combines both aggr_df data frames into one

    Args:
        my_tweets (pandas.core.frame.DataFrame): The tweets to be labelled
        my_tokenized_tweets (pandas.core.frame.DataFrame): The tokenized form of the tweets to be labelled
        my_model: The trained model which labels the tweets

    Return:
        my_tweets (pandas.core.frame.DataFrame): The labelled tweets

    """
    my_tweets['sentiment'] = my_model.predict(my_tokenized_tweets)

    return my_tweets
