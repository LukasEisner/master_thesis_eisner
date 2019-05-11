# Imports

import pandas as pd

# Functions

def aggregate_sentiment(df_sentiment):
    """This function aggregates the sentiment values to daily

    Args:
        df_sentiment (pandas.core.frame.DataFrame): Data frame with sentiment values

    Return:
        aggr_df_sentiment (pandas.core.frame.DataFrame): Data frame with aggregated sentiment values

    """

    aggr_df_sentiment = pd.DataFrame()
    aggr_df_sentiment['amount_of_tweets'] = df_sentiment['amount_of_tweets'].resample('D').sum()
    aggr_df_sentiment['average_sentiment'] = df_sentiment['sentiment'].resample('D').mean()

    return aggr_df_sentiment


def aggregate_financials(df_financials):
    """This function aggregates the financial values to daily

    Args:
        df_financials (pandas.core.frame.DataFrame): Data frame with financial values

    Return:
        aggr_df_financials (pandas.core.frame.DataFrame): Data frame with aggregated financial values

    """

    aggr_df_financials = df_financials.resample('D').first()

    return aggr_df_financials


def calculate_financial_change(coin: str, aggr_df_financials, hm_days: int):
    """This function calculates the percentage change of financial value for (hm_days) and
       adds it to the aggr_df_financials

    Args:
        coin (str): Name of the coin to be looked into
        aggr_df_financials (pandas.core.frame.DataFrame): Data frame with aggregated financial values
        hm_days (int): For how many days into the future should the percentage change be calculated

    Return:
        aggr_df_financials (pandas.core.frame.DataFrame): Data frame with aggregated financial values and
                                                          percentage financial change

    """

    # Calculate the percentage change for (hm_days)
    for i in range(1, hm_days + 1):
        aggr_df_financials['{}_{}d'.format(coin, i)] = \
            (aggr_df_financials[coin].shift(-i) - aggr_df_financials[coin]) / aggr_df_financials[coin]

    # Fill empty cells with 0
    aggr_df_financials.fillna(0, inplace=True)

    # Remove last (hm_days) rows because the future financial change will be 0
    aggr_df_financials = aggr_df_financials[:-hm_days]
    return aggr_df_financials


def merge_sentiment_financials(aggr_df_sentiment, aggr_df_financials):
    """This function combines both aggr_df data frames into one

    Args:
        aggr_df_sentiment (pandas.core.frame.DataFrame): Aggregated sentiment data frame
        aggr_df_financials (pandas.core.frame.DataFrame): Aggregated financials data frame

    Return:
        df_merged (pandas.core.frame.DataFrame): Combined data frames with all aggregated data

    """

    # Merge the two dataframes
    df_merged = pd.concat([aggr_df_financials, aggr_df_sentiment], axis=1)

    return df_merged


def preprocess_tweets(df_tweets):
    """This function preprocesses the tweets (lower case and remove certain characters)

    Args:
        df_tweets (pandas.core.frame.DataFrame): Data frame with labelled tweets

    Return:
        df_tweets (pandas.core.frame.DataFrame): Preprocessed data frame with labelled tweets

    """

    # Replace upper case letters with their lower case letter counterparts
    df_tweets['text'] = df_tweets['Text'].apply(lambda x: x.lower())

    return df_tweets


def concat_btc_files(coin: str, folderpath: str = 'data/'):
    """This function concatenates tweet.csv files from both first and second scrape into one giant data frame

    Args:
        coin (str): Which coin are we talking about
        folderpath (str): What subfolder are the COIN_v1 and COIN_v2 folders in

    Return:
        df_tweets (pandas.core.frame.DataFrame): Concatenated data frame with all tweets

    """
    filenames_v1 = glob.glob(folderpath + 'coins/' + coin + '_v1/*.csv')
    filenames_v2 = glob.glob(folderpath + 'coins/' + coin + '_v2/*.csv')
    filenames = filenames_v1 + filenames_v2

    df_tweets = pd.concat([pd.read_csv(f) for f in filenames])

    return df_tweets

def concat_tweet_files(coin: str, folderpath: str = 'data/'):
    """This function concatenates tweet.csv files into one giant data frame

    Args:
        coin (str): Which coin are we talking about
        folderpath (str): What subfolder are the COIN folders in

    Return:
        df_tweets (pandas.core.frame.DataFrame): Concatenated data frame with all tweets

    """
    filenames = glob.glob(folderpath + 'coins/' + coin + '/*.csv')

    df_tweets = pd.concat([pd.read_csv(f) for f in filenames])

    return df_tweets



def filter_tweet_file(df_tweets, semantic_filter_filepath: str):
    """This function filters the tweets according to the semantic filter file

    Args:
        df_tweets (pandas.core.frame.DataFrame): Data frame with all tweets
        semantic_filter_filepath (str): Path to the .csv file with the filterwords

    Return:
        df_tweets (pandas.core.frame.DataFrame): Filtered data frame with all tweets

    """
    len_list = list()
    # Remove duplicates
    print('Shape before filtering: ' + str(len(df_tweets)))
    len_list.append(len(df_tweets))
    df_tweets = df_tweets.drop_duplicates()
    df_tweets = df_tweets.drop_duplicates(subset='Text')
    print('Shape after dropping duplicates: ' + str(len(df_tweets)))
    len_list.append(len(df_tweets))

    # Load the words to be filtered for and make a list from it
    semantic_search_words = pd.read_csv(semantic_filter_filepath, usecols=[0, 1])
    pos_exp = list(semantic_search_words['pos_words'].values.flatten())
    neg_exp = list(semantic_search_words['neg_words'].values.flatten())

    # make a string with separator | from the list
    pos_exp_string = "|".join(pos_exp)
    neg_exp_string = "|".join(neg_exp)

    # make new column for data frame which contains boolean if the row is relevant
    df_tweets = df_tweets[df_tweets['Text'].str.contains(pos_exp_string, case=False)]
    print('Shape after positive word filter: ' + str(len(df_tweets)))
    len_list.append(len(df_tweets))
    df_tweets = df_tweets[~df_tweets['Text'].str.contains(neg_exp_string, case=False)]
    print('Shape after negative word filter: ' + str(len(df_tweets)))
    len_list.append(len(df_tweets))
    return df_tweets, len_list





