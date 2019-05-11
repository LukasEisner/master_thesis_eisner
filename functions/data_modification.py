# Imports

import numpy as np
import pandas as pd
from functools import partial
from collections import Counter


# Functions

def buy_sell_hold(*args: float, threshold: float):
    """This function decides on buy (1), hold (0), sell (-1) based on a threshold

    Args:
        *args (float):
        threshold (float): the percentage change which has to be crossed for a (1) or (-1) decision

    Return:
        boolean: buy (1), hold (0), sell (-1)

    """

    cols = [c for c in args]
    for col in cols:
        if col > threshold:
            return 1
        if col < -threshold:
            return -1
    return 0


def extract_featuresets(coin: str, tickers: list, df_merged, hm_days: int, my_threshold: float):
    """This function calculates the decision for the features and adds it to the data frame

    Args:
        coin (str): Name of the coin to be looked into
        tickers (list): List of all the coins whose financial data is taken into consideration
        df_merged (pandas.core.frame.DataFrame): merged data set with all input data
        hm_days (int): For how many days into the future should the percentage change be calculated
        my_threshold (float): the percentage change which has to be crossed for a (1) or (-1) decision

    Return:
        X (numpy.ndarray): The whole input for the machine learning model (features)
        y (numpy.ndarray): The output for the machine learning model (labels)
        df_merged (pandas.core.frame.DataFrame): A data frame which contains both features and labels

    """

    # Map buy/sell/hold decision for whole dataset
    df_merged['{}_target'.format(coin)] = list(
        map(partial(buy_sell_hold, threshold=my_threshold),
            *[df_merged['{}_{}d'.format(coin, i)] for i in range(1, hm_days + 1)]))

    # check and print data spread
    vals = df_merged['{}_target'.format(coin)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:', Counter(str_vals))

    # avoid infinite numbers
    df_merged.fillna(0, inplace=True)
    df_merged = df_merged.replace([np.inf, -np.inf], np.nan)
    df_merged.dropna(inplace=True)

    # normalize the price values and make it percent change
    df_vals = df_merged[[coin for coin in tickers]].pct_change()

    # check again for infinite numbers
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    # extract features and label for machine learning
    X = df_vals.values
    y = df_merged['{}_target'.format(coin)].values

    return X, y, df_merged


def aggregate_sentiment(my_tweets, frequence: str = '1H'):
    """This function hourly aggregates the tweets. Important note: The 'time' column indicates the end of the
        considered timedelta

    Args:
        my_tweets (pandas.core.frame.DataFrame): The labelled tweets
        frequence (str): The frequence with which to aggregate. Default: '1H'

    Return:
        my_aggregated_tweets (pandas.core.frame.DataFrame): The aggregated tweets

    """
    my_tweets.columns = map(str.lower, my_tweets.columns)
    my_tweets['time'] = pd.to_datetime(my_tweets['exact date'], format='%Y-%m-%d %H:%M:%S')
    my_tweets = my_tweets.set_index(my_tweets['time'])
    my_tweets['amount_of_tweets'] = 1
    my_tweets['weighted_sentiment'] = (my_tweets['retweets']+1)*my_tweets['sentiment']
    my_tweets['weighted_sentiwordnet'] = (my_tweets['retweets'] + 1) * my_tweets['sentiwordnet']

    # Aggregate the tweets to get hourly information and fillna with 0
    my_aggregated_tweets = my_tweets.resample(frequence).agg({'retweets': np.sum,
                                                            'favorites': np.sum,
                                                            'amount_of_tweets': np.sum,
                                                            'sentiment': np.mean,
                                                            'sentiwordnet': np.mean,
                                                            'weighted_sentiment': np.sum,
                                                            'weighted_sentiwordnet': np.sum}).fillna(0)

    my_aggregated_tweets['weighted_sentiment'] = my_aggregated_tweets['weighted_sentiment'] /\
                                                 (my_aggregated_tweets['amount_of_tweets'] +
                                                 my_aggregated_tweets['retweets'])
    my_aggregated_tweets['weighted_sentiwordnet'] = my_aggregated_tweets['weighted_sentiwordnet'] / \
                                                 (my_aggregated_tweets['amount_of_tweets'] +
                                                  my_aggregated_tweets['retweets'])
    my_aggregated_tweets['weighted_sentiment'] = my_aggregated_tweets['weighted_sentiment'].fillna(0)
    my_aggregated_tweets['weighted_sentiwordnet'] = my_aggregated_tweets['weighted_sentiwordnet'].fillna(0)
    # Shift all columns to reflect that 'time' indicates the end of the timedelta
    my_aggregated_tweets.index = my_aggregated_tweets.index.shift(1)


    return my_aggregated_tweets


def aggregate_financials(df_financials, frequence: str = '1H'):
    """ This function aggregates the financial data to fit the frequency of the sentiment data

    Args:
        df_financials (pandas.core.frame.DataFrame): The financial values
        frequence (str): The frequence with which sentiment was aggregated. Default: '1H'

    Return:
        df_financials_aggr (pandas.core.frame.DataFrame): The aggregated financial data

    """



    return df_financials_aggr