# Imports

import pandas as pd


# Functions

def load_financial_data(filepath: str, start_date: str, end_date: str, tickers: str):
    """This function loads the hourly financial data for the top 21 crypto coins.

    Args:
        filepath (str): Dir to the csv to be read.
        start_date (str): Date from which the data should be read.
        end_date (str): Date until which the data should be read.
        tickers (str): List of coins to be read.

    Return:
        df_financials (pandas.core.frame.DataFrame): Data frame with all coin data for top 21 coins

    """
    # Load financial data from /sources
    df_financials = pd.read_csv(filepath, index_col=0)

    # Set Index (time) to datetime in correct format
    df_financials.index = pd.to_datetime(df_financials.index, format='%d/%m/%Y %H:%M')

    # Adjust Timeframe
    df_financials = df_financials[start_date: end_date]

    # Choose coins
    df_financials = df_financials[tickers]

    # Fill possible empty values with 0
    df_financials.fillna(0, inplace=True)

    return df_financials


def load_sentiment_data(filepath: str, start_date: str, end_date: str):
    """This function loads the hourly sentiment data

    Args:
        filepath: (str): Dir to the csv to be read.
        start_date (str): Date from which the data should be read.
        end_date (str): Date until which the data should be read.

    Return:
        df_sentiment (pandas.core.frame.DataFrame): Data frame with sentiment values

    """

    # Load sentiment data
    df_sentiment = pd.read_csv(filepath, index_col=0)

    # Adjust index and use correct format
    df_sentiment.index = pd.to_datetime(df_sentiment.index, format='%Y-%m-%d %H:%M:%S')

    # Only keep rows within start_date to end_date
    df_sentiment = df_sentiment[start_date: end_date]

    return df_sentiment

def load_labelled_data (filepath: str):
    """This function loads the manually labelled tweets

    Args:
        filepath: (str): Dir to the csv to be read.

    Return:
        df_tweets (pandas.core.frame.DataFrame): Data frame with labelled tweets

    """

    df_tweets = pd.read_csv(filepath, usecols=['text', 'label'], encoding='latin-1')

    return df_tweets


def load_tweets (filepath: str):
    """This function loads the scraped tweets

        Args:
            filepath: (str): Dir to the csv to be read.

        Return:
            df_tweets (pandas.core.frame.DataFrame): Data frame with scraped tweets

        """
    # Load tweets and keep only relevant columns
    my_tweets = pd.read_csv(filepath, usecols=["Text", "Exact Date", "Retweets", "Favorites"], encoding="latin-1")
    my_tweets.columns = ["text", "exact_date", "retweets", "favorites"]

    return my_tweets

