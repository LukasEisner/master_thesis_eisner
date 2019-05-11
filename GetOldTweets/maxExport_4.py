import sys, getopt, datetime, codecs
import got3
import pandas as pd
from datetime import datetime
from datetime import timedelta
from time import sleep


def getTweets(Name="", Term="", Start="", End="", maxDayTweets=5000):
    tweetCriteria = got3.manager.TweetCriteria()

    twList = []
    delta = datetime.strptime(End, "%Y-%m-%d")- datetime.strptime(Start, "%Y-%m-%d")
    day_count = delta.days

    for index in range(0,day_count, 1):

        dateStart = (datetime.strptime(Start, "%Y-%m-%d") + timedelta(days=index)).strftime("%Y-%m-%d")
        dateEnd = (datetime.strptime(Start, "%Y-%m-%d") + timedelta(days=index + 1)).strftime("%Y-%m-%d")

        tweetCriteria = got3.manager.TweetCriteria().setQuerySearch(Term).setSince(dateStart).setUntil(
            dateEnd).setMaxTweets(maxDayTweets)

        tweets = got3.manager.TweetManager.getTweets(tweetCriteria)

        for tweet in tweets:
            twList.append([Name, tweet.text, tweet.id, tweet.date, tweet.retweets, tweet.favorites, tweet.mentions, tweet.hashtags])

        twDf = pd.DataFrame(columns=["Company", "Text", "ID", "Exact Date", "Retweets", "Favorites", "Mentions", "Hashtags"], data=twList)

    return twDf


# main loow
if __name__ == '__main__':

    df = pd.read_csv("to_scrape_part_4.csv", sep=",")
    df = df[["Company", "Start", "End", "Lookup Term"]]
    print(df.head())
    coldict = dict(zip(["Company", "Start", "End", "Lookup Term"], [0, 1, 2, 3]))


    for index, row in df.iterrows():
        starttime = datetime.now()
        outputFileName = row[coldict['Company']].lower() + "_tweets.csv"
        folderName = outputFileName[:-14]
        tweetsDf = getTweets(row[coldict['Company']], row[coldict["Lookup Term"]], row[coldict["Start"]], row[coldict["End"]],
                             100000)

        print(outputFileName + " now contains " + str(len(tweetsDf)) + " Tweets scraped in " + str(datetime.now() - starttime))
#        print(tweetsDf.head())

        tweetsDf.to_csv("Done_Scrapes/" + folderName + "/" + outputFileName, header=True, index=False)

input("Press enter to exit")
    # if index%10 == 0:
    #	sleep(300)