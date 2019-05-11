# master_thesis_eisner

## This git provides the code used in the master thesis of Lukas Eisner at ETH Zurich, May 2019

**Order**:  

**GetOldTweets** - Scrape twitter for tweets. Adapted from https://github.com/Jefferson-Henrique/GetOldTweets-python  
**filter_tweets.ipynb** - Jupyter Notebook containing the filtering process  
**benchmark_sentiment_analysis.ipynb** - Check which sentiment analysis model fits best for the task at hand  
**train_sentiment_analysis.ipynb** - Train the actual model with the best choice from the benchmark (Random Forest)  
**validate_sentiment_analysis.ipynb** - Test the accuracy of the SentiWordNet Model (by thresholding)  
**execute_sentiment_analysis.ipynb** - Apply the Random Forest and the SentiWordNet Model to all tweets
**descriptive_analysis.ipynb** - Basic statistical analysis and plots  
**crosscorr_analysis.ipynb** - Lagged Cross-Correlation Analysis  
**thermal_optimal_path.ipynb** - Application and plot of the TOP method. Adapted from https://github.com/amwatt/thermal_optimal_path
