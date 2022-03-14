# Part 3: Mining text data.

# Return a pandas dataframe containing the data set.
# Specify a 'latin-1' encoding when reading the data.
# data_file will be populated with the string 'wholesale_customers.csv'.
from nltk.stem import PorterStemmer
import pandas as pd
from collections import Counter
import requests

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder


def read_csv_3(data_file):
    df = pd.read_csv(data_file, encoding='latin-1')
    return df


# Return a list with the possible sentiments that a tweet might have.
def get_sentiments(df):
    unique_sentiments = pd.unique(df["Sentiment"]).tolist()
    return unique_sentiments


# Return a string containing the second most popular sentiment among the tweets.
def second_most_popular_sentiment(df):
    popular_sentiment = df["Sentiment"].value_counts().keys()[1]
    return popular_sentiment


# Return the date (string as it appears in the data) with the greatest number of extremely positive tweets.
def date_most_popular_tweets(df):
    ordered_by_date = df.groupby("TweetAt")["Sentiment"].value_counts().to_dict()
    result = sorted([(key, ordered_by_date[key]) for key in ordered_by_date if key[1] == "Extremely Positive"],
                    key=lambda x: x[1])
    return result[-1][0][0]


# Modify the dataframe df by converting all tweets to lower case.
def lower_case(df):
    df["OriginalTweet"] = df["OriginalTweet"].str.lower()
    return df


# Modify the dataframe df by replacing each characters which is not alphabetic or whitespace with a whitespace.
def remove_non_alphabetic_chars(df):
    df["OriginalTweet"] = df["OriginalTweet"].str.replace('[^a-zA-Z\s]', " ", regex=True)
    return df


# Modify the dataframe df with tweets after removing characters which are not alphabetic or whitespaces.
def remove_multiple_consecutive_whitespaces(df):
    df["OriginalTweet"] = df["OriginalTweet"].str.replace('\s+', ' ', regex=True)
    return df


# Given a dataframe where each tweet is one string with words separated by single whitespaces,
# tokenize every tweet by converting it into a list of words (strings).
def tokenize(df):
    df["OriginalTweet"] = df["OriginalTweet"].str.split()
    return df


# Given dataframe tdf with the tweets tokenized, return the number of words in all tweets including repetitions.
def count_words_with_repetitions(tdf):
    list_tweets = [x for x in tdf["OriginalTweet"]]
    count = 0
    for tweet in list_tweets:
        count += len(tweet)
    return count


# Given dataframe tdf with the tweets tokenized, return the number of distinct words in all tweets.
def count_words_without_repetitions(tdf):
    list_tweets = []
    for x in tdf["OriginalTweet"]:
        list_tweets.extend(x)
    unique_words = set(list_tweets)
    return len(unique_words)


# Given dataframe tdf with the tweets tokenized, return a list with the
# k distinct words that are most frequent in the tweets.
def frequent_words(tdf, k):
    list_tweets = []
    for x in tdf["OriginalTweet"]:
        list_tweets.extend(x)
    count = Counter(list_tweets)
    frequent = []
    most_common = count.most_common(k)
    for item in most_common:
        frequent.append(item[0])
    return frequent


# Given dataframe tdf with the tweets tokenized, remove stop words and words with <=2 characters from each tweet.
# The function should download the list of stop words via:
# https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt
def remove_stop_words(tdf):
    r = requests.get("https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt")
    stop_words = r.content.decode("utf-8").split("\n")
    stop_words = [i for i in stop_words if len(i) > 2]
    new_col = []
    for lst in tdf["OriginalTweet"]:
        filtered = [i for i in lst if len(i) > 2]
        filtered = [i for i in filtered if i not in stop_words]
        new_col.append(filtered)
    tdf["OriginalTweet"] = new_col
    return tdf


# Given dataframe tdf with the tweets tokenized, reduce each word in every tweet to its stem.
def stemming(tdf):
    porter = PorterStemmer()
    new_col = []
    for idx, lst in enumerate(tdf["OriginalTweet"]):
        stemmed = [porter.stem(x) for x in lst]
        new_col.append(stemmed)
    tdf["OriginalTweet"] = new_col
    return tdf


# Given a pandas dataframe df with the original coronavirus_tweets.csv data set,
# build a Multinomial Naive Bayes classifier. 
# Return predicted sentiments (e.g. 'Neutral', 'Positive') for the training set
# as a 1d array (numpy.ndarray). 
def mnb_predict(df):
    mnb = MultinomialNB()
    X = df["OriginalTweet"].to_list()
    X_cv = CountVectorizer(analyzer="word", ngram_range=(0, 4))
    X = X_cv.fit_transform(X)
    y = df["Sentiment"].to_list()
    y_cv = LabelEncoder()
    y = y_cv.fit_transform(y)
    mnb.fit(X, y)
    labels = y_cv.classes_
    y_hat = mnb.predict(X)
    y_hat = [labels[i] for i in y_hat]
    return y_hat


# Given a 1d array (numpy.ndarray) y_pred with predicted labels (e.g. 'Neutral', 'Positive')
# by a classifier and another 1d array y_true with the true labels, 
# return the classification accuracy rounded in the 3rd decimal digit.
def mnb_accuracy(y_pred, y_true):
    count = 0
    for i in range(len(y_true)):
        if y_pred[i] == y_true[i]:
            count += 1
    return round(count / len(y_true), 3)
