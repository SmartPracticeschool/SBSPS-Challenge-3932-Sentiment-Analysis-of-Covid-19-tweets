'''-------------------Importing Libraries-------------------'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk

# nltk.download('vedar_lexicon')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')

import tweepy
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import sns as sns
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.tokenize import sent_tokenize
from nltk.sentiment.util import *
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score

import time
from random import shuffle
from nltk import NaiveBayesClassifier
from nltk import classify
# from sklearn.externals import joblib



'''Data Read'''






'''----------Extraction of English Tweets----------'''

tweets = dataset[['user_id','text']][dataset['lang'] == 'en'].reset_index()
tweets.drop(["index"], axis=1)
tweet_copy = tweets.copy()  # getting a copy of the original tweets
# tweets.shape
# tweets.head()
# tweets.info()





'''Data-Preprocessing'''
















'''--------------Labeling the tweets---------------'''

def detect_polarity(text):
    return TextBlob(text).sentiment.polarity

tweets['polarity'] = tweets.text.apply(detect_polarity)
tweets.head()
# tweets.describe()

def detect_subjectivity(text):
    return TextBlob(text).sentiment.subjectivity

tweets['subjectivity'] = tweets.text.apply(detect_subjectivity)
tweets.head()
# tweets.describe()

def detect_sentiment(text):
    if TextBlob(text).sentiment.polarity > 0:
        return 1
    elif TextBlob(text).sentiment.polarity < 0:
        return -1
    else:
        return 0

tweets['sentiment'] = tweets.text.apply(detect_sentiment)
tweets.head()


'''--------------Vectorization--------------'''

x = tweets['text']
y = tweets['sentiment']

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_word_vectorizer = TfidfVectorizer(stop_words='english',max_features=1000,decode_error='ignore',use_idf=True)
x = tfidf_word_vectorizer.fit_transform(x)


