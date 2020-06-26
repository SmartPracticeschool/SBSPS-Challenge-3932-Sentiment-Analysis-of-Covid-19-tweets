'''-------------------Importing Libraries-------------------'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk

# nltk.download('vedar_lexicon')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

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


'''-------------Data Read-------------'''

dataset = pd.read_csv('DataSet.csv')
dataset.head()
#tweets.shape



'''Extraction of English Tweets'''











'''---------------------Data-Preprocessing----------------------'''

'''-------------Removing Emojis-------------'''

def removeEmoji(result):
    return result.encode('ascii', 'ignore').decode('ascii')

tweets['text'] = [removeEmoji(i) for i in tweets['text']]
#tweets.head()


'''-------------Remove URLs-------------'''

def removeURL(str):
    temp = ''
    clean_1 = re.match('(.*?)http.*?\s?(.*?)', str)
    clean_2 = re.match('(.*?)https.*?\s?(.*?)', str)

    if clean_1:
        temp = temp + clean_1.group(1)
        temp = temp + clean_1.group(2)
    elif clean_2:
        temp = temp + clean_2.group(1)
        temp = temp + clean_2.group(2)
    else:
        temp = str
    return temp

tweets['text'] = tweets['text'].apply(lambda tweet: removeURL(tweet))
#tweets.head()


'''-------------Lowercase---------------'''

tweets['text'] = [i.lower() for i in tweets['text']]
#tweets.head()


'''-------------Removing all Punctuations---------------'''

tweets['text'] = [re.sub('[^a-zA-Z]', ' ', i) for i in tweets['text']]
#tweets.head()



'''-------------Remove StopWords--------------'''

stopWords = set(stopwords.words("english"))
tweets['text'] = tweets['text'].apply(lambda tweet: ' '.join([word for word in tweet.split() if word not in stopWords]))

focused_words = ['coronavirus', 'covid', 'quarantine', 'coronavirusoutbreak', 'virus', 'corona', 'lockdown', 'economy']
#tweets.head()


'''-------------Stemming------------'''

ps = PorterStemmer()

def stemWords(word):
    if word in focused_words:
        return word
    else:
        return ps.stem(word)

tweets['text'] = tweets['text'].apply(lambda tweet: ' '.join([stemWords(word) for word in tweet.split()]))
#tweets.head()


'''--------------Lemmatization-------------'''

wnl = WordNetLemmatizer()

def lemmatizeWords(word):
    if word in focused_words:
        return word
    else:
        return wnl.lemmatize(word)

tweets['text'] = tweets['text'].apply(lambda tweet: ' '.join([lemmatizeWords(word) for word in tweet.split()]))
#tweets.head()




'''Labeling the tweets'''





'''Vectorization'''






