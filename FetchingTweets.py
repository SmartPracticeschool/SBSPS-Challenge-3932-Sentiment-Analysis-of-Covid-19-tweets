'''-------------------Importing Libraries-------------------'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk

nltk.download('vedar_lexicon')
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



# '''-----------------Twitter API credentials keys-----------------'''
# 
# Put your own credential keys
# consumerKey = ''
# consumerKeySecret = ''
# accessToken = ''
# accessTokenSecret = ''



# '''----------------Authenticating---------------------'''

# # Create the authentication object
# auth = tweepy.OAuthHandler(consumerKey,consumerKeySecret)

# # Create the authentication access
# auth.set_access_token(accessToken,accessTokenSecret)

# # Create the API object
# api = tweepy.API(auth,wait_on_rate_limit=True)



# '''------------Fetching the data-------------'''

# # Open/create a file to append data to
# csvFile = open('DataSet.csv', 'a',encoding="utf-8",newline='')

# #Use csv writer
# csvWriter = csv.writer(csvFile)

# # focus_word = ['coronavirus', 'covid', 'quarantine', 'coronavirusoutbreak', 'virus', 'corona', 'lockdown', 'economy']

# backoff_counter = 1
# while True:
#     if backoff_counter == 3:
#         break
#     try:
#         for tweet in tweepy.Cursor(api.search,count = 10000,q = ('COVID-19' or 'coronavirus' or 'covid' or 'coronavirusoutbreak' or 'corona' or 'lockdown' or 'economy' or 'extension'), tweet_mode='extended',lang = "en",location='india').items():
#             csvWriter.writerow([tweet.full_text,tweet.user.screen_name,tweet.user.location])    # Write a row to the CSV file. I use encode UTF-8
#         csvFile.close()
#         break
#     except tweepy.TweepError:
#         time.sleep(120)
#         backoff_counter += 1
#         continue



# '''------------Creating a CSV file------------'''
# 
# file_name='text.txt'
# data = pd.read_csv(file_name, sep = ',',names = ['text','user_id','location'])
# # data.shape
# # data.head(10)
# # data.info()
# data['lang'] = 'en'
# data.to_csv('DataSet.csv',index=False)
# data


#'''-------------Check for the Data-------------'''

#dataset = pd.read_csv('DataSet.csv')
#dataset.head()
