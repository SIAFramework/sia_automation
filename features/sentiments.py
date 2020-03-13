"""
Created on Fri Nov  8 11:45:25 2019
@author: AkOjha
"""

import pandas as pd
import numpy as np
import datetime as dt
import stanfordnlp
import re
import ftfy
from ftfy import fix_text
from nltk.stem import LancasterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag.stanford import StanfordPOSTagger
from nltk.corpus import stopwords
from nltk import WordPunctTokenizer
from textblob import TextBlob


def extract_sentiment_stanford(x, sentiment_nlp):
    sentiment = []
    res = sentiment_nlp.annotate(x,
                   properties={
                       'annotators': 'sentiment',
                       'outputFormat': 'json',
                       'timeout': 50000,
                   })
    for s in res["sentences"]:
        sentiment = s["sentiment"]
    return sentiment



def extract_sentiment(text, sentiment_nlp):
    polarity = TextBlob(text).sentiment.polarity   
    if polarity>=0.8:
        return 'Verypositive'
    elif (polarity < 0.8) and (polarity > 0):
        return 'Positive'
    elif polarity == 0:
        return extract_sentiment_stanford(text, sentiment_nlp)
    elif (polarity < 0) and (polarity > -0.8):
        return 'Negative'
    elif polarity <= -0.8:
        return 'Verynegative'