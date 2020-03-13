"""
Created on Fri Nov  8 12:15:19 2019
@author: AkOjha
"""

import pandas as pd
import numpy as np
import datetime as dt
import stanfordnlp
from nltk.tokenize import sent_tokenize
import re
import ftfy
from ftfy import fix_text
from nltk.stem import LancasterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag.stanford import StanfordPOSTagger
from nltk.corpus import stopwords
from nltk import WordPunctTokenizer
import spacy
import itertools
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob, Word
import os
from tqdm import tqdm
import time
from tqdm import tqdm_gui
import copy
import string 


def stemSentence(sentence, porter):
    token_words=word_tokenize(sentence)
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)


def lemmatize_with_postag(sentence):
    sent = TextBlob(sentence)
    tag_dict = {"J": 'a', 
                "N": 'n', 
                "V": 'v', 
                "R": 'r'}
    words_and_tags = [(w, tag_dict.get(pos[0], 'n')) for w, pos in sent.tags]    
    lemmatized_list = [wd.lemmatize(tag) for wd, tag in words_and_tags]
    return " ".join(lemmatized_list)


def ner(x, spacy_nlp):
    doc = spacy_nlp(x)
    entities = [(x.text, x.label_) for x in doc.ents]
    names = [i[0] for i in entities if (i[1]=='PERSON' or i[1]=='ORG' or i[1]=='GPE')]
    names1 = [char.split() for char in names]
    names2 = list(itertools.chain.from_iterable(names1))
    review_without_person_name = ' '.join([word for word in x.split() if word not in names2])
    return review_without_person_name


def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text


def create_theme_excetions():
    general_theme_exceptions1 = ['am','rt' ,'th', 'to']
    general_theme_exceptions2 = ['can','cans','homework','let','others', 'ok','fine','good','great']
    general_theme_exceptions3 = ['bit', 'side', 'usual', 'it', 'part', 'way', 'favorite', 'love', 'best', 'walmart', 'local', 'time', 'lot']
    general_theme_exceptions4 = ['end', 'chance', 'item', 'items', 'glad', 'tuesday', 'monday', 'awesome', 'thing',
                                 'one', 'It', 'way', 'bad', 'side', 'mg', 'oz', 'ammount', 'thanks', 'today', 'better', 'aed', 'name', 'sad']
    general_theme_exceptions5 = ['first','right', 'next', 'same', 'able','cc','motherfuckers', 'fuck', 'cup', 'shit']
    en_stopword = stopwords.words('english')
    theme_exception = general_theme_exceptions1 + general_theme_exceptions2 + general_theme_exceptions3 + general_theme_exceptions4 + general_theme_exceptions5 + en_stopword
    return theme_exception


def remove_stop_words(x):
    custom_stop_words = ['aaaa', 'aaa', 'ive', 'w', 'httpbit', 'u', 'yall', 'id', 'th', 'bc', 'httpst', 'oz', 'n', 's',
                         'ta', 'm', 'st', 'ur', 'TRUE', 'ya', 'hi', 'x', 'k', 'b', 'nd', 'etc', 'iv', 'p', 'c', 'lil',
                         'httpdlvr', 'yo', 'rn', 'smh', 'vs']
    en_stopword = stopwords.words('english')
    custom_stop_words = custom_stop_words + en_stopword
    querywords = x.split()
    resultwords  = [word for word in querywords if word.lower() not in custom_stop_words]
    result = ' '.join(resultwords)
    return result
 

def extract_theme(x, nlp):
    doc = nlp(x)
    adj = []
    pos = [[word.text,word.upos] for sent in doc.sentences for word in sent.words]
    theme_exceptions = create_theme_excetions() 
    noun = [d[0] for d in pos if d[1] in ['NOUN'] and d[0].lower() not in theme_exceptions]    
    if len(noun)==0:
        adj = [d[0] for d in pos if d[1] in ['ADJ'] and d[0].lower() not in theme_exceptions]
    else:
        adj = [] 
    noun_adj = list(np.unique(noun+adj))
    return noun_adj


def rm_longwords(input_string):
    return ' '.join(w for w in input_string.split() if len(w)<20)
    

def tag_themes(all_Reviews_df, spacy_nlp, nlp):
    all_Reviews_df['sentence'] = all_Reviews_df['sentence'].apply(lambda x: re.sub(r'\.+', ".", x))
    all_Reviews_df['sentence'] = all_Reviews_df['sentence'].apply(lambda x: str(x).strip())
    all_Reviews_df['sentence'] = all_Reviews_df['sentence'].apply(lambda x:ner(x, spacy_nlp))
    all_Reviews_df['sentence'] = all_Reviews_df['sentence'].apply(lambda x: fix_text(x))
    all_Reviews_df['sentence'] = all_Reviews_df['sentence'].apply(lambda x: ' '.join(word for word in x.split(' ') if not word.startswith('RT')))   
    all_Reviews_df['sentence'] = all_Reviews_df['sentence'].apply(lambda x: x.lower())    
    all_Reviews_df['sentence']  = all_Reviews_df['sentence'].apply(remove_punctuations) 
    all_Reviews_df['sentence']  = all_Reviews_df['sentence'].apply(lambda x: remove_stop_words(x)) 
    all_Reviews_df['sentence'] = all_Reviews_df['sentence'].apply(lambda x: re.sub(r'(?:m+){3,}','',x))
    all_Reviews_df['sentence'] = all_Reviews_df['sentence'].apply(lambda x: re.sub(r"\bhttp\w+", "", x))
    all_Reviews_df['sentence'] = all_Reviews_df['sentence'].apply(lambda x: re.sub(r"\brt\w+", "", x))
    all_Reviews_df['len'] = all_Reviews_df['sentence'].apply(lambda x:len(x))
    all_Reviews_df = all_Reviews_df[all_Reviews_df['len']>2]  
    all_Reviews_df = all_Reviews_df[all_Reviews_df['len']<300]
    
    all_Reviews_df1 = all_Reviews_df.iloc[0:20000,]
    all_Reviews_df2 = all_Reviews_df.iloc[20000:40000,]
    all_Reviews_df3 = all_Reviews_df.iloc[40000:,]

    all_Reviews_df1['themes'] = all_Reviews_df1['sentence'].apply(lambda x: extract_theme(x, nlp))
    all_Reviews_df2['themes'] = all_Reviews_df2['sentence'].apply(lambda x: extract_theme(x, nlp))
    all_Reviews_df3['themes'] = all_Reviews_df3['sentence'].apply(lambda x: extract_theme(x, nlp))

    all_Reviews_df = pd.concat([all_Reviews_df1, all_Reviews_df2, all_Reviews_df3], axis=0)
    all_Reviews_df['themes_bow'] = all_Reviews_df['themes'].apply(lambda x: '|'.join(x))
    all_Reviews_df['themes_bow'] = all_Reviews_df['themes_bow'].apply(lambda x: re.sub(r'\d+','',x))
    all_Reviews_df['themes_bow'] = all_Reviews_df['themes_bow'].apply(lambda x: re.sub('<[^<]+?>','', x))
    all_Reviews_df['themes_bow'] = all_Reviews_df['themes_bow'].apply(lambda x: re.sub('<[^<]+?>','', x))
        
    themes_for_clustering = all_Reviews_df['themes_bow'].str.split("|",expand=True)
    number_of_themes = len(themes_for_clustering.columns)    
    all_Reviews_df = pd.concat([all_Reviews_df,themes_for_clustering], axis=1)        
    all_Reviews_df = pd.melt(all_Reviews_df, id_vars=['review','date','source','review_id','sentences',
                             'sentence_id', 'sentence', 'sent_len','sentiment_new','themes', 'themes_bow'],
                             value_vars=list(range(0,number_of_themes)), value_name='themes_relation_extraction')
    all_Reviews_df = all_Reviews_df.sort_values(by=['review_id','sentence_id'], ascending=[True,True])    
    all_Reviews_df = all_Reviews_df[all_Reviews_df['themes_relation_extraction'].isna()==False]
    all_Reviews_df['themes_keyword'] = all_Reviews_df['themes_relation_extraction'].apply(lambda x: lemmatize_with_postag(x))
    all_Reviews_df['themes_keyword'] = all_Reviews_df['themes_keyword'].apply(lambda x: remove_stop_words(x))
    all_Reviews_df['theme_len'] = all_Reviews_df['themes_keyword'].apply(lambda x: len(x))
    all_Reviews_df = all_Reviews_df[all_Reviews_df['theme_len']>2]
    
    return all_Reviews_df