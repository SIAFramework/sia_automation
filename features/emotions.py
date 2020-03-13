"""
Created on Fri Nov  8 16:22:30 2019
@author: AkOjha
"""

from __future__ import division
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
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob, Word
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from numbers import Number
from pandas import DataFrame
import sys, codecs, numpy



def relationship_exception():

    general_relationship_exceptions1 = ['am','rt' ,'th', 'to', 'coffee', 'cafe', 'starbucks', 'nestle']
    general_relationship_exceptions2 = ['can','cans','homework','let','others']
    general_relationship_exceptions3 = ['bit', 'side', 'usual', 'it', 'part', 'walmart', 'local']
    general_relationship_exceptions4 = ['end', 'chance', 'item', 'items', 'tuesday', 'monday', 'thing', 'one', 'It',  'side', 'mg', 'oz', 'ammount',  'today',  'aed', 'name']
    general_relationship_exceptions5 = ['first', 'next', 'same', 'cc','motherfuckers']
    general_relationship_exceptions6 = ['aaaa', 'aaa', 'ive', 'w', 'httpbit', 'u', 'yall', 'id', 'th', 'bc', 'httpst', 'oz', 'n', 's', 'ta', 'm', 'st', 'ur', 'TRUE', 'ya', 'hi', 'x', 'k', 'b', 'nd', 'etc', 'iv', 'p', 'c', 'lil', 'httpdlvr', 'yo', 'rn', 'smh', 'vs']
    general_relationship_exceptions = general_relationship_exceptions1 + general_relationship_exceptions2 + general_relationship_exceptions3 + general_relationship_exceptions4 + general_relationship_exceptions5 + general_relationship_exceptions6
    return general_relationship_exceptions


def lemmatize_with_postag(x):
    sent = TextBlob(x)
    tag_dict = {"J": 'a', 
                "N": 'n', 
                "V": 'v', 
                "R": 'r'}
    words_and_tags = [(w, tag_dict.get(pos[0], 'n')) for w, pos in sent.tags]    
    lemmatized_list = [wd.lemmatize(tag) for wd, tag in words_and_tags]
    return " ".join(lemmatized_list)


def remove_relation_stopwords(x):
    custom_stop_words = relationship_exception()
    querywords = x.split()
    resultwords  = [word for word in querywords if word.lower() not in custom_stop_words]
    result = ' '.join(resultwords)
    return result


def extract_relationship(row, nlp):
    
    input_doc = row.sentence
    theme_word = row.themes_relation_extraction
    
    input_doc = nlp(input_doc)
    relation_list = [(word.text, (input_doc.sentences[0].words[word.governor-1].text if word.governor > 0 else 'root'),
                      word.dependency_relation,) for word in input_doc.sentences[0].words]
    relation_df = pd.DataFrame(relation_list)
    relation_df.columns = ['theme_word', 'emotion', 'relationship']
    c = relation_df[relation_df['theme_word']==theme_word]
    if c.shape[0]<1:
        return None
    #Rule first
    if (c.relationship.values[0] in ['amod', 'nmod', 'advmod']) and (c.emotion.values[0] not in ['spam', 'musubi', 'spambrand']):
        return c.head(1).emotion.values[0]
    elif c.relationship.values[0] in ['nsubj', 'obj']:
        return c.head(1).emotion.values[0]
    elif (c.relationship.values[0] in ['xcomp','compound']) and ([word.upos for sent in input_doc.sentences for word in sent.words if (word.text==c.head(1).emotion.values[0] and word.upos=='ADJ')]):
        return c.head(1).emotion.values[0]
    elif (c.relationship.values[0] =='obl') and ([word.upos for sent in input_doc.sentences for word in sent.words if (word.text==c.head(1).emotion.values[0] and word.upos=='ADJ')]):  
        return c.head(1).emotion.values[0]
    elif (c.relationship.values[0] =='advcl') and ([word.upos for sent in input_doc.sentences for word in sent.words if (word.text==c.head(1).emotion.values[0] and word.upos=='ADJ')]):  
        return c.head(1).emotion.values[0]
    elif (c.relationship.values[0] =='conj') and ([word.upos for sent in input_doc.sentences for word in sent.words if (word.text==c.head(1).emotion.values[0] and word.upos=='ADJ')]):  
        return c.head(1).emotion.values[0]
    elif (c.relationship.values[0] =='obl') and ([word.upos for sent in input_doc.sentences for word in sent.words if (word.text==c.head(1).emotion.values[0] and word.upos=='VERB')]):  
        return c.head(1).emotion.values[0]
    elif c.relationship.values[0] == 'root' or c.relationship.values[0] == 'obl' or c.relationship.values[0] == 'conj' or c.relationship.values[0] == 'xcomp' or c.relationship.values[0] == 'compound' or c.relationship.values[0] == 'ccomp':
        d = relation_df[relation_df['emotion']==theme_word]
        if sum(d['relationship'].isin(['amod']))>0:
            return d[d['relationship'].isin(['amod'])].head(1).theme_word.values[0]
        elif sum(d['relationship'].isin(['advmod','nmod']))>0:
                return d[d['relationship'].isin(['advmod','nmod'])].head(1).theme_word.values[0]
        elif sum(d['relationship'].isin(['obj']))>0:
             if d[d['relationship'].isin(['obj'])].head(1).relationship.values[0] == 'ADJ':
                return d[d['relationship'].isin(['obj'])].head(1).theme_word.values[0]
        elif sum(d['relationship'].isin(['acl','xcomp','compound']))>0:
            if d[d['relationship'].isin(['acl','xcomp','compound'])].head(1).relationship.values[0] == 'ADJ':
                return d[d['relationship'].isin(['acl','xcomp','compound'])].head(1).theme_word.values[0]
        elif c.relationship.values[0] == 'obl':
            e = relation_df[relation_df['emotion']==theme_word]
            if sum(e['relationship'].isin(['advmod', 'amod', 'obj', 'nmod']))>0:
                return e[e['relationship'].isin(['advmod', 'amod', 'obj', 'nmod'])].head(1).theme_word.values[0]


def rm_stopwords(x, english_stopwords):
    
    querywords = x.split("|")
    resultwords  = [word for word in querywords if word.lower() not in english_stopwords]
    result = '|'.join(resultwords)
    return result


def tag_emotions(themes_df, english_stopwords, nlp):
    
    themes_df = themes_df.sort_values(by = ['review_id', 'sentence_id', 'variable'], ascending=[True, True, True])  
    themes_df = themes_df[themes_df['themes_keyword'].isna()==False]
    themes_df['relation_word'] = themes_df.apply(lambda x: extract_relationship(x, nlp),axis=1)
    themes_df['relation_word'].fillna('', inplace=True)
    themes_df['relation_word'] = themes_df['relation_word'].apply(lambda x: remove_relation_stopwords(x))    
    themes_df['emotion_keyword'] = themes_df['relation_word'].apply(lambda x: lemmatize_with_postag(x))
    themes_df['emotion_keyword'] = themes_df['relation_word'].apply(lambda x: rm_stopwords(x, english_stopwords))
    
    return themes_df