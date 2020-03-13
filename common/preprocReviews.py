import pandas as pd
import numpy as np
from googletrans import Translator
from sklearn.feature_extraction.text import TfidfVectorizer
import gender_guesser.detector as gender
import re
import itertools
from re import search
import operator
import functools
import ftfy
from ftfy import fix_text
from nltk.tokenize import sent_tokenize, word_tokenize, WordPunctTokenizer
import string
from bs4 import BeautifulSoup
import functools
import copy

import logging
logger = logging.getLogger('sialogger')


def detect_language(x, spacy_nlp):
    sent = spacy_nlp(x)
    sentence_lang = sent._.language['language']
    return sentence_lang


def search_substring(substring, fullstring):
    if search(substring, fullstring):
        return fullstring
    else:
        return substring + ". " + fullstring


def identify_names(x, spacy_nlp):
    doc = spacy_nlp(x)
    entities = [(x.text, x.label_) for x in doc.ents]
    names = [i[0] for i in entities if i[1] == 'PERSON']
    names1 = [char.split() for char in names]
    names2 = list(itertools.chain.from_iterable(names1))
    review_without_person_name = ' '.join([word for word in x.split() if word not in names2])
    return review_without_person_name


def rmMentions(text):
    text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())
    return text


def rmStringPunc(text):
    text = ''.join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text


def tweet_cleaner(text):
    tok = WordPunctTokenizer()
    pat1 = r'@[A-Za-z0-9]+'
    pat2 = r'https?://[A-Za-z0-9./]+'
    combined_pat = r'|'.join((pat1, pat2))

    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    stripped = re.sub(combined_pat, '', souped)
    try:
        clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except Exception as e:
        logger.info("Exception is {}".format(e))
        clean = stripped
    letters_only = re.sub("[^a-zA-Z]", " ", clean)
    # lower_case = letters_only.lower()
    # During the letters_only process two lines above, it has created unnecessay white spaces,
    # I will tokenize and join together to remove unneccessary white spaces
    words = tok.tokenize(letters_only)
    return (" ".join(words)).strip()


def rm_dup_strings(text):
    text = word_tokenize(text)
    dedup_str = []
    for num in text:
        if num not in dedup_str:
            dedup_str.append(num)
    return " ".join(dedup_str)


def extract_emojis(x, demoji):
    emoji_dict = demoji.findall(x)
    emoji_desc = list(emoji_dict.values())
    emoji_desc = ','.join(emoji_desc)
    return emoji_desc


# def rm_bias(author):
#     author = author.lower()
#     brand_list = ['cafe', 'coffee', 'starbucks', 'mccafe', 'dunkin', 'nescafe', 'nestle', 'folgers', 'maxwell', 'peet', 'kirkland']
#
#     bias = [1 if search(word, author) else 0 for word in brand_list]
#     if sum(bias)>0:
#         return ''
#     else:
#         return author


def rm_long_words(x):
    words = x.split()
    text = ' '.join([word for word in words if len(word) < 18])
    return text


def rm_junk(x):
    text = ' '.join(word for word in x.split() if not word.startswith('gt'))
    return text


def rm_hypertext(x):
    text = ' '.join(word for word in x.split() if not word.startswith('http'))
    text = ' '.join(word for word in x.split() if not word.startswith('www'))
    return text


def amazonPreProcess(amazon_data, spacy_nlp):
    amazon_data = amazon_data[['title', 'body', 'date', 'usefullness', 'author', 'Review_Count']]
    amazon_data['body'] = amazon_data['body'].apply(lambda x: str(x).replace(")", ""))
    amazon_data['body'] = amazon_data['body'].apply(lambda x: str(x).replace("(", ""))
    amazon_data['body'] = amazon_data['body'].apply(lambda x: str(x).strip())
    amazon_data['body'] = amazon_data['body'].apply(lambda x: re.sub(r'[!@#$+?<>*//\\]', '', x))
    amazon_data['title'] = amazon_data['title'].apply(lambda x: str(x).replace(")", ""))
    amazon_data['title'] = amazon_data['title'].apply(lambda x: str(x).replace("(", ""))
    amazon_data['title'] = amazon_data['title'].apply(lambda x: str(x).strip())
    amazon_data['title'] = amazon_data['title'].apply(lambda x: re.sub(r'[!@#$+?<>*//\\]', '', x))
    amazon_data['final_content'] = amazon_data[['title', 'body']].apply(lambda x: search_substring(x.title, x.body),
                                                                        axis=1)
    amazon_data['final_content'] = amazon_data['final_content'].apply(lambda x: identify_names(x, spacy_nlp))
    amazon_data.dropna(subset=['final_content'], inplace=True)
    amazon_data['language'] = amazon_data['final_content'].apply(lambda x: detect_language(x, spacy_nlp))
    amazon_data = amazon_data[amazon_data['language'] == "en"]
    amazon_data['final_content'] = amazon_data['final_content'].apply(lambda x: rm_long_words(x))
    amazon_data['final_content'] = amazon_data['final_content'].apply(lambda x: rm_junk(x))
    amazon_data['final_content'] = amazon_data['final_content'].apply(lambda x: rm_hypertext(x))
    amazon_data['len'] = amazon_data['body'].apply(lambda x: len(x))
    amazon_data = amazon_data[amazon_data['len'] > 2]
    amazon_data_pp = copy.deepcopy(amazon_data)
    amazon_data_pp['usefullness'] = amazon_data['usefullness'].apply(
        lambda x: str(x).replace("One", "1") if None != x else x)
    amazon_data_pp['usefullness'] = amazon_data_pp['usefullness'].apply(lambda x: str(x)[0] if None != x else x)
    amazon_data_pp.loc[amazon_data_pp['usefullness'] == 'n', 'usefullness'] = None
    amazon_data_pp.loc[amazon_data_pp['usefullness'] == 'P', 'usefullness'] = None
    amazon_data_pp.rename(columns={'final_content': 'review', 'usefullness': 'Repeat_Count'}, inplace=True)
    amazon_data_pp['source'] = 'Amazon'
    return amazon_data_pp


def fbPreProcess(facebook_data, spacy_nlp):
    facebook_data.rename(columns={'commentWithAuthorname': 'comment_text'}, inplace=True)
    facebook_data = facebook_data[facebook_data['comment_text'] != True]
    facebook_data['comment_text'] = facebook_data['comment_text'].apply(lambda x: identify_names(x, spacy_nlp))
    facebook_data['comment_text'] = facebook_data['comment_text'].apply(lambda x: str(x).replace(")", ""))
    facebook_data['comment_text'] = facebook_data['comment_text'].apply(lambda x: str(x).replace("(", ""))
    facebook_data['comment_text'] = facebook_data['comment_text'].apply(lambda x: str(x).strip())
    facebook_data['comment_text'] = facebook_data['comment_text'].apply(lambda x: re.sub(r'[!@#$+?<>*//\\]', '', x))
    facebook_data['comment_text'] = facebook_data['comment_text'].apply(lambda x: re.sub(r'[!@#$+?<>*//\\]', '', x))
    facebook_data['comment_text'] = facebook_data['comment_text'].apply(lambda x: rm_long_words(x))
    facebook_data['comment_text'] = facebook_data['comment_text'].apply(lambda x: rm_junk(x))
    facebook_data['comment_text'] = facebook_data['comment_text'].apply(lambda x: rm_hypertext(x))
    facebook_data['len'] = facebook_data['comment_text'].apply(lambda x: len(x))
    facebook_data = facebook_data[facebook_data['len'] > 2]
    facebook_data.rename(columns={'comment_text': 'review', 'time': 'date'}, inplace=True)
    facebook_data.drop('len', axis=1, inplace=True)
    facebook_data_pp = copy.deepcopy(facebook_data)
    facebook_data_pp['source'] = 'Facebook'

    return facebook_data_pp


def twitterPreProcess(twitter_data, spacy_nlp):
    twitter_data = twitter_data.dropna(subset=["text"], inplace=False)
    twitter_data['key'] = list(range(1, twitter_data.shape[0] + 1, 1))
    twitter_data['language'] = twitter_data['text'].apply(lambda x: detect_language(x, spacy_nlp))
    twitter_data = twitter_data[twitter_data['language'] == "en"]
    twitter_data['text'] = twitter_data['text'].apply(lambda x: identify_names(x, spacy_nlp))
    twitter_data_pp = twitter_data.drop_duplicates(subset='text')
    # twitter_data_pp['Author'] = twitter_data_pp['Author'].fillna(' ')
    # twitter_data_pp['author_name'] = twitter_data_pp['Author'].apply(lambda x: rm_bias(x))
    # twitter_data_pp = twitter_data_pp[twitter_data_pp['author_name'] != '']
    twitter_data_pp['text'] = twitter_data_pp['text'].apply(lambda x: re.sub('[^., a-zA-Z0-9]', '', x))
    twitter_data_pp['text'] = twitter_data_pp['text'].apply(lambda x: rm_dup_strings(x))
    twitter_data_pp['text'] = twitter_data_pp['text'].apply(lambda x: str(x).replace(")", ""))
    twitter_data_pp['text'] = twitter_data_pp['text'].apply(lambda x: str(x).replace("(", ""))
    twitter_data_pp['text'] = twitter_data_pp['text'].apply(lambda x: str(x).strip())
    twitter_data_pp['text'] = twitter_data_pp['text'].apply(lambda x: re.sub(r'[!@#$+?<>*//\\]', '', x))
    twitter_data_pp['text'] = twitter_data_pp['text'].apply(lambda x: re.sub(r'[!@#$+?<>*//\\]', '', x))
    twitter_data_pp['text'] = twitter_data_pp['text'].apply(lambda x: rm_long_words(x))
    twitter_data_pp['text'] = twitter_data_pp['text'].apply(lambda x: rm_junk(x))
    twitter_data_pp['text'] = twitter_data_pp['text'].apply(lambda x: rm_hypertext(x))
    twitter_data_pp['len'] = twitter_data_pp['text'].apply(lambda x: len(x))
    twitter_data_pp = twitter_data_pp[twitter_data_pp['len'] > 2]
    twitter_data_pp.drop('len', axis=1, inplace=True)
    twitter_data_pp.rename(columns={'text': 'review', 'time': 'date'}, inplace=True)
    twitter_data_pp['source'] = 'Twitter'

    return twitter_data_pp


def create_final_input(all_Reviews_df,demoji):
    all_Reviews_df['review'] = all_Reviews_df['review'].apply(lambda x: re.sub(r'\.+', ".", x))
    all_Reviews_df['review'] = all_Reviews_df['review'].apply(lambda x: re.sub(r'\.', ". ", x))
    all_Reviews_df['review'] = all_Reviews_df['review'].apply(lambda x: fix_text(x))
    all_Reviews_df['sentences'] = all_Reviews_df['review'].apply(lambda x: sent_tokenize(x))
    all_Reviews_df['id'] = list(range(1, all_Reviews_df.shape[0] + 1, 1))
    all_Reviews_df['sentences1'] = all_Reviews_df['sentences'].apply(lambda x: '|'.join(x))
    all_sentences = all_Reviews_df.sentences1.str.split('|', expand=True)
    all_Reviews_df = pd.concat([all_Reviews_df, all_sentences], axis=1)
    all_Reviews_df = pd.melt(all_Reviews_df, id_vars=['review', 'date', 'source', 'id', 'sentences'],
                             value_vars=list(range(0, len(all_sentences.columns), 1)))
    all_Reviews_df = all_Reviews_df.sort_values(by=['id', 'variable'], ascending=[True, True])
    all_Reviews_df = all_Reviews_df[all_Reviews_df['value'].isna() == False]
    all_Reviews_df.rename(columns={'id': 'review_id', 'variable': 'sentence_id', 'value': 'sentence'}, inplace=True)
    all_Reviews_df['sent_len'] = all_Reviews_df['sentence'].apply(lambda x: len(x))
    all_Reviews_df = all_Reviews_df[all_Reviews_df['sent_len'] > 2]
    all_Reviews_df['emoji_desc'] = all_Reviews_df['sentence'].apply(lambda x: extract_emojis(x,demoji))
    all_Reviews_df['sentence'] = all_Reviews_df['sentence'].map(str) + ", " + all_Reviews_df['emoji_desc'].map(str)
    all_Reviews_df['sentence'] = all_Reviews_df['sentence'].apply(lambda x: re.sub(r'[^a-zA-Z., ]', '', x))

    return all_Reviews_df
