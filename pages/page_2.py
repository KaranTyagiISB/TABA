## setup chunk
import time   # to time 'em opns
t0 = time.time()    # start timer
import numpy as np
import pandas as pd
import nltk, re, requests
nltk.download('punkt')
nltk.download('popular')

from nltk.stem.snowball import SnowballStemmer as stemmer
from nltk import sent_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import matplotlib.pyplot as plt
import seaborn as sns


from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
from wordcloud import WordCloud

from bs4 import BeautifulSoup
import re
import os
import codecs
import string
import spacy

import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

from sklearn import feature_extraction
import gensim
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

import warnings
warnings.filterwarnings('ignore')
import streamlit as st


st.set_page_config(page_title="Topic Modeling", page_icon="📊")
st.markdown("# Topic Modeling")
st.sidebar.header("Topic Modeling")

##df = pd.read_csv('C:/Users/ktyagi/Desktop/ISB/Text Analytics/Assignment/uber_reviews_itune.csv', encoding = 'latin1')
df = pd.read_csv('https://raw.githubusercontent.com/KaranTyagiISB/TABA/main/uber_reviews_itune.csv', encoding = 'latin1')


st.text("")
st.text("")
st.text("")


##no._of_topics = st.text_input('Enter the Number of Topics:', value = '')
##st.write('The current movie title is', no._of_topics)


uber_reviews = list(df.Review)

## Removing newline character and encoded emoticon characters.

uber_review_clean = []

for i in uber_reviews :
    clean = re.sub(".\w+[+]+\d+\w+\d+[<>]","", i)
    clean = re.sub("\n\n"," ",clean)
    clean = re.sub("\n"," ",clean)
    clean = clean.lower()
    
    
    uber_review_clean.append(clean)
    
stopwords = nltk.corpus.stopwords.words('english')

custom_stopwords = ["uber","it's","i've","i'm", 'drivers','rides','idk']

stopwords = stopwords + custom_stopwords

clean_reviews = []
for x in uber_review_clean:
    str1 = ""
    for y in x:
        if (y in string.punctuation) == True:
            continue
        else:
            str1 = str1+y
    clean_reviews.append(str1)
clean_reviews1 = []

for x in clean_reviews:
    str1 = ""
    for y in x.split(" "):
        if "\x92" in y or "\x93" in y or "\x94" in y :
            continue
        else:
            str1 = str1+y+" "
    clean_reviews1.append(str1)

reviews = []
for x in clean_reviews1:
    l1 = []
    str1 = ''
    doc = x.split()
    for y in doc:
        if (y in stopwords)==True:
            continue
        elif y.isnumeric()==True:
            continue
        else:
            str1 = str1+y+' '
    #l1.append(str1)
            
    reviews.append(str1)

def word_tokenized(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    
    token_new = []
    
    for token in tokens :
        if token not in stopwords :
            token_new.append(token)
        else :pass
    
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in token_new:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
            
    return filtered_tokens

def stem(text):
    
    filtered_tokens =  word_tokenized(text)
    stems = []
    for token in filtered_tokens:
        stem = stemmer.stem(token)
        stems.append(stem)
    
    return stems

def total_tokens(data,col):
    if isinstance(data, list):lis = data
    else :lis = list(data[col])
        
    tokenized = []

    for i in lis:
        allwords_tokenized = word_tokenized(i)
        tokenized.extend(allwords_tokenized)
    
    return tokenized
    
def total_stem(data,col):
    if isinstance(data, list):lis = data
    
    else :lis = list(data[col])
        
    stemmed = []

    for i in lis:
        allwords_stemmed = stem(i)
        stemmed.extend(allwords_stemmed)
    
    return stemmed

# Use above funcs to iterate over the list of synopses to create two vocabularies: one stemmed and one only tokenized. 

totalvocab_tokenized = total_tokens(data = reviews,col= "")


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(reviews))

# Create Dictionary
id2word = corpora.Dictionary(data_words)

# Create Corpus
texts = data_words

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]



lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=5, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
                                           

#pyLDAvis.enable_notebook()

vis = gensimvis.prepare(lda_model,corpus,id2word,mds = 'mmds', R=30)


html_string = pyLDAvis.prepared_data_to_html(vis)
from streamlit import components
components.v1.html(html_string, width=1300, height=800, scrolling=True)






