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



import warnings
warnings.filterwarnings('ignore')
import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome !")
st.write("# Uber Voice of Customer Analysis (Reviews)!")
st.subheader("Namaste, salaam, satsriakal.")
         
st.sidebar.success("Select a demo above.")

st.markdown(
    """
    Uber Inc in the US wants to know:

    1. The major complaints premium users have about their cab services,
    2. How these impact service ratings.
     - **Data Source** - The data are API collected from itunes for iOS users. The dataset uber_reviews_itune.csv is small, containing a mere 490 records.
     
    **The goal of this analysis is to leverage text analytics techniques to find the key themes around customer issues/concerns/complaints as well as happy experiences of customers.**
    
"""
)

