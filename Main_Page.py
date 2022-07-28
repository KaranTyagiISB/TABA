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
    
    
    ### Want to learn more?
    - Check out [streamlit.io](https://streamlit.io)
    - Jump into our [documentation](https://docs.streamlit.io)
    - Ask a question in our [community
        forums](https://discuss.streamlit.io)
    ### See more complex demos
    - Use a neural net to [analyze the Udacity Self-driving Car Image
        Dataset](https://github.com/streamlit/demo-self-driving)
    - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
"""
)

df = pd.read_csv('https://raw.githubusercontent.com/KaranTyagiISB/TABA/main/uber_reviews_itune.csv', encoding = 'latin1')
st.write(df.head(5))

st.text("")
st.text("")
st.text("")

uber_reviews = list(df.Review)
uber_review_clean = []

for i in uber_reviews :
    clean = re.sub(".\w+[+]+\d+\w+\d+[<>]","", i)
    clean = re.sub("\n\n"," ",clean)
    clean = re.sub("\n"," ",clean)
    clean = clean.lower()
    
    
    uber_review_clean.append(clean)
    
    
rating = pd.DataFrame(df["Rating"].value_counts().reset_index())
rating.columns = ["Rating","Count"]

plt.figure(figsize = (10,4))
 
ax = sns.barplot(x="Rating", y="Count", data=rating)    

for container in ax.containers:
    ax.bar_label(container)


plt.xlabel("Ratings", fontsize = 15)
plt.ylabel("Counts", fontsize = 15)

plt.yticks(fontsize = 12)
plt.title("Ratings", fontsize = 18, pad = 20)

plt.text(3,300,"Rating 1 - Extremely Low", fontsize = 12)
plt.text(3,270,"Rating 5 - Extremely High", fontsize = 12)

# Show Plot
st.pyplot(plt)


#For wordcloud
title = list(df["Title"].astype(str).str.lower())

import string
clean_title = []

for strings in title : 
    str1 = ""
    for y in strings:
        if (y in string.punctuation) == True:
            continue
        else:
            str1 = str1+y
            
    if str1 == "": pass
    else : clean_title.append(str1)

# load nltk's English stopwords as variable called 'stopwords'
stopwords = nltk.corpus.stopwords.words('english')

custom_stopwords = ["uber"]

stopwords = stopwords + custom_stopwords


import matplotlib.pyplot as plt
from wordcloud import WordCloud

custom_stop_words = ["uber", "t","s"]

use_stopwords = stopwords +custom_stop_words

text = " ".join(cat.split()[0] for cat in clean_title)
word_cloud = WordCloud(collocations = True, background_color = 'white',stopwords= use_stopwords).generate(text)

# Display the generated Word Cloud
plt.figure(figsize=(12, 8))
plt.imshow(word_cloud, interpolation='bilinear')

plt.title("WordCloud - Titles of the Reviews", fontsize = 18,pad=20)


plt.axis("off")

#plt.show()
st.pyplot(plt)
