import pandas as pd
import numpy as np  #csv
import matplotlib as plot
 
import nltk
'''
nltk.download('wordnet') # WordNet is a lexical database for the English language
nltk.download('stopwords')
#nltk.download('punkt')
nltk.download('stopwords')'''
 
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
 
#import hashedindex as indexed
 
corpus = pd.read_csv('disney_plus_shows.csv')
head = corpus.head()
 
def preprocessing(text):
   return [WordNetLemmatizer.lemmatize(token) for token in word_tokenize(text.lower()) if not token.is_stop]
 
print(preprocessing(head))
 
 

