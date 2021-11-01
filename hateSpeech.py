import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

import pandas as pd
import preprocessor as prep
def getUserTweets(textsFileName):
   df = pd.read_csv(textsFileName, header=None, sep="\n", names=['userTweets'] ) #separator is \n
   df['userID'] = df.index + 1
   df['sentences'] = df['userTweets'].apply(sent_tokenize)
   pd.set_option('display.max_colwidth', None)
   print(df['sentences'])
getUserTweets('samples.txt')