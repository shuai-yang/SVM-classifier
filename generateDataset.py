'''
import pandas as pd
import numpy as np  
import matplotlib as plot
'''
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd
'''
nltk.download('wordnet') # WordNet is a lexical database for the English language
nltk.download('stopwords')
nltk.download('punkt')

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
'''

import preprocessor as p # $ pip install tweet-preprocessor
df = pd.read_csv('samples.txt')
print(df.head())
#sent_text = nltk.sent_tokenize(texts) # this gives us a list of sentences

'''
tokens = p.tokenize('Preprocessor is #awesome 👍 @tweeter https://github.com/s/preprocessor')
# print(tokens)  # Preprocessor is $HASHTAG$ $EMOJI$ $URL$ #char array
charArray = "".join(tokens) # string
#print(tokens)
i = 0
j = 0
for char in charArray+" ":
   if char == ' ':
      print(charArray[i:j])  # split items by whiltespace
      i = j+1
   j = j +1





for token in tokens:
   print(token)

numOfTokens = 0
for i in tokens:
   numOfTokens = numOfTokens + 1
print(numOfTokens)


userID    tweetID      tokens
001       001          [Preprocessor, is, $HASHTAG$, $EMOJI$, $MENTION$, $URL$]
          002
          003
'''