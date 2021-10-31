'''
import pandas as pd
import numpy as np  
import matplotlib as plot
'''
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, TweetTokenizer
'''
nltk.download('wordnet') # WordNet is a lexical database for the English language
nltk.download('stopwords')
nltk.download('punkt')

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class mapping:
    def __init__(self):
       self.d = {}
    def add(self, k, v):
       self.d[k] = v
       self.d[v] = k
    def remove(self, k):
       self.d.pop(self.d.pop(k))
    def get(self, k):
       return self.d[k]
'''


import pandas as pd
def generateDataframe(textsFileName):
   df = pd.read_csv(inputFilename, header=None, sep="\n", names=['userTweets'] ) #separator is \n
   df['userID'] = df.index + 1
   df = df[[ 'userID','userTweets']]
   #print(df)
   return df
def generateUserToScoreMapping(scoresFileName):
   scores = pd.read_csv('score.txt', header=None, sep="\n", names=['userScores'] ) #separator is \n
   scores['userID'] = df.index + 1
   scores = scores[[ 'userID','userScores']]
   #print(scores)
   return scores
df = generateDataframe('texts.txt')
scores = generateUserToScoreMapping('scores.txt')


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