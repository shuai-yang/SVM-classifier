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
import preprocessor as prep
#tokens = prep.tokenize('Preprocessor is #awesome 👍 @tweeter https://github.com/s/preprocessor')
# print(tokens)  # Preprocessor is $HASHTAG$ $EMOJI$ $URL$ #char array
#charArray = "".join(tokens) # string
#print(tokens)

import pandas as pd
import preprocessor as prep
#from IPython.display import HTML
def getUserToTweets(textsFileName):
   df = pd.read_csv(textsFileName, header=None, sep="\n", names=['userTweets'] ) #separator is \n
   df['userTweetsProcessed'] = df['userTweets'].apply(TweetTokenizer)
   pd.set_option('display.max_colwidth', None)
   print(df)
   #df['userTweets'] = df['userTweets'].astype(str)
   #df['userID'] = df.index + 1
   #df = df[[ 'userID','userTweets']]
   '''
   for tweet in df[]
      userToLabel = df.set_index('userID').to_dict('userLabels')
   return userToLabel
   '''
   #return df

def getUserToLabel(labelsFileName):
   df = pd.read_csv(labelsFileName, header=None, sep="\n", names=['userLabel'] ) #separator is \n
   df['userID'] = df.index + 1
   userToLabel = df.set_index('userID').to_dict('list')
   return userToLabel
   #scores['userID'] = scores.index + 1
   #scores = scores[[ 'userID','userScores']]
   #userToScores = mapping()
   #for userID in df['userID']:
   #   userToScores[userID] = scores
   #userToScores.add(scores['userID'], scores['userScores'])
   #return userToScores
df = getUserToTweets('samples.txt')
#print(df)
#UserToLabel = getUserToLabel('labels.txt')
#print(UserToLabel)


'''

   userToLabel = {}
   for index in scores['userScores']:
      print(score)
      userToLabel[score.index] = score
   return userToLabel 


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