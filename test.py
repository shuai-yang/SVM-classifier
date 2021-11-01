import os
Current_Directory = os.getcwd()
#print(Current_Directory)
CORPUS_DIR='E:\SVM-classifier'
DOCUMENTS = os.listdir(CORPUS_DIR) if os.path.exists(CORPUS_DIR) else []
#print(DOCUMENTS)
import re
TITLE_CONTENT_REGEX = re.compile(r'Title: (.*?)\n[.\n]*?\"(.*)', flags=re.MULTILINE|re.DOTALL)
import preprocessor as p
def parse_raw_file(document):
    '''Return title and content of raw file.'''
    with open(os.path.join(CORPUS_DIR, document)) as fp:
        data = fp.read()
        print(data)
    #return '', ''
parse_raw_file('samples.txt')
'''
 # get document sentences
with open(os.path.join(CORPUS_DIR, 'tweets.txt')) as f:
    data = f.read()
sentences = sent_tokenize(data)
'''