# coding: utf-8

import codecs,  logging, gensim, nltk
from gensim.models.doc2vec import LabeledSentence

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO

default_stopwords = nltk.corpus.stopwords.words('german')

class TweetWordTrainer(object):

    def __init__(self, dir_name):
        self.dir_name = dir_name
    
    def __iter__(self):
        for line in codecs.open(self.dir_name,'r','utf-8'):
            try:
                translated_phrase = line.decode('unicode-escape').replace('#','')
                words = [word.lower() for word in translated_phrase.split() if word not in default_stopwords]
                if words:
                    yield words
            except UnicodeDecodeError:
                pass

class TweetSentenceTrainer(object):

    def __init__(self, dir_name):
        self.dir_name = dir_name
    
    def __iter__(self):
        for idx,line in enumerate(codecs.open(self.dir_name,'r','utf-8')):
            try:
                translated_phrase = line.decode('unicode-escape').replace('#','')
                words = [word.lower() for word in translated_phrase.split() if word not in default_stopwords]
                if words:
                    yield LabeledSentence(words, tags=['%s'%idx])
            except UnicodeDecodeError:
                pass


class WikiWordTrainer(object):

    def __init__(self, dir_name):
        self.dir_name = dir_name
    
    def __iter__(self):
        for idx,file_name in enumerate(os.listdir(self.dir_name)):
            for idxx,line in enumerate(codecs.open(os.path.join(self.dir_name, file_name),'r','utf-8')):
                translated = line.replace(',','').replace('.','')
                words = [word.lower() for word in translated.split() if (word not in default_stopwords)]
                yield words


class WikiSentenceTrainer(object):

    def __init__(self, dir_name):
        self.dir_name = dir_name
    
    def __iter__(self):
        for idx,file_name in enumerate(os.listdir(self.dir_name)):
            for idxx,line in enumerate(codecs.open(os.path.join(self.dir_name, file_name),'r','utf-8')):
                translated = line.replace(',','').replace('.','')
                words = [word.lower() for word in translated.split() if (word not in default_stopwords) ]
                yield LabeledSentence(words, tags=['%s'%idx+'%s'%idxx])
