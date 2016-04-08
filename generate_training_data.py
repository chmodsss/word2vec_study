# coding: utf-8

import os, codecs, logging, pickle, gensim, csv, nltk
import numpy as np
from string import punctuation
import multiprocessing as mp
from gensim.models.doc2vec import LabeledSentence
from collections import defaultdict, namedtuple
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import SGDClassifier, LogisticRegression
from gensim.models import Doc2Vec
from extractlabeledtweets import ExtractLabeledTweets
from emotionslist import *
from itertools import izip_longest

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO

'''
Only the tweets with emotion hashtags are filtered and the derived hashtags
which occur more than hashtag_frequency are collected
'''
hashtag_frequency = 100
labeled_twitter_path = ExtractLabeledTweets(unlabeled_twitter_path, hashtag_frequency).extract_labels()
derived_emotions_categorised = labeled_dataset.derived_emotions_categorised

derived_emotions_list = [category.keys() for category in derived_emotions_categorised]

concatenated_emotions_list = [[] for x in range(len(emotion_categories)]
for idx in range(len(emotion_categories)):
    concatenated_emotions_list[idx] = root_emotions_categorised[idx] + derived_emotions_list[idx]


'''
Set the path of labeled twitter database when the preprocessing is done
'''
labeled_twitter_path = 'set/labeled/twitter/path'


'''
Split the tweets into messages and hashtags
'''
def split_line(line):s
    tweet = namedtuple('tweet', 'message hashtags')
    try:
        tweet.hashtags = [word for word in line.split() if word.startswith('#')]
        tweet.message = [word for word in line.split() if (word not in tweet.hashtags)]
    except Exception as e:
        tweet.hashtags = None
    return tweet


'''
The emotional category for the respective hashtag or return 'none' for no emotions
'''
def build_emotion_dict(hashtags):
    emotion_category = None
    for each_hashtag in hashtags:
        emotion_category = [emotion_categories.items()[idx][0] for idx in range(6) if each_hashtag[1:] in concatenated_emotions_list[idx]]
        if (emotion_category):
            return emotion_category[0]
        else:
            return 'none'


'''
Splits the data for parallel processing, much better when multi-core system is used
'''
def grouper(n, iterable, padvalue=None):
    return izip_longest(*[iter(iterable)]*n, fillvalue=padvalue)


'''
Returns a dictionary containing the message as the key and the respective
emotion category to be its value
'''
def worker(tweet):
    if(tweet):
        tweet_property = split_line(tweet)
        if tweet_property.hashtags:
            emotion_category = build_emotion_dict(tweet_property.hashtags)
            emotioned_tweet = {}
            if emotion_category:
                emotioned_tweet[' '.join(tweet_property.message)] = emotion_category
                return emotioned_tweet
            else:
                emotioned_tweet[' '.join(tweet_property.message)] = 'none'
                return emotioned_tweet
    return None


'''
Multiprocessing module from python is employed so that the whole dataset is
splitted into chunks and each chunk is processed parallely which saves time
'''
cores = 48
block = 1000
tweet_message = []
tweet_emotion = []

with codecs.open(labeled_twitter_path,'r','utf-8') as infile:
    raw_tweets = infile.read().splitlines()

# Remove the backward slashes in the dataset as it clashes with escape sequences and causes error
tweets = []
double_slash = dict()
double_slash[ord('\\')]=None

for line in raw_tweets:
    try:
        if(line):
            line = line.decode('unicode-escape').translate(double_slash)
            tweets.append(line)
    except UnicodeDecodeError:
        pass

p = mp.Pool(cores)

for chunk in grouper(block, tweets):
    results = p.map(worker, chunk)
    for r in results:
        if (r):
            tweet_message.append(r.keys()[0].split())
            tweet_emotion.append(r.values()[0])


'''
Annotated dataset from Emorobot project sentences
'''
annotated_sents_dict = {}
with open('evaluation_sents.csv','r') as infile:
    reader = csv.reader(infile)
    for rows in reader:
        annotated_sents_dict[rows[0].decode('utf-8')] = rows[1].decode('utf-8')


'''
Process the annotated dataset to remove punctuations and stop words
'''
def preprocess_test_data(evaluation_data):
    processed_lines = []
    processed_emotions = []
    punct = punctuation.replace('#','').replace('\\','')
    default_stopwords = nltk.corpus.stopwords.words('german')
    default_stopwords = [word.encode('utf-8') for word in default_stopwords]
    for idx,line in enumerate(evaluation_data.keys()):
        translated_phrase = line.encode('utf-8').translate(None, punct)
        translated_phrase = [word.decode('utf-8') for word in translated_phrase.lower().split() if word not in default_stopwords]
        if translated_phrase:
            processed_lines.append(translated_phrase)
            processed_emotions.append(evaluation_data.values()[idx].decode('utf-8'))
    processed_data = namedtuple('processed_data','sentence emotion')
    processed_data.sentence = processed_lines
    processed_data.emotion = processed_emotions
    return processed_data

processed_test_data = preprocess_test_data(annotated_sents_dict)
annotated_sentences = processed_test_data.sentence
annotated_emotions = processed_test_data.emotion

'''
counts the number of instances in each emotion category
'''
emotion_count = defaultdict(int)
for emotions in tweet_emotion:
    emotion_count[emotions] += 1

'''
Sample the training instances randomly upto the limit specified if there are
more instances than the limit
'''
def sample_data(messages, emotions, limit):
    equalised_messages = []
    equalised_emotions = []
    for value in emotion_count.iteritems():
        specific_messages = [msg for idx,msg in enumerate(messages) if emotions[idx]==value[0]]
        emotion_limit = limit if limit < value[1] else value[1]
        perm = np.random.permutation(len(specific_messages))
        equalised_messages.extend([y for x,y in sorted(zip(perm, specific_messages))[:emotion_limit]])
        equalised_emotions.extend([value[0] for idx in range(emotion_limit)])
    equalised_data = namedtuple('sampled_data','message emotion')
    equalised_data.message = equalised_messages
    equalised_data.emotion = equalised_emotions
    return equalised_data

sample_limit = 10000
sampled_data = sample_data(tweet_message, tweet_emotion, sample_limit)
twt_sentences = sampled_data.message
twt_emotions = sampled_data.emotion

'''
The twitter data training set and annotated test set are pickled for later use
'''
pickle.dump(twt_sentences, open('twt_sentences','wb'))
pickle.dump(twt_emotions, open('twt_emotions','wb'))
pickle.dump(annotated_sentences, open('annotated_sentences','wb'))
pickle.dump(annotated_emotions, open('annotated_emotions','wb'))


'''
The twitter dataset is split into twt80 and twt20 to evaluate as how the feature
vectors perform in the same dataset
'''
twt80_sentences, twt20_sentences, twt80_emotions, twt20_emotions = train_test_split(twt_sentences, twt_emotions, test_size=0.2)
pickle.dump(twt80_sentences, open('twt80_sentences','wb'))
pickle.dump(twt80_emotions, open('twt80_emotions','wb'))
pickle.dump(twt20_sentences, open('twt20_sentences','wb'))
pickle.dump(twt20_emotions, open('twt20_emotions','wb'))
