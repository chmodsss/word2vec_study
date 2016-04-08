import codecs, os
from operator import countOf
from collections import namedtuple
from emotionslist import *

class MySentences(object):
    
    def __init__(self, file_name):
        self.file_name = file_name
    
    def __iter__(self):
        for lines in codecs.open(self.file_name, 'r', 'utf-8'):
            yield lines.split()

class ExtractLabeledTweets(object):

    global categories_length
    
    categories_length = len(emotion_categories)
    
    def __init__(self, target_file, limit):
        self.target_file = target_file
        self.limit = limit
        self.line_count = 0
        self.no_emotion_count = 0
        self.no_emotion_limit = 20000
        self.child_emotions_categorised = [{} for i in range(categories_length)]
        self.limited_child_emotions_categorised = [{} for i in range(categories_length)]
        self.unduplicated_emotions_categorised = [{} for x in range(categories_length)]
        self.limited_child_emotions_container = []

    def _check_single_root_emotion(self, hashtags):        
        single_emotion_property = namedtuple('single_emotion_property','status tags category')
        single_emotion_property.status = False
        
        single_emotion_property.tags = [tag for tag in hashtags if tag in root_emotions_container]
        single_emotion_property.category = list({idx for idx in range(categories_length) for tag in hashtags if tag in root_emotions_categorised[idx]})

        
        if len(single_emotion_property.category) == 1:
            single_emotion_property.status = True
        
        return single_emotion_property
    
    def _check_single_emotion(self, root_emotion, child_emotion):
        root_category = list({idx for idx in range(categories_length) for tag in root_emotion if tag in root_emotions_categorised[idx]})
        child_category = list({idx for idx in range(categories_length) for tag in child_emotion if tag in self.limited_child_emotions_categorised[idx].keys()})
        if len(root_category) and len(child_category) == 1:
            if root_category[0] == child_category[0]:
                return True
        elif (len(root_category) == 1 or len(child_category) == 1):
            return True
        return False
    
    def _add_child_emotions(self, hashtags, category):
        for tag in hashtags:
            self.child_emotions_categorised[category][tag] = self.child_emotions_categorised[category].get(tag,0) + 1
        return
    
    def _remove_duplicate_emotions(self, duplicated_emotions):
        for idx in range(categories_length):
            for emotion in duplicated_emotions[idx]:
                flag = False
                for idxx in range(categories_length):
                    if (idx is not idxx):
                        if (emotion in duplicated_emotions[idxx].keys()) and (duplicated_emotions[idx][emotion] <= duplicated_emotions[idxx][emotion]):
                            flag = True
                if not flag:
                    self.unduplicated_emotions_categorised[idx][emotion] = duplicated_emotions[idx][emotion]
        return self.unduplicated_emotions_categorised
    
    def _file_len(self, fname):
        with open(fname) as f:
            for i, l in enumerate(f):
                pass
        return i + 1
    
    def extract_labels(self):

        with codecs.open(self.target_file, 'r', 'utf-8') as infile:
            raw_tweets = infile.readlines()
        total_lines = self._file_len(self.target_file)
        self.line_count = 0
        
        for line in raw_tweets:
            self.line_count += 1
            print "{:.2f}".format(self.line_count*100.0/total_lines) ,"\r", "Data processed for root emotions. . .  ",

            hashtags = [words[1:] for words in line.split() if words.startswith('#')]
            if hashtags:
                single_emotion_result = self._check_single_root_emotion(hashtags)
                if single_emotion_result.status:
                    child_hashtags = [tags for tags in hashtags if tags not in single_emotion_result.tags]
                    self._add_child_emotions(child_hashtags, single_emotion_result.category[0])        

        for idx in range(categories_length):
            self.limited_child_emotions_categorised[idx] = dict(tags for tags in self.child_emotions_categorised[idx].items() if tags[1]>self.limit)

        self.limited_child_emotions_categorised = self._remove_duplicate_emotions(self.limited_child_emotions_categorised)
                    
        self.limited_child_emotions_container = dict(emotions for category in self.limited_child_emotions_categorised for emotions in category.items())
        
        self.cnt = 0
        print "\n"
        
        base,ext = os.path.splitext(self.target_file)
        write_file = open(self.target_file.rstrip(ext) + '_e' + ext ,'w')
        self.line_count = 0
        for line in raw_tweets:
            print "{:.2f}".format(self.line_count*100.0/total_lines) ,"\r", "Data processed for child emotions. . .  ",
            self.line_count += 1
            hashtags = [words[1:] for words in line.split() if words.startswith('#')]
            root_emotion = [tag for tag in hashtags if tag in root_emotions_container]
            child_emotion = [tag for tag in hashtags if tag in self.limited_child_emotions_container.keys()]
            single_emotion_result = self._check_single_emotion(root_emotion, child_emotion)
            if single_emotion_result:
                write_file.write(line)
            elif self.no_emotion_count < self.no_emotion_limit:
                if (root_emotion==[]) and (child_emotion==[]):
                    self.no_emotion_count += 1
                    write_file.write(line)
        write_file.close()
        labeled_dataset = namedtuple('labeled_dataset','file_name derived_emotions_categorised')
        labeled_dataset.file_name = write_file.name
        labeled_dataset.derived_emotions_categorised = self.limited_child_emotions_categorised
        return labeled_dataset