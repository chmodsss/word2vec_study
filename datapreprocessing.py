import os
import nltk
import codecs
import collections
import hashlib
from nltk import sent_tokenize
from string import punctuation
import multiprocessing as mp
from multiprocessing import pool
from itertools import izip_longest

global del_words, punct, default_stopwords, cores, block, swear_words

del_words = '@'
punct = punctuation.replace('#', '').replace('\\', '')
default_stopwords = nltk.corpus.stopwords.words('german')
cores = 48
block = 10000
swear_words = []

def grouper(n, iterable, padvalue = None):
    return izip_longest(fillvalue = padvalue, *[
        iter(iterable)] * n)


def removeSpecialCharsWorker(line):
    if line:
        line = line.rstrip()
        if 'http' not in line:
            translated_phrase = line.encode('utf-8').translate(None, punct)
            words_list = [word for word in translated_phrase.split() if (not word.startswith(del_words)) and (not word.isdigit()) and (word not in default_stopwords)]
            if words_list:
                    return ' '.join(words_list).lower()
    return None


def removeSpecialChars(target_file):
    with codecs.open(target_file, 'r', 'utf-8') as infile:
        read_data = infile.read().splitlines()
    (base, ext) = os.path.splitext(target_file)
    write_file = codecs.open(target_file.rstrip(ext) + '_c' + ext, 'w', 'utf-8')
    p = mp.Pool(cores)
    for chunk in grouper(block, read_data):
        results = p.map(removeSpecialCharsWorker, chunk)
        for r in results:
            if r:
                write_file.write(r)
                write_file.write('\n')
    
    p.close()
    p.join()
    write_file.close()
    return write_file.name


def removeObscenityWorker(line):
    if line:
        line = line.rstrip()
        if not any([word for word in line.split() if word in swear_words]):
            return line
        return None

def removeObscenity(target_file, swear_file):
    with codecs.open(target_file, 'r', 'utf-8') as infile:
        read_data = infile.read().splitlines()
    (base, ext) = os.path.splitext(target_file)
    write_file = codecs.open(target_file.rstrip(ext) + '_o' + ext, 'w', 'utf-8')
    with codecs.open(swear_file, 'r', 'utf-8') as infile:
        raw_swear_words = infile.read().splitlines()
    tag_swear_words = [ '#' + word for word in raw_swear_words ]
    global swear_words 
    swear_words = raw_swear_words + tag_swear_words
    p = mp.Pool(cores)
    for chunk in grouper(block, read_data):
        results = p.map(removeObscenityWorker, chunk)
        for r in results:
            if r:
                write_file.write(r)
                write_file.write('\n')
                continue
    
    p.close()
    p.join()
    write_file.close()
    return write_file.name


class MySentences(object):
    
    def __init__(self, file_name):
        self.file_name = file_name

    
    def __iter__(self):
        for lines in codecs.open(self.file_name, 'r', 'utf-8'):
            yield lines
        

def removeDuplicates(target_file):
    hash_table = collections.defaultdict(list)
    (base, ext) = os.path.splitext(target_file)
    write_file = codecs.open(target_file.rstrip(ext) + '_d' + ext, 'w', 'utf-8')
    sentences = MySentences(target_file)
    for line in sentences:
        if line:
            line = line.rstrip()
            id = hashlib.sha512(line).digest()
            key = id[0:2]
            value = id[2:]
            if value not in hash_table[key]:
                write_file.write(line)
                write_file.write('\n')
                hash_table[key].append(value)
            
    write_file.close()
    return write_file.name