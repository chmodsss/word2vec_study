# coding: utf-8

import codecs, nltk
import numpy as np
import gensim, logging
from __future__ import division

default_stopwords = nltk.corpus.stopwords.words('german')

'''
Reading odd one out / doesnt match evaluation file
'''
with codecs.open('doesnt_match_eval.txt','r','utf-8') as infile:
    ooo_text = infile.read().splitlines()


'''
Reading of opposite semantic analogy evaluation file
'''
with codecs.open('opposite_eval.txt','r','utf-8') as infile:
    opp_text = infile.read().splitlines()


'''
Reading of semantic analogy evaluation file
'''
with codecs.open('semantic_eval.txt','r','utf-8') as infile:
    syn_text = infile.read().splitlines()

'''
Separating questions and answers from the whole text intput
'''

doesntmatch_list = []
for lines in ooo_text:
    doesntmatch_list.append([words for words in lines.lower().split()])


opp_list = []
for lines in opp_text:
    opp_list.append([words for words in lines.lower().split()])


syn_list = []
for lines in syn_text:
    syn_list.append([words for words in lines.lower().split()])


'''
our of vocabulary evaluation method
'''
def test_ooo(model, test_list):
    result = []
    for items in test_list:
        try:
            ooo = model.doesnt_match(items)
            result.append(True if ooo==items[3] else False)
        except ValueError:
            print questions+[answers]
    return result


'''
opposite semantic analogy evaluation method
'''
def test_opp(model, test_list):
    result = []
    for items in test_list:
        try:
            opp = model.most_similar(positive =[items[1],items[2]], negative=[items[0]])[0][0]
            if (items[1] or items[2] or items[3]) in default_stopwords:
                break
            result.append(True if opp==items[3] else False)
        except KeyError:
            print items
    return result


'''
Semantic analogy evaluation method
'''
def test_syn(model, test_list):
    result = []
    for items in test_list:
        try:
            opp = model.most_similar(positive =[items[1],items[2]], negative=[items[0]])[0][0]
            if (items[1] or items[2] or items[3]) in default_stopwords:
                break
            result.append(True if opp==items[3] else False)
        except KeyError:
            print items
    return result



wiki_cbow_model = gensim.models.Word2Vec.load('/path/to/wiki/cbow/model')
wiki_sg_model = gensim.models.Word2Vec.load('/path/to/wiki/sg/model')
wiki_dm_model = gensim.models.Doc2Vec.load('/path/to/wiki/dm/model')


cbow_model = gensim.models.Word2Vec.load('/path/to/twt/cbow/model')
sg_model = gensim.models.Word2Vec.load('/path/to/twt/sg/model')
dm_model = gensim.models.Doc2Vec.load(('/path/to/twt/dm/model')


'''
Evaluation is carried out on 3 models, as dbow do not produce word vectors.
These 3 models with variations in twitter and wiki dataset are evaluated against
these analogy based queries.
'''

cbow_ooo_result = test_ooo(cbow_model, doesntmatch_list)
sg_ooo_result = test_ooo(sg_model, doesntmatch_list)
dm_ooo_result = test_ooo(dm_model, doesntmatch_list)


cbow_opp_result = test_opp(cbow_model, opp_list)
sg_opp_result = test_opp(sg_model, opp_list)
dm_opp_result = test_opp(dm_model, opp_list)


cbow_syn_result = test_syn(cbow_model, syn_list)
sg_syn_result = test_syn(sg_model, syn_list)
dm_syn_result = test_syn(dm_model, syn_list)


wiki_cbow_ooo_result = test_ooo(wiki_cbow_model, doesntmatch_list)
wiki_sg_ooo_result = test_ooo(wiki_sg_model, doesntmatch_list)
wiki_dm_ooo_result = test_ooo(wiki_dm_model, doesntmatch_list)

wiki_cbow_opp_result = test_opp(wiki_cbow_model, opp_list)
wiki_sg_opp_result = test_opp(wiki_sg_model, opp_list)
wiki_dm_opp_result = test_opp(wiki_dm_model, opp_list)

wiki_cbow_syn_result = test_syn(wiki_cbow_model, syn_list)
wiki_sg_syn_result = test_syn(wiki_sg_model, syn_list)
wiki_dm_syn_result = test_syn(wiki_dm_model, syn_list)


'''
The total % represents how much percent of the data is considered 
and not out of vocabulary in the model.
'''

print "CBOW odd one out :", sum(cbow_ooo_result)/len(cbow_ooo_result), " Total %:", len(cbow_ooo_result)/len(doesntmatch_list)
print "SG odd one out :", sum(sg_ooo_result)/len(sg_ooo_result), " Total %:", len(sg_ooo_result)/len(doesntmatch_list)
print "DM odd one out :", sum(dm_ooo_result)/len(dm_ooo_result), " Total %:", len(dm_ooo_result)/len(doesntmatch_list)


print "CBOW OPP :", sum(cbow_opp_result)/len(cbow_opp_result), " Total %:", len(cbow_opp_result)/len(opp_list)
print "SG OPP :", sum(sg_opp_result)/len(sg_opp_result), " Total %:", len(sg_opp_result)/len(opp_list)
print "DM OPP :", sum(dm_opp_result)/len(dm_opp_result), " Total %:", len(dm_opp_result)/len(opp_list)


print "CBOW SYN :", sum(cbow_syn_result)/len(cbow_syn_result), " Total %:", len(cbow_syn_result)/len(syn_list)
print "SG SYN :", sum(sg_syn_result)/len(sg_syn_result), " Total %:", len(sg_syn_result)/len(syn_list)
print "DM SYN :", sum(dm_syn_result)/len(dm_syn_result), " Total %:", len(dm_syn_result)/len(syn_list)


print "WIKI CBOW odd one out :", sum(wiki_cbow_ooo_result)/len(wiki_cbow_ooo_result), " Total %:", len(wiki_cbow_ooo_result)/len(doesntmatch_list)
print "WIKI SG odd one out :", sum(wiki_sg_ooo_result)/len(wiki_sg_ooo_result), " Total %:", len(wiki_sg_ooo_result)/len(doesntmatch_list)
print "WIKI DM odd one out :", sum(wiki_dm_ooo_result)/len(wiki_dm_ooo_result), " Total %:", len(wiki_dm_ooo_result)/len(doesntmatch_list)


print "WIKI CBOW opp :", sum(wiki_cbow_opp_result)/len(wiki_cbow_opp_result), " Total %:", len(wiki_cbow_opp_result)/len(opp_list)
print "WIKI SG opp :", sum(wiki_sg_opp_result)/len(wiki_sg_opp_result), " Total %:", len(wiki_sg_opp_result)/len(opp_list)
print "WIKI DM opp :", sum(wiki_dm_opp_result)/len(wiki_dm_opp_result), " Total %:", len(wiki_dm_opp_result)/len(opp_list)


print "WIKI CBOW syn :", sum(wiki_cbow_syn_result)/len(wiki_cbow_syn_result), " Total %:", len(wiki_cbow_syn_result)/len(syn_list)
print "WIKI SG syn :", sum(wiki_sg_syn_result)/len(wiki_sg_syn_result), " Total %:", len(wiki_sg_syn_result)/len(syn_list)
print "WIKI DM syn :", sum(wiki_dm_syn_result)/len(wiki_dm_syn_result), " Total %:", len(wiki_dm_syn_result)/len(syn_list)
