
# coding: utf-8

import os, pickle, gensim, matplotlib
import matplotlib.pyplot as plt
from matplotlib import pylab
from sklearn.decomposition import PCA
import numpy as np
from simpleemotionaltweets import *
from collections import defaultdict, namedtuple


model = gensim.models.Word2Vec.load('path/to/vector/model')

'''
this example provides the result when v(king)-v(man)+v(woman)
Thus the vectors to be added are given in positive section and 
the vectors to subtract should be in negative section.
'''
model.most_similar(positive=[u'könig', u'frau'], negative=[u'mann'])

'''
This example also works good for the following conditions.
sonne - hitze = mond
    model.most_similar(positive=[u'sonne'], negative=[u'hitze'])
frau + kind = mutter
planet + wasser = erde
planet - wasser = saturn, jupiter
haus + film = kino
computer + telefonieren = handy
koenig - mann + frau = koenigin
'''


'''
Function returns the vectors with 2 dimensions by using PCA
the limit parameter controls how much vocabulary is added in addition
to the words to be plotted so that the PCA finds a generalised plane.
for the nintendo, xbox, computer example limit is set 0, as the variations 
in the vocabulary is enough to plot the differences.
In case of countries-capitals, the limit is around 50.
Limit is also 50 for the king queen example.
'''

def getPlottingVecs(nouns_in):
    voc = []
    voc.extend([wiki_model[word] for word in nouns_in])

    limit = 10
    for idx,x in enumerate(wiki_model.vocab.iteritems()):
        if idx<limit:
            voc.append(wiki_model[x[0]])
        else:
            break

    pca_vecs = PCA(n_components=2, whiten=True).fit(voc).transform(voc)

    vecs_to_plot = []
    for x in range(len(nouns_in)):
        vecs_to_plot.append(pca_vecs[x,:])
    return vecs_to_plot

'''
Function plots the 2d vectors of the words computed via PCA
'''

def plotSimilarVectors(data, label, clrs):
    fig, ax = plt.subplots(figsize=(9,9))
    for idx,val in enumerate(data):
        plt.scatter(val[0], val[1], marker='o', s=55, color=clrs[idx])
        plt.annotate(label[idx] ,xytext=pos[idx], textcoords='offset points', xy=(val[0], val[1]), arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))
        ax.grid('on')
        plt.xlabel('vector space')
        plt.ylabel('vector space')
    plt.show()
    
'''
One of the following set of examples could be tested.
The pos list gives the position of the words annotated in the plot.

nouns = [ u'nintendo', u'xbox', u'playstation',u'computer',u'handy',u'fernseher', u'pc', u'laptop', u'digitalkamera',u'demokratie', u'landwirtschaft']
pos = [(-70,40), (40,-40), (-30,-40), (-70,40), (-80,10) , (-50,70), (-70,40), (-70,-40),(20,-60),(-70,40),(-90,40)]
clrs = ['blue','blue','blue','blue','blue','blue','blue','blue','blue','blue','blue','blue','blue','blue',
       'blue','blue','blue','blue']


nouns = [u'usa', u'washington',u'deutschland','berlin', u'russland', u'moskau',u'türkei',u'ankara', 
         u'portugal',u'lissabon', u'japan',u'tokio', u'schweden',u'stockholm',
         u'griechenland',u'athen',u'bulgarien',u'sofia']
clrs = ['green','green','blue','blue','brown','brown','red','red','orange','orange','black','black','cyan','cyan',
       'magenta','magenta','pink','pink']
pos = [(35,-45),   (35,-45),    (-30,-65),     (-30,-45),   (30,-45),  (35,-55),(35,-45),  (35,0),   (35,0),
       (35,0),    (35,-45),(35,-35),  (25,0),   (35,0),      (35,0),      (35,0), (-35,45),  (35,0)]


nouns = [u'könig',u'frau',u'mann',u'königin']
pos = [(15,-15),   (-45,-5),    (-10,-25),     (15,-25)]
clrs = ['green','green','blue','blue']
'''
plotting_vecs = getPlottingVecs(nouns)
plotSimilarVectors(plotting_vecs, nouns, clrs)