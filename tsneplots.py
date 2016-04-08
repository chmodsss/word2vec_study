
# coding: utf-8

from gensim.models import Doc2Vec
from collections import defaultdict
import numpy as np
import random, gensim, logging
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD, PCA
from bokeh.plotting import figure, show, ColumnDataSource, output_notebook
from bokeh.models import HoverTool
from collections import namedtuple
from emotionslist import root_emotions_categorised, emotion_categories
import cPickle as pickle


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO

'''
bokeh models are available only in ipython notebook or the output figure could
be saved as a html image. It doesnt support pop up display as matplotlib.
'''
output_notebook()


model = gensim.models.Doc2Vec.load('/path/to/vector/model')


x_train = pickle.load(open('/path/to/training/sentences','rb'))
y_train = pickle.load(open('/path/to/training/emotions','rb'))


derived_emotions_categorised_dict = pickle.load(open('/path/to/derived/emotions','rb'))
derived_emotions_categorised_list = [category.keys() for category in derived_emotions_categorised_dict]


concatenated_emotions_list = [[] for x in range(len(emotion_categories))]
for idx in range(len(emotion_categories)):
    concatenated_emotions_list[idx] = root_emotions_categorised[idx] + derived_emotions_categorised_list[idx]

'''
Returns the sum of all the word vectors in the sentence with respective emotions.
When out of vocabulary word appears, the instance is skipped.
'''
def getWordVecs(model, corpus, category, size):
    vecs = []
    cat = []
    for i,words in enumerate(corpus):
        curr_vec = np.zeros((size))
        for word in words:
            try:
                curr_vec += model[word].reshape(size)
            except KeyError:
                break
        else:
            vecs.append(curr_vec)
            cat.append(category[i])
    result = namedtuple('result','vecs,cat')
    result.vecs = vecs
    result.emos = emos
    return result

'''
Returns the respective document vector by infer_vector method which uses
similarity measure to find the closest vector in the learnt vector space.
'''
def getDocVecs(model, corpus, emotions, size):
    vecs = []
    for z in corpus:
        vecs.append(np.array(model.infer_vector(z)).reshape((size)))
    result = namedtuple('result','vecs emos')
    result.vecs = vecs
    result.emos = emotions
    return result


model_features = getWordVecs(model, x_train, y_train, 400)
'''
In case of document vectors
sentence_vecs = getDocVecs(model_cbow, x_train, y_train, 400)
'''

emotion_count = defaultdict(int)
for emotion in model_features.emos:
    emotion_count[emotion] += 1

'''
vectors are randomly sampled in each category which reduces computational
complexity of the dimensionality reduction algorithm.
'''
def sample_data(trained_model, limit):
    sampled_train_vecs = []
    sampled_train_emotions = []
    for value in emotion_count.iteritems():
        specific_doc_vectors_in_model = [vec for idx,vec in enumerate(trained_model.vecs) if trained_model.emos[idx]==value[0]]
        emotion_limit = limit if limit < value[1] else value[1]
        perm = np.random.permutation(len(specific_doc_vectors_in_model))
        sampled_train_vecs.extend([y for x,y in sorted(zip(perm, specific_doc_vectors_in_model))[:emotion_limit]])
        sampled_train_emotions.extend([value[0] for idx in range(emotion_limit)])
    sampled_data = namedtuple('sampled_data','vectors,emotions')
    sampled_data.vectors = sampled_train_vecs
    sampled_data.emotions = sampled_train_emotions
    return sampled_data

limit = 5000
sampled_data = sample_data(model, limit)

emotion_categories.update({'none':'gray'})
c_val = [emotion_categories[emotion] for emotion in sampled_data.emotions]


def tsne_plot(data):
    x_val = []
    y_val = []

    for vals in data:
        x_val.append(vals[0])
        y_val.append(vals[1])

    source = ColumnDataSource(
            data = dict(
            x = x_val,
            y = y_val,
            z = c_val,
            legend = sampled_data.emotions,
        )
    )

    TOOLS="pan,wheel_zoom,box_zoom,reset,hover,save"

    plt = figure(title = "Tweets dataset visualised under TSNE", tools=TOOLS)

    plt.scatter('x','y', color = 'z',source=source)

    hover = plt.select(dict(type=HoverTool))
    hover.point_policy = "follow_mouse"

    show(plt)

'''
PCA plots
'''
pca_vecs = PCA(n_components=2).fit_transform(sampled_data.vectors)

'''
In order to compute tsne, high dimension vectors have to be reduced 
to a reasonable dimension using TruncatedSVD.
'''
svd_vecs = TruncatedSVD(n_components=60).fit_transform(sampled_data.vectors)
tsne_vecs = TSNE(n_components=2, method='barnes_hut', verbose=10).fit_transform(svd_vecs)
tsne_plot(tsne_vecs)