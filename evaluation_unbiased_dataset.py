# coding: utf-8

import gensim, pickle, csv, nltk, matplotlib, pylab, logging
from sklearn.metrics import roc_curve, auc
import numpy as np
from scipy import interp
from collections import namedtuple, defaultdict
from string import punctuation
from sklearn.metrics import confusion_matrix
from simpleemotionaltweets import emotion_categories
from matplotlib import pyplot as plt
from sklearn.linear_model import SGDClassifier

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO

'''
'none' is detected when there are no hashtags related to other categories, so once the
training data is developed it is also appended with the emotion_categories during the evaluation.
'''
emotion_categories.update({'none':'gray'})


'''
Set the respective file paths for model, traning data and test data.
'''
model_path = 'path/to/vector/model'
x_train_path = 'path/to/training/sentences'
y_train_path = 'path/to/training/emotions'
x_test_path = 'path/to/test/sentences'
y_test_path = 'path/to/test/emotions'


model = gensim.models.Word2Vec.load(model_path)

x_train = pickle.load(open(x_train_path,'rb'))
y_train = pickle.load(open(y_train_path,'rb'))

x_test = pickle.load(open(x_test_path,'rb'))
y_test = pickle.load(open(y_test_path,'rb'))


'''
Returns the sum of all the word vectors in the sentence with respective emotions.
When out of vocabulary word appears, the instance is skipped.
'''
def getWordVecs(model, corpus, category, size):
    vecs = []
    emos = []
    for i,words in enumerate(corpus):
        curr_vec = np.zeros((size))
        for word in words:
            try:
                curr_vec += model[word].reshape(size)
            except KeyError:
                break
        else:
            vecs.append(curr_vec)
            emos.append(category[i])
    result = namedtuple('result','vecs emos')
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

trained_model = getWordVecs(model, x_train, y_train, 400)
test_model = getWordVecs(model, x_test, y_test, 400)
'''
For document vector models
trained_model = getDocVecs(model, x_train, y_train, 400)
test_model = getDocVecs(model, x_test, y_test, 400)
'''

'''
emotion_count_revised orders the dictionary based on emotion categories order.
'''
emotion_count = defaultdict(int)
for emotions in trained_model.emos:
    emotion_count[emotions] += 1

emotion_count_revised = OrderedDict()
for k in emotion_categories.keys():
    emotion_count_revised[k] = emotion_count[k]



clustered_vecs = [[] for x in range(len(emotion_categories))]
clustered_emos = [[] for x in range(len(emotion_categories))]

for idx,val in enumerate(emotion_categories.keys()):
    for idxx,values in enumerate(trained_model.emos):
        if values == val:
            clustered_vecs[idx].append(trained_model.vecs[idxx])
            clustered_emos[idx].append(trained_model.emos[idxx])

'''
The clusters for each category is found out by trial and error method.
Clusters with less samples could not pick required samples, so the cluster
size is decreaed and more samples fall into the cluster.
tot_samples is set as 1300, because the lowest samples among all the categories
is 1300 from disgust.
sampling the instances by clustering is implemented.
'''
clusters = [4, 6, 6, 1, 2, 3, 4]
sampled_vecs = []
sampled_emos = []
tot_samples = 1300
    
for idx,val in enumerate(emotion_categories.keys()):
    kms = KMeans(n_clusters = clusters[idx])
    kms.fit(clustered_vecs[idx])
    limit = tot_samples if tot_samples < emotion_count_revised.values()[idx] else emotion_count_revised.values()[idx]
    samples = limit/clusters[idx]

    for cluster in range(clusters[idx]):
        sample_count = 0
        for idxx,label in enumerate(kms.labels_):
            if cluster == label:
                sampled_vecs.append(clustered_vecs[idx][idxx])
                sampled_emos.append(clustered_emos[idx][idxx])
                sample_count += 1
                if sample_count>=samples:
                    break


'''
Linear regression classifier with stochastic gradient learning is used
in our evaluation scenario.
'''
lr = SGDClassifier(loss='log', penalty='l1')
lr.fit(sampled_vecs, sampled_emos)


'''
emotion categories predicted for the test vectors
'''
predicted_op = lr.predict(test_model.vecs)


'''
decision_function provides the value by which the hyperplane is
separated which is used in ROC curves
'''
predicted_score = lr.decision_function(test_model.vecs)


def plot_confusion_matrix(cm, cmap=plt.cm.Greens):
    fig, ax = plt.subplots(figsize=(9,9))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    cb = plt.colorbar()
    cb.set_label("Predicted values")
    tick_marks = np.arange(len(emotion_categories))
    plt.xticks(tick_marks, emotion_categories.keys(), rotation=45)
    plt.yticks(tick_marks, emotion_categories.keys())
    width, height = np.shape(cm)
    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(cm[x][y]), xy=(y, x), horizontalalignment='center', verticalalignment='center')
    ax.xaxis.tick_top()
    ax.xaxis.set_ticks_position('both') # THIS IS THE ONLY CHANGE

    plt.ylabel('Emotion categories', labelpad=15)
    plt.xlabel('Predicted categories', labelpad=15)
    ax.xaxis.set_label_position('top')
    plt.show()

'''
Expected emotion categories and predicted categories are plotted aginst eachother
using confusion matrices
'''
cmm = confusion_matrix(test_model.emos, predicted_op, labels=emotion_categories.keys())
plot_confusion_matrix(cmm)


'''
Both the predicted categories and expected categories are binarized inorder to
plot the ROC curves for each emotional category
'''
binarized_train = label_binarize(sampled_emos, classes=emotion_categories.keys())
binarized_expected = label_binarize(test_model.emos, classes=emotion_categories.keys())

'''
OneVsRestClassifier is applied on the logistic regression classifier to detect
binarized outputs too
'''
lr = SGDClassifier(loss='log',penalty='l1')
ovr = OneVsRestClassifier(lr)
ovr.fit(sampled_vecs, binarized_train)

predicted_ovr = ovr.decision_function(test_model.vecs)

'''
computing false positive rates and true positive rates for each category
'''
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = len(emotion_categories)
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(binarized_expected[:,i], predicted_ovr[:,i])
    roc_auc[i] = auc(fpr[i], tpr[i])

'''
Computing the true positive and false positive values for micro average ROC curve
'''
fpr["micro"], tpr["micro"], _ = roc_curve(binarized_expected.ravel(), predicted_ovr.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

'''
Computing the micro-average for ROC curves referenced from
http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
'''

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure(figsize=(9,9))

plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ({0:0.2f})'
               ''.format(roc_auc["micro"]),
         linewidth=3)

for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label='{0} ({1:0.2f})'
        ''.format(emotion_categories.keys()[i], roc_auc[i]), linewidth=1.5, color=emotion_categories.values()[i])

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.grid('on')
plt.xlabel('False Positive Rate', fontsize=18)
plt.ylabel('True Positive Rate', fontsize=18)
plt.legend(loc="upper left", bbox_to_anchor=(-0.06,-0.1), ncol=3, columnspacing=0.5)
#pylab.savefig(roc_title, dpi=150 ,bbox_inches='tight')
plt.show()

'''
In order to plot only the micro-average of all curves, the values need to be saved.
All the results of the roc curves are pickled
'''
roc_property = namedtuple('roc_prop','fpr tpr roc_auc')
pickle.dump(roc_property, open('/path/for/roc/curves'))

'''
To unpickle, the namedtuple has to be set first.
'''
roc_prop = namedtuple('roc_prop','fpr tpr roc_auc')
# for example
cbow_roc_property = pickle.load(open('/path/to/cbow/roc')) # for example
cbow_fpr = cbow_roc_property.fpr
cbow_tpr = cbow_roc_property.tpr
cbow_roc_auc = cbow_roc_property.roc_auc
