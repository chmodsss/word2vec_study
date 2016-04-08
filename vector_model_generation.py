#!/usr/bin/python
# coding: utf-8
from datapreprocessing import removeSpecialChars, removeObscenity, removeDuplicates
import global_paths
from extractlabeledtweets import ExtractLabeledTweets
from modeltrainer import TweetWordTrainer, TweetSentenceTrainer, WikiWordTrainer, WikiSentenceTrainer

if __name__ == '__main__':
	main()

## preprocessing steps are applied on the raw dataset

# Unwanted characters and stop words are removed from the dataset
special_chars_removed_path = removeSpecialChars(global_paths.dataset_path)

# Obscene words are removed from the dataset
obscenes_removed_path = removeObscenity(special_chars_removed_path, global_paths.swear_words_path)

# Duplicates are removed
unlabeled_twitter_path = removeDuplicates(obscenes_removed_path)

## Training a word vector model

'''
The following each block of text generate each model on the database specified
The parameters sg, dm decides which model is being used
for cbow model, set sg=0
for sg model, set sg=1
for dbow model, set dm =0
for dm model, set dm=1
'''
# Twitter CBOW model
tweet_words = TweetWordTrainer(unlabeled_twitter_path)
twt_cbow = gensim.models.Word2Vec(min_count=5, window=10, iter=3, sg=0, size=400, workers=40)
twt_cbow.build_vocab(tweet_words)
twt_cbow.train(tweet_words)
twt_cbow.save('Tweet_cbow.model')

# Twitter SG model
tweet_words = TweetWordTrainer(unlabeled_twitter_path)
twt_sg = gensim.models.Word2Vec(min_count=5, window=10, iter=3, sg=1, size=400, workers=40)
twt_sg.build_vocab(tweet_words)
twt_sg.train(tweet_words)
twt_sg.save('Tweet_sg.model')

# Twitter DBOW model
tweet_sentences = TweetSentenceTrainer(unlabeled_twitter_path)
twt_dbow = gensim.models.Doc2Vec(min_count=5, window=10, dm=0, size=400, iter=3, workers=40)
twt_dbow.build_vocab(tweet_sentences)
twt_dbow.train(tweet_sentences)
twt_dbow.save('Tweet_dbow.model')

# Twitter DM model
tweet_sentences = TweetSentenceTrainer(unlabeled_twitter_path)
twt_dm = gensim.models.Doc2Vec(min_count=5, window=10, dm=1, size=400, iter=3, workers=40)
twt_dm.build_vocab(tweet_sentences)
twt_dm.train(tweet_sentences)
twt_dm.save('Tweet_dm.model')

# Wiki CBOW model
wiki_words = TweetWordTrainer(wiki_path)
wiki_cbow = gensim.models.Word2Vec(min_count=5, window=10, iter=3, sg=0, size=400, workers=40)
wiki_cbow.build_vocab(wiki_words)
wiki_cbow.train(wiki_words)
wiki_cbow.save('Wiki_cbow.model')

# Wiki SG model
wiki_words = TweetWordTrainer(wiki_path)
wiki_sg = gensim.models.Word2Vec(min_count=5, window=10, iter=3, sg=1, size=400, workers=40)
wiki_sg.build_vocab(wiki_words)
wiki_sg.train(wiki_words)
wiki_sg.save('Wiki_sg.model')

# Wiki DBOW model
wiki_sentences = TweetSentenceTrainer(wiki_path)
wiki_dbow = gensim.models.Doc2Vec(min_count=5, window=10, dm=0, size=400, iter=3, workers=40)
wiki_dbow.build_vocab(wiki_sentences)
wiki_dbow.train(wiki_sentences)
wiki_dbow.save('Wiki_dbow.model')

# Wiki DM model
wiki_sentences = TweetSentenceTrainer(wiki_path)
wiki_dm = gensim.models.Doc2Vec(min_count=5, window=10, dm=1, size=400, iter=3, workers=40)
wiki_dm.build_vocab(wiki_sentences)
wiki_dm.train(wiki_sentences)
wiki_dm.save('Wiki_dm.model')