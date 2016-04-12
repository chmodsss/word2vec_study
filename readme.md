## Emotion detection using word embeddings (word2vec & doc2vec)

Emotion classification from german sentences with the help of feature vectors generated from word2vec and doc2vec models is experimented.
From generating word and sentence vectors to classifying emotions, all the process are written in the python scripts.


In order to generate high quality feature vectors, huge datasets are needed.
Mainly two datasets are used in this work.
- Twitter dataset
- Wikipedia dataset

Twitter dataset is generated manually by streaming german tweets for more than 3 months (November 2015 to February 2016).
The Twitter dataset is publicly available and could be used by anyone at
https://app.box.com/german-twitter-dataset.
It contains 8 folders each of appx. 2.5M sentences.
Concatenate all the files to a single file for use.

Selected files from http://www.cls.informatik.uni-leipzig.de are used in wikipedia dataset and is available at http://www.cls.informatik.uni-leipzig.de.
It contains 18 folders each of 1M sentences.
The model could be learnt by iterating via all the files in a directory without the need of concatenating all the files.

Training and test data for emotion classification is generated from the raw twitter dataset by automatic labeling based on the hashtags.

The following functions are tried out in the python scripts.

##### 1) Preprocessing the raw data and generation of vector models:
The raw data containing stop words, numbers, obscene words, unwanted characters are preprocessed and the vectors are generated from the processed data.
`vector_model_generation.py` explains the steps used in the process.
It makes use of processing functions in `dataprocessing.py` for preprocessing and `modeltrainer.py` for generating vector models.

##### 2) Word cloud representation of hashtags:
To easily visualise the hashtags present in the twitter data, word clouds are used.
The script `hashtag_cloud.py` outputs a word cloud graph from randomly sampled hashtags using `extracthashtags.py`

##### 3) Analogy based evaluation for vector models:
To test the goodness of the vectors generated, they are evaluated against some analogy based questions.
The script `analogy_test.py` evaluates the model for odd one out using `doesnt_match_eval.txt`, semantic analogy using `semantic_eval.txt` and opposite semantic analogy using `opposite_eval.txt`. The relations among certain words could also be visualised with the help of `word_relations.py`

##### 4) Data visualisation of vectors using PCA and TSNE:
To get a visualisation of the vectors being dealt with, those vectors are randomly sampled and visualised using PCA and TSNE using `tsneplots.py`.

##### 3) Generate training and test dataset for the classification:
The automatic labeling of the twitter dataset based on the hashtags in the tweets is applied in `extractlabeledtweets.py` and from the tweets and the hashtags belonging to each emotional category, the training and test data are generated using `generate_training_data.py`.
The root emotions from which the derived emotions are generated are in `emotionslist.py`

##### 5) Evaluation of vector models on the datasets applied.
The evaluation of vector models for emotion classification is done both on datasets with unequal samples in each emotion category due to the dataset's nature and unbiased dataset by downsampling the samples using clusters.
Both the scripts `evaluation_biased_dataset.py` and `evaluation_unbiased_dataset.py` uses `evaluation_sents.csv` as the annotated evaluation file.
