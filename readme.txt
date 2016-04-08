
This folder contains the python scripts used in the work of detecting emotions from text.

Mainly two datasets are used in this work.
Twitter dataset and wikipedia dataset
Twitter dataset is generated manually by streaming german tweets for more than 3 months
from November 2015 to February 2016.
The twitter dataset will be uploaded soon and the link will be mentioned
at https://github.com/sivasuryas
The wikipedia dataset is available at http://www.cls.informatik.uni-leipzig.de/
18 million sentences are used in our work from the available datasets.

Some of the files explain the process of working and some of the files
are importable.
The files which walk through the processes are pointed out under 
each functions and the importable modules and files used is denoted under it.


1) Preprocessing the raw data and generation of vector models:
- vector_model_generation.py
--- dataprocessing.py
    -- swear_words_de.txt
--- modeltrainer.py


2) Word cloud representation of hashtags:
- hashtag_cloud.py
--- extracthashtags.py

3) Analogy based evaluation for vector models:
- analogy_test.py
--- doesnt_match_eval.txt
--- semantic_eval.txt
--- opposite_eval.txt

4) Data visualisation of vectors using PCA and TSNE:
- tsneplots.py


3) Generate training and test dataset for the classification:
- generate_training_data.py
--- extractlabeledtweets.py


5) Evaluation of vector models on the datasets applied.
- evaluation_biased_dataset.py
--- evaluation_sents.csv
--- emotionslist.py
- evaluation_unbiased_dataset.py
--- evaluation_sents.csv
--- emotionslist.py

Incase of any additional files, put the path in global_paths.py and import it.