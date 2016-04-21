# Keyword-extraction
Information Retrieval - Keyword Extraction Project

#Goal of the project
Introduce the use of k-means clustering, SVD and TextRake (RAKE weight + TextRank reweight) for the keyword extraction task

#Requirement
Make surey you have the proper nltk package installed (for example, we used punkt for tokenizing, stemmers, lemmatizers etc...)

#Evaluation
Using the SemEval2010 dataset test part (92 documents). RAKE implemetation we benchmarked against can be found at https://github.com/zelandiya/RAKE-tutorial

#TODO
Improve keyphrase building by using Jaccard coefficient to remove keywords that are similar to others.

#HowTo
We provided demo.py to toy around with the 3 algorithms. Simply put your text files in a single folder and input in your command line: python demo.py *algo* *folder_link* *type*; where algo is either svd, raketr or cluster, and type is either "single" or "phrases"

For evaluation on dataset: please refer to evaluate_*.py to see how it works. In general, simply run evaluate_single.py (with one command line argument specifying the method to use, which can be: 'svd' or 'raketr' or 'cluster'; the same applies to keyphrases extraction) to evaluate single keyword extraction and evaluate_multiple.py for keyphrases extraction.

python evaluate_{single, multiple}.py {svd, raketr, cluster}

Also we provided a simple web program in Flask to toy with!

#Note
Running "evaluate_*.py cluster" may not yield the exact same values in our report, may be off by 1% due to convergence.
