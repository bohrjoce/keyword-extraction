# Keyword-extraction
Information Retrieval - Keyword Extraction Project

#Goal of the project
Introduce the use of k-means clustering, SVD and TextRake (RAKE weight + TextRank reweight) for the keyword extraction task

#Requirement
Make surey you have the proper nltk package installed (for example, we used punkt for tokenizing, stemmers, lemmatizers etc...)

#Evaluation
Using the SemEval2010 dataset. Around 300 documents!

#TODO
Improve keyphrase building by using Jaccard coefficient to remove keywords that are similar to others.

#HowTo
Please refer to evaluate.py to see how it works. In generally, simply run evaluate_single.py (with one command line argument specifying the method to use, which can be: 'svd' or 'textrake' or 'cluster'; the same applies to keyphrases extraction) to evaluate single keyword extraction and evaluate_multiple for keyphrases extraction.

Also we provided a simple web program in Flask to toy with!
