# keyword-extraction
Information Retrieval - Keyword Extraction Project

#Goal of the project
Introduce the use of k-means clustering, SVD and RakeRank (RAKE weight + TextRank reweight) for the keyword extraction task

#Evaluation
Using the SemEval2010 dataset

#TODO
So far, our 3 schemes works relatively well compare to RAKE. But right now we're still working on support for phrase extraction.

#HowTo
Please refer to evaluate.py to see how it works. In generally, simply run evaluate_single.py (with one command line argument specifying the method to use, which can be: 'svd' or 'textrake' or 'cluster'; the same applies to keyphrases extraction) to evaluate single keyword extraction and evaluate_multiple for keyphrases extraction.

Also we provided a simple web program in Flask to toy with!
