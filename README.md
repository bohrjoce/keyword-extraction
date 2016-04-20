# keyword-extraction
Information Retrieval - Keyword Extraction Project

#Goal of the project
Introduce the use of k-means clustering, SVD and RakeRank (RAKE weight + TextRank reweight) for the keyword extraction task

#Evaluation
Using the SemEval2010 dataset test part. RAKE implemetation we benchmarked against can be found at https://github.com/zelandiya/RAKE-tutorial

#TODO
So far, our 3 schemes works relatively well compare to RAKE. But right now we're still working on support for phrase extraction.

#HowTo
Please refer to evaluate_*.py to see how it works. In general, simply run evaluate_single.py (with one command line argument specifying the method to use, which can be: 'svd' or 'raketr' or 'cluster'; the same applies to keyphrases extraction) to evaluate single keyword extraction and evaluate_multiple.py for keyphrases extraction.

python evaluate_{single, multiple}.py {svd, raketr, cluster}

Also we provided a simple web program in Flask to toy with!
