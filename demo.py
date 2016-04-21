#!/bin/python
# -*- coding: utf-8 -*-
from sklearn import cluster
import numpy as np
import nltk
import os
from nltk.corpus import stopwords
from clustering import kcluster
from nltk.stem.snowball import EnglishStemmer
from svd import svd
import feature_extract
import raketr
import re
#import rake
import sys

nltk.data.path.append('/home/jocelyn/usb/nltk_data')

def get_stemmed_keywords(keywords):

  stemmer = EnglishStemmer()
  stemmed_keywords = list(keywords)
  # split into list of list
  stemmed_keywords = [keyword.split() for keyword in stemmed_keywords]
  # stem individual words
  stemmed_keywords = [list(stemmer.stem(word) for word in keyword) for keyword in stemmed_keywords]
  # list of words to string
  stemmed_keywords = [' '.join(keyword).encode('ascii') for keyword in stemmed_keywords]

  return stemmed_keywords

def main():

  semeval_dir = 'data/maui-semeval2010-test/'
  manual_keywords = []
  total_precision = 0
  total_recall = 0
  total_docs = 0
  method = str(sys.argv[1])
  fdir = str(sys.argv[2])
  single = False
  if str(sys.argv[3]) == 'single':
    single = True
  filenames = sorted(os.listdir(fdir))

  for filename in filenames:
    if (filename[0] =='.'):
      continue
    print(filename)
    f = open(fdir + filename, 'r')
    content = f.read()
    if method == 'svd':
      keywords = svd(content, 1, single)
    elif method == 'textrake':
      keywords = raketr.main(content, single)
    elif method == 'cluster':
      keywords = kcluster(content, 6, 15, single)
    else:
      print('methods accepted: svd textrake cluster, please specify')
      exit(0)
    print('keyphrases found')
    print(keywords)
if __name__ == '__main__':
  main()
