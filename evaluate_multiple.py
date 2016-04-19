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
import rake_tr
import re
import rake

nltk.data.path.append('/home/jocelyn/usb/nltk_data')

def get_stemmed_keywords(keywords):

  stemmer = EnglishStemmer()
  stemmed_keywords = list(keywords)
  # replace - with space ???
#  stemmed_keywords = [keyword.replace('-', ' ') for keyword in stemmed_keywords]
  # split into list of list
  stemmed_keywords = [keyword.split() for keyword in stemmed_keywords]
  # stem individual words
  stemmed_keywords = [list(stemmer.stem(word) for word in keyword) for keyword in stemmed_keywords]
  # list of words to string
  stemmed_keywords = [' '.join(keyword) for keyword in stemmed_keywords]

  return stemmed_keywords

def main():

  semeval_dir = 'data/maui-semeval2010-test/'
  filenames = sorted(os.listdir(semeval_dir))
  manual_keywords = []
  total_precision = 0
  total_recall = 0
  total_docs = 0

  for filename in filenames:
    if filename[-3:] == 'key':
      if filename == "H-5.key":
        continue
      with open(semeval_dir + filename, 'r') as f:
        last_key_file = filename
        key_lines = f.read().splitlines()
        #for word in key_lines:
          #print word
          #word = unicode(word, 'utf-8')
          # if filename == "H-5.key":
          #   print word
        key_lines = [word.encode('ascii') for word in key_lines]
        manual_keywords = get_stemmed_keywords(key_lines)

    elif filename[-3:] == 'txt':
      if filename == "H-5.txt":
        continue
      total_docs += 1
      print(filename)
      with open(semeval_dir + filename, 'r') as f:
        correct = 0
        f = open(semeval_dir + filename, 'r')
        content = f.read()
#        keywords = svd(content)
        # keywords = rake_tr.main(content)
#        keywords = kcluster(content)
        keywords = rake.main(content)
        # print(keywords)
        print('-'*100)
#        print('--------manual keywords---------')
#        print(manual_keywords)
#        print('--------extracted keywords---------')
#        print(keywords)
        stemmed_keywords = get_stemmed_keywords(keywords)
#        print "FILENAME: " + filename
        for keyword in stemmed_keywords:
          if keyword in set(manual_keywords):
            correct += 1
        if len(manual_keywords) == 0:
          print(filename)
          print(last_key_file)
          print('^^^^ issue with this file ^^^^')
          exit(0)
        total_precision += correct/float(len(keywords))
        total_recall += correct/float(len(manual_keywords))


  # total_docs = len(filenames)/2
  total_precision /= total_docs
  total_recall /= total_docs
  total_fmeasure = round(2*total_precision*total_recall/(total_precision + total_recall), 2)
  print('total docs: ' + str(total_docs))
  print('total precision: ' + str(total_precision))
  print('total recall: ' + str(total_recall))
  print('total fmeasure: ' + str(total_fmeasure))

if __name__ == '__main__':
  main()
