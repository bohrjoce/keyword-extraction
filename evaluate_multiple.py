#!/bin/python
# -*- coding: utf-8 -*-
import rake
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
  filenames = sorted(os.listdir(semeval_dir))
  manual_keywords = []
  total_precision = 0
  total_recall = 0
  total_docs = 0
  method = str(sys.argv[1])

  for filename in filenames:
    if filename[-3:] == 'key':
      # ignored due to issue on Mac or empty keyfile
      if filename == "H-5.key" or filename == "C-86.key":
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
      # ignored due to issue on Mac or empty keyfile
      if filename == "H-5.txt" or filename == "C-86.txt":
        continue
      total_docs += 1
      print(filename)
      with open(semeval_dir + filename, 'r') as f:
        correct = 0
        f = open(semeval_dir + filename, 'r')
        content = f.read()
        if method == 'svd':
          keywords = svd(content, 1, False)
        elif method == 'textrake':
          keywords = raketr.main(content, False)
        elif method == 'cluster':
         keywords = kcluster(content, 6, 15, False)
#        keywords = rake.main(content)
#        keywords = rake_object.run(content)[:15]
#        keywords = [word[0] for word in keywords]
#        keywords = [''.join([i if ord(i) < 128 and i != '\n' else ' ' for i in keyword]).encode('ascii') for keyword in keywords]
        else:
          print('methods accepted: svd textrake cluster')
          exit(0)
        print(keywords)
        print('-'*100)
#        print('--------manual keywords---------')
#        print(manual_keywords)
#        print('--------extracted keywords---------')
#        print(keywords)
        stemmed_keywords = get_stemmed_keywords(keywords)
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
  total_fmeasure = round(2*total_precision*total_recall/(total_precision + total_recall), 5)
  print('total docs: ' + str(total_docs))
  print('total precision: ' + str(total_precision))
  print('total recall: ' + str(total_recall))
  print('total fmeasure: ' + str(total_fmeasure))

if __name__ == '__main__':
  main()
