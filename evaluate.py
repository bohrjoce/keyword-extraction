#!/bin/python

from sklearn import cluster
import numpy as np
import nltk
import os
from feature_extract import get_rakeweight_data
from nltk.corpus import stopwords
from clustering import kcluster
#from rake_tr import *
from svd import svd

nltk.data.path.append('/home/jocelyn/usb/nltk_data')
def main():

  semeval_dir = 'data/maui-semeval2010-test/'
  filenames = sorted(os.listdir(semeval_dir))
  manual_keywords = []
  total_precision = 0
  total_recall = 0

  for filename in filenames:

    if filename[-3:] == 'key':
      with open(semeval_dir + filename, 'r') as f:
        last_key_file = filename
        key_lines = f.read().splitlines()
        # list of list of keywords by line
        manual_keywords = [line.split() for line in key_lines]
        # flatten list
        manual_keywords = [word for line in manual_keywords for word in line]
        manual_keywords = list(set(manual_keywords))
        manual_keywords = [t for t in manual_keywords if ( (len(t) > 1) and (t.lower()not in stopwords.words('english')) )]

    elif filename[-3:] == 'txt':
      print(filename)
      with open(semeval_dir + filename, 'r') as f:
        correct = 0
        f = open(semeval_dir + filename, 'r')
        content = f.read()
        tokens, data, mapping_back = get_rakeweight_data(content)
#        keywords = svd(content)
#        keywords = rake_tr.main()
        keywords = kcluster(mapping_back, 6, data, tokens)
        keywords = list(set(keywords))
        keywords = [word.encode('ascii') for word in keywords]
        print('--------manual keywords---------')
        print(manual_keywords)
        print('--------extracted keywords---------')
        print(keywords)
        for keyword in keywords:
          if keyword in set(manual_keywords):
            correct += 1
        if len(manual_keywords) == 0:
          print(filename)
          print(last_key_file)
          exit(0)
        total_precision += correct/float(len(keywords))
        total_recall += correct/float(len(manual_keywords))

  total_docs = len(filenames)/2
  total_precision /= total_docs
  total_recall /= total_docs
  total_fmeasure = round(2*total_precision*total_recall/(total_precision + total_recall), 2)
  print('total docs: ' + str(total_docs))
  print('total precision: ' + str(total_precision))
  print('total recall: ' + str(total_recall))
  print('total fmeasure: ' + str(total_fmeasure))

if __name__ == '__main__':
  main()
