#!/bin/python

from sklearn import cluster
import numpy as np
import nltk
import os
from feature_extract import get_rakeweight_data
from clustering import kcluster
from rake_tr import *

def main():

  nltk.data.path.append('/home/jocelyn/usb/nltk_data')
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

    elif filename[-3:] == 'txt':
      with open(semeval_dir + filename, 'r') as f:
        correct = 0
        f = open(semeval_dir + filename, 'r')
        content = f.read()
        tokens, data = get_rakeweight_data(content)
        # keywords = svd(...)
        # keywords = kcluster(5, data, tokens)
        keywords = rake_tr.main()
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
  print('total docs: ' + str(total_docs))
  print('total precision: ' + str(total_precision))
  print('total recall: ' + str(total_recall))

if __name__ == '__main__':
  main()
