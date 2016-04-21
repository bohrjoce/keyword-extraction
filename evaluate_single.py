#!/bin/python

from sklearn import cluster
import numpy as np
import nltk
import os
from nltk.corpus import stopwords
from clustering import kcluster
from svd import svd
import feature_extract
import raketr
import re
import sys

nltk.data.path.append('/home/jocelyn/usb/nltk_data')

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
        # list of list of keywords by line
        manual_keywords = [line.split() for line in key_lines]
        # flatten list
        manual_keywords = [word for line in manual_keywords for word in line]
        manual_keywords = list(set(manual_keywords))
        manual_keywords = [t for t in manual_keywords if ( (len(t) > 1) and (t.lower()not in stopwords.words('english')) )]

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
          keywords = svd(content, 1, True)
        elif method == 'raketr':
          keywords = raketr.main(content, True)
        elif method == 'cluster':
          keywords = kcluster(content, 6, 10, True)
        else:
          print('methods accepted: svd raketr cluster')
          exit(0)
        keywords = list(set(keywords))
        keywords = [word.encode('ascii') for word in keywords]
#        print('--------manual keywords---------')
#        print(manual_keywords)
        print(keywords)
        print('-'*100)
        for keyword in keywords:
          if keyword in set(manual_keywords):
            correct += 1
        if len(manual_keywords) == 0:
          print(filename)
          print(last_key_file)
          print('^^^^ issue with this file ^^^^')
          exit(0)
        total_precision += correct/float(len(keywords))
        total_recall += correct/float(len(manual_keywords))

  total_precision /= total_docs
  total_recall /= total_docs
  total_fmeasure = round(2*total_precision*total_recall/(total_precision + total_recall), 5)
  print('total docs: ' + str(total_docs))
  print('total precision: ' + str(total_precision))
  print('total recall: ' + str(total_recall))
  print('total fmeasure: ' + str(total_fmeasure))

if __name__ == '__main__':
  main()
