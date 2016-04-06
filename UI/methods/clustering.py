#!/bin/python

from sklearn import cluster
import numpy as np
import nltk
import os
import feature_extract

def kcluster(mapping_back, num_cluster, weight_array, one_hot_tokens, num_key = 10):
  # num_key is the number of keyword that we extract from a cluster
  # we can find the union of the extracted keys from each cluster

  # in case we have less vector than cluste number
  num_cluster = min(num_cluster, len(weight_array))
  k_clusters = cluster.k_means(weight_array, num_cluster)[0]
  union_array = []
  num_key = min(num_key, len(one_hot_tokens))
  for i,vec in enumerate(k_clusters):
    tmp = sorted(range(len(vec)), key=lambda i: vec[i])[-num_key:]
    union_array = list(set(tmp) | set(union_array))
  res = []
  for ind in union_array:
    res.append(mapping_back[one_hot_tokens[ind]])
  return res

def main():
  nltk.data.path.append('/home/jocelyn/usb/nltk_data')
  semeval_dir = 'data/maui-semeval2010-test/'
  filenames = sorted(os.listdir(semeval_dir))
  for filename in filenames:
    if filename[-3:] == 'key':
      continue
    f = open(semeval_dir + filename, 'r')
    print(filename)
    content = f.read()
    tokens, data, mapping_back = feature_extract.get_rakeweight_data(content)
    vec = kcluster(mapping_back, 5, data, tokens)
    for v in vec:
      print(v)
    break

if __name__ == '__main__':
  main()
