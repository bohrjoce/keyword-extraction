#!/bin/python

from sklearn import cluster
import numpy as np
import nltk
import random
import os
from feature_extract import get_rakeweight_data
from postprocess import get_keyphrases
from postprocess import get_keyphrase_weights
from collections import defaultdict

def kcluster(content, num_cluster = 6, num_key = 15, single = False, num_top = -1):
  # num_key is the number of keyword that we extract from a cluster
  # we can find the union of the extracted keys from each cluster

  one_hot_tokens, weight_array, mapping_back, postprocess_sentences, C = get_rakeweight_data(content)

  # in case we have less vector than cluste number
  num_cluster = min(num_cluster, len(weight_array))
  token_weights = defaultdict(float)
  keyword_weights = defaultdict(float)
  k_clusters = cluster.k_means(weight_array, num_cluster)[0]
  union_array = []
  keywords = []
  num_key = min(num_key, len(one_hot_tokens))
  for i,vec in enumerate(k_clusters):
    tmp = sorted(range(len(vec)), key=lambda i: vec[i])[-num_key:]
    union_array = list(set(tmp) | set(union_array))
    for ind in tmp:
      token = one_hot_tokens[ind]
      degree = sum(C[token].values())
      freq = C[token][token]
      # currently degree. update to different weight scheme if needed
      token_weights[token] += float(degree)
  for ind in union_array:
    token = one_hot_tokens[ind]
    keyword = mapping_back[token]
    keywords.append(keyword.encode('ascii'))
    # for all tokens that map to keyword
    keyword_weights[keyword] += token_weights[token]

  keywords = list(set(keywords))
  if single:
    keywords = set(keywords)
    if (num_top < 0):
      num_top = len(keywords)
    return random.sample(keywords, min(len(keywords), num_top))
  # get keyphrases
  keyphrases,keyphrase_freq = get_keyphrases(keywords, postprocess_sentences)
  # keyphrases_weights = sum keyword_weights[word] / total_words
  # for all words in keywords
  keyphrases_weights = get_keyphrase_weights(keyphrases, keyword_weights, keyphrase_freq)
  keyword_weights.update(keyphrases_weights)
  if num_top < 0:
    num_top = len(keyword_weights)/3
  top_keywords = sorted(keyword_weights, key=keyword_weights.get, reverse=True)[:min(num_top, len(keyword_weights))]
#  for keyword in top_keywords:
#    print(keyword + ' '*(40-len(keyword)) + str(keyword_weights[keyword]))
  return top_keywords

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
    tokens, data = get_rakeweight_data(content)
    vec = kcluster(5, data, tokens)
    for v in vec:
      print(v)
    break

if __name__ == '__main__':
  main()
