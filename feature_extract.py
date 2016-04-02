#!/bin/python

import os
import nltk
import numpy as np
from nltk.stem.snowball import EnglishStemmer
from nltk.tokenize.api import StringTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer
from collections import defaultdict
import nltk.data
import copy

def get_rakeweight_data(doc):

  content = doc.lower()

  # replace non-ascii and newline characters with space
  content = ''.join([i if ord(i) < 128 and i != '\n' else ' ' for i in content])

  # return pretrained sentence tokenizer
  sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

  # segment content into sentences
  sentences = sent_detector.tokenize(content)

  # tokenize words in sentences
  sentences = [word_tokenize(sent) for sent in sentences]

  # remove stopwords
  sentences = [list(t for t in sent if ( (len(t) > 1) and (t.lower()
      not in stopwords.words('english')) )) for sent in sentences]

  # stem tokens
  # dont stem for now
#  stemmer = EnglishStemmer()
#  sentences = [list(stemmer.stem(t) for t in sent) for sent in sentences]

  # get list of all tokens
  all_tokens = [t for sent in sentences for t in sent]

  # remove duplicates and sort
  all_tokens = sorted(list(set(all_tokens)))

  # ---- replace rakeweight with differenct weighting scheme ----

  # construct co-occurrence matrix
  C = dict.fromkeys(all_tokens, 0)
  for t in all_tokens:
    C[t] = defaultdict(int)
  prev = 0
  for sent in sentences:
    for word1 in sorted(sent):
      for word2 in sorted(sent):
        C[word1][word2] += 1

  # training data matrix
  data = np.zeros((len(sentences), len(all_tokens)))
  vec = dict.fromkeys(all_tokens, 0)

  total_freq = dict.fromkeys(all_tokens, 0)
  # compute degree/freq for each sentence, add to data
  for i,sent in enumerate(sentences):
    cur_vec = copy.deepcopy(vec)
    for word in sent:
      degree = sum(C[word].values())
      freq = C[word][word]
      # various methods to weight. freq and degree work pretty well. degree/freq...not so much
      cur_vec[word] = float(degree)/float(freq)
      cur_vec[word] = float(freq)
#      cur_vec[word] = float(degree)

    arr = []
    for key in sorted(cur_vec):
      arr.append(cur_vec[key])
      total_freq[key] += cur_vec[key]
    data[i,:] = np.array(arr)

  for key in sorted(total_freq, key=total_freq.get, reverse=True):
    print(key + ' ' + str(total_freq[key]))
  print('--------')
  return all_tokens, data

# counts how many times each token is used as noun/verb/adj...
def pos_count(list_sentences, tokens):
  pos_dict = {}
  for tok in tokens:
    pos_dict[tok] = {'NN':0, 'VB':0, 'JJ':0}
  for sent in list_sentences:
    tokenized = word_tokenize(sent)
    pos_tok = nltk.pos_tag(tokenized)
    for tok, pos in pos_tok:
      if 'NN' in pos:
        pos_tok[tok]['NN'] +=1
      if 'JJ' in pos:
        pos_tok[tok]['JJ'] += 1
      if 'VRB' in pos:
        pos_tok[tok]['VRB'] += 1
  return pos_dict

def pos_reweight(noun_weight, verb_weight, adj_weight, data, pos_dict):
  for tok in tokens:
    # find which position tok is used most
    max_pos = max(pos_dict, key=pos_dict.get)
    # readjust weight based on max_pos, but need to agree on the form of data
    # .. a few lines of code to finish this function! should be easy
