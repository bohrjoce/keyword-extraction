#!/bin/python

import os
import nltk
import numpy as np
from nltk.stem.snowball import EnglishStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.api import StringTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
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

  # remove tokens in each sentence that aren't Noun, Verb or Adj
  sentences = remove_non_nva_sen(sentences)

  # do the transformation as follow:
  # replace each token in each sentence by their lemmatized->stemmed->tagged version. 
  # also need to keep a mapping back from stemmed-tagged version to un-stemmed but lemmatized
  mapping_back = {}
  # maybe use a flag here instead of True
  if True:
    sentences = lemmatize_sen(sentences)
    sentences, mapping_back = stem_sen(sentences)
  else:
    # if we don't want to use the transformation, just keep mapping_back be a mirror from each token to itself
    for sent in sentences:
      tok_list = word_tokenize(sent)
      for tok in tok_list:
        mapping_back[tok] = tok


  # tokenize words in sentences
  sentences = [word_tokenize(sent) for sent in sentences]

  # TODO: maybe put a flag in featrue extract for all of these options
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

  return all_tokens, data, mapping_back

# look at each sentences and remove word that aren't Noun, verb, or adj
def remove_non_nva_sen(list_sentences):
  res_list = []
  for sent in list_sentences:
    tmp_list = ''
    tok_list = word_tokenize(sent)
    pos_tok = nltk.pos_tag(tok_list)
    for tok,pos in pos_tok:
      if ('NN' not in pos) and ('JJ' not in pos) and ('VB' not in pos):
        continue
      tmp_list = tmp_list + tok + ' '
    res_list.append(tmp_list)
  return res_list


# a bad thing that can happen when lemmatize token is that we lose plurality information
def lemmatize_sen(list_sentences):
  lem = WordNetLemmatizer()
  res_list = []
  for sent in list_sentences:
    tmp_list = ''
    tok_list = word_tokenize(sent)
    for tok in tok_list:
      tok = lem.lemmatize(tok)
      tmp_list = tmp_list + tok + ' '
    res_list.append(tmp_list)
  return res_list

def stem_sen(list_sentences):
  stemmer = EnglishStemmer()
  # map back should be a dict with words, 
  # each word map to 3 version: noun, adj, verb,
  # and each version is a list of pair
  mapping_back = {}
  res_list = []
  stemmer = EnglishStemmer()

  # of course we want to return a list of sentences back as well
  for sent in list_sentences:
    tmp_list = ''
    tok_list = word_tokenize(sent)
    tok_pos = nltk.pos_tag(tok_list)
    for tok,pos in tok_pos:
      stem_tok = stemmer.stem(tok)
      if (stem_tok not in mapping_back):
        mapping_back[stem_tok] = {}
      if pos not in mapping_back[stem_tok]:
        mapping_back[stem_tok][pos] = {}
      
      # increase count
      if tok not in mapping_back[stem_tok][pos]:
        mapping_back[stem_tok][pos][tok] = 1
      else:
        mapping_back[stem_tok][pos][tok] += 1

      # assume NN, JJ, VB always in the front of pos
      pos = pos[:2]
      tmp_list = tmp_list + stem_tok + '-' + pos + ' '
    res_list.append(tmp_list)
  res_map = {}

  # do the second run through to find the most frequent - mapping
  for tok in mapping_back:
    for pos in mapping_back[tok]:
      tmp_tok = tok + '-' + pos
      # find the most frequently, unstemmed word correspond to the stemmer + tagged 
      most_freq = max(mapping_back[tok][pos], key = mapping_back[tok][pos].get)
      res_map[tmp_tok] = most_freq

  return res_list, res_map
# THE REMAINING FUNCTIONS MIGHT NOT BE USEFUL
# counts how many times each token is used as noun/verb/adj...
# not sure if this is really useful
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
      if 'VB' in pos:
        pos_tok[tok]['VB'] += 1
  return pos_dict

def pos_reweight(noun_weight, verb_weight, adj_weight, data, pos_dict):
  for tok in tokens:
    # find which position tok is used most
    max_pos = max(pos_dict, key=pos_dict.get)
    # readjust weight based on max_pos, but need to agree on the form of data
    # .. a few lines of code to finish this function! should be easy
