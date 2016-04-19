#!/bin/python

from collections import defaultdict

def get_keyphrases(keywords, sentences):

  is_keyword = [list(True if word in keywords else False for word in sent) for sent in sentences]
  keyphrases = []
  keyphrase = []
  run = 0
  for i,sent in enumerate(is_keyword):
    for j,val in enumerate(sent):
      if val is True:
        word = sentences[i][j]
        if word not in keyphrase:
          keyphrase.append(word)
          run += 1
        elif word in keyphrase:
          if run > 1:
            keyphrases.append(' '.join(keyphrase))
          run = 1
          keyphrase = [word]

      if val is False:
        if run > 1:
          keyphrases.append(' '.join(keyphrase))
        run = 0
        keyphrase = []
    # reset run at end of sentence
    run = 0
    keyphrase = []
  keyphrases = list(set(keyphrases))
  return keyphrases

def get_keyphrase_weights(keyphrases, keyword_weights):

  # keyphrases_weights = sum keyword_weights[word] / total_words
  # for all words in keywords, add optional bonus for long keyword
  keyphrase_weights = defaultdict(float)
  keyphrase_weight = 0
  for keyphrase in keyphrases:
    keyphrase_list = keyphrase.split()
    for word in keyphrase_list:
      keyphrase_weight += keyword_weights[word]
    keyphrase_weight /= len(keyphrase_list)
    # bonus for long keywords
#    keyphrase_weight += len(keyphrase_list)
    keyphrase_weights[keyphrase] = keyphrase_weight
    keyphrase_weight = 0

  return keyphrase_weights

