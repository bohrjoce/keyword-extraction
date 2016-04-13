#!/bin/python

from nltk.stem.snowball import EnglishStemmer

def postprocess(keywords, stemmed_sentences):

  stemmer = EnglishStemmer()
  stemmed_keywords = [stemmer.stem(keyword) for keyword in keywords]
  stem_keyword_mapping = dict(zip(stemmed_keywords, keywords))
  is_keyword = [list(True if word in stemmed_keywords else False for word in sent) for sent in stemmed_sentences]
  keyphrases = []
  keyphrase = []
  run = 0
  for i,sent in enumerate(is_keyword):
    for j,val in enumerate(sent):
      if val is True:
        word = stem_keyword_mapping[stemmed_sentences[i][j]]
        if word not in keyphrase:
          keyphrase.append(word)
          run += 1
        elif word in keyphrase and run > 1:
          keyphrases.append(" ".join(keyphrase))
          run = 1
          keyphrase = [word]

      if val is False:
        if run > 1:
          keyphrases.append(" ".join(keyphrase))
        run = 0
        keyphrase = []
    # reset run at end of sentence
    run = 0
    keyphrase = []
  keyphrases = list(set(keyphrases))
  keywords.extend(keyphrases)
  return keywords
