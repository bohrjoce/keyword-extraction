
import nltk.data
#tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
from nltk.stem import WordNetLemmatizer
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re
from nltk.stem.porter import *
import math
import numpy as np
from feature_extract import get_rakeweight_data
from postprocess import get_keyphrases
from collections import defaultdict
from postprocess import get_keyphrases
from postprocess import get_keyphrase_weights
#read document

#interfaces:
#svd function, out put the principal component
#performe svd algorithm and extract num_of_comp principal components
def svd_mat(mat, vocalst, map_back, C, num_of_comp = 3):

  mat = np.transpose(mat)
  U,s,V = np.linalg.svd(mat, full_matrices=True)
  keyword_weights = defaultdict(float)
  idx = num_of_comp;
  S = np.diag(s[0:idx])
  up = U[:, 0:idx]
  vp = V[0:idx, :]
  recover = np.dot(up, np.dot(S, vp))
  flst = np.sum(recover,axis=1)
  sortidx = list(np.argsort(flst))
  sortidx.reverse()
  rlst = []
  for i in range(0,min(25, len(sortidx))):
    token = vocalst[sortidx[i]]
    keyword = map_back[token]
    freq = C[token][token]
    degree = sum(C[token].values())
    keyword_weights[keyword] += float(degree)
    rlst.append(keyword)
  return rlst, keyword_weights

#process the file get a list of sentences
def process_file(docstr):
#input is the string of the document
  s = docstr
  return get_rakeweight_data(s)


def make_mat(lst):
#construct matrix over a list of sentences
#which is not used in evaluation, used for developing purpose
  outlst = []
  vocabulary = dict()
  N = len(lst)
  tokenizer = RegexpTokenizer(r'\w+')
  for st in lst:
    suboutlst = []
    st1 = st.lower();
    sublst = tokenizer.tokenize(st1)
    for word in sublst:  #normalize and remove stopwords

      w1 = str(word)
      if len(w1) == 1:
        continue
      if w1.isdigit():
        continue
      if (w1 in stopwords.words('english')):continue
      if (w1 not in suboutlst): vocabulary[w1] = vocabulary.get(w1, 0) + 1
      suboutlst.append(w1)
    outlst.append(suboutlst)

  vocalst = list(vocabulary.keys())
  dim = len(vocalst)
  mat = np.empty((dim, 0), float)
  for solst in outlst:
    v = np.zeros(dim)
    v.shape = (dim, 1)
    counter = Counter(solst)
    for wd in counter.keys():
      idx = vocalst.index(wd)
      score = math.log(N*1.0/vocabulary[wd]) * counter[wd]   #tfidf -- will give the author as high rank
      #score = counter[wd]  #naive weighting    #will give the frequent term as high rank
      v[idx] = score
    mat = np.concatenate((mat, v), axis = 1)
  return mat, vocalst

def svd(filename, num_of_comp = 3, single = False):
#interface with the evaluation script
#filename is the input document string as in process_file function
#get document matrix and associated data structures
  tokens, data, map_back, postprocess_sentences, C = process_file(filename)
  #perform svd
  keywords, keyword_weights = svd_mat(data, tokens, map_back, C, num_of_comp)
  if single:
    return keywords
  # combine into multiple keywords
  keyphrases,keyphrase_freq = get_keyphrases(keywords, postprocess_sentences)
  keyphrase_weights = get_keyphrase_weights(keyphrases, keyword_weights, keyphrase_freq)
  keyword_weights.update(keyphrase_weights)
  top_keywords = sorted(keyword_weights, key=keyword_weights.get, reverse=True)[:min(15, len(keyword_weights))]
  return top_keywords
