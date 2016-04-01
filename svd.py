
import nltk.data
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
from nltk.stem import WordNetLemmatizer
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re
from nltk.stem.porter import *
import math
import numpy as np
#read document

f = open('C-1.txt')
s = f.read()
s1 = s.replace('\n', ' ')
s1 = re.sub("\<.+\>", "", s1)
vocabulary = dict()  	#count df

lst = tokenizer.tokenize(s1)
N = len(lst) 	#number of documents
outlst = []
stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')
for st in lst:
	suboutlst = []
	st1 = st.lower();
	sublst = tokenizer.tokenize(st1)
	for word in sublst:	#normalize and remove stopwords

		w1 = str(word)
		if (w1 in stopwords.words('english')):continue
		if (w1 not in suboutlst): vocabulary[w1] = vocabulary.get(w1, 0) + 1
		suboutlst.append(w1)
	outlst.append(suboutlst)

#the above is the first pass, construct vocabulary
#construct matrix
vocalst = list(vocabulary.keys())
dim = len(vocalst)
mat = np.empty((dim, 0), float)
for solst in outlst:
	v = np.zeros(dim)
	v.shape = (dim, 1)
	counter = Counter(solst)
	for wd in counter.keys():
		idx = vocalst.index(wd)
		score = math.log(N*1.0/vocabulary[wd]) * counter[wd] 	#tfidf -- will give the author as high rank
		#score = counter[wd]	#naive weighting		#will give the frequent term as high rank
		v[idx] = score
	mat = np.concatenate((mat, v), axis = 1)

#do svd
U,s,V = np.linalg.svd(mat, full_matrices=True)
u1 = U[:, 0]
u1x = np.absolute(u1)
sortidx = list(np.argsort(u1x))
sortidx.reverse()
print ("first level: with score: " + str(s[0]))
for i in range(0, 10):
	print vocalst[sortidx[i]]



u2 = U[:,1]
sortidx2 = list(np.argsort(u2))
sortidx2.reverse()
print ("second level: with socre: " + str(s[1]))
for i in range(0, 10):
	print vocalst[sortidx2[i]]


