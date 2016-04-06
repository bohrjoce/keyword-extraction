
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

#interfaces:
#svd function, out put the principal component
def svd_mat(mat, vocalst):

	U,s,V = np.linalg.svd(mat, full_matrices=True)
	idx = 0;
	for i in range(1, len(s)):
		if (s[i] < s[0]*0.5): 
			idx = i
			break;
	S = np.diag(s[0:idx])
	up = U[:, 0:idx]
	vp = V[0:idx, :] 
	recover = np.dot(up, np.dot(S, vp))
	flst = np.sum(recover,axis=1)
	sortidx = list(np.argsort(flst))
	sortidx.reverse()
	rlst = []
	for i in range(0,7):
		rlst.append(vocalst[sortidx[i]])
	# print(rlst)
	return rlst
	'''
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

	for i in range(0, 20):
		print s[i]
	'''

#process the file get a list of sentences
def process_file(text):

	#f = open(filename)
	#s = f.read()
	s = text
	s1 = s.replace('\n', ' ')
	s1 = re.sub("\<.+\>", "", s1)
	lst = tokenizer.tokenize(s1)
	return lst


def make_mat(lst):
	outlst = []
	vocabulary = dict()
	N = len(lst)
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

	vocalst = list(vocabulary.keys())
	dim = len(vocalst)
	mat = np.empty((dim, 0), float)
	for solst in outlst:
		v = np.zeros(dim)
		v.shape = (dim, 1)
		counter = Counter(solst)
		for wd in counter.keys():
			idx = vocalst.index(wd)
			#score = math.log(N*1.0/vocabulary[wd]) * counter[wd] 	#tfidf -- will give the author as high rank
			score = counter[wd]	#naive weighting		#will give the frequent term as high rank
			v[idx] = score
		mat = np.concatenate((mat, v), axis = 1)
	return mat, vocalst

def svd(text):
	lst = process_file(text)
	mat, vocalst = make_mat(lst)
	rlst = svd_mat(mat, vocalst)
	return rlst

# def getSVD(text):
# 	mat, vocalst = make_mat(text)
# 	rlst = svd_mat(mat, vocalst)
# 	return rlst


