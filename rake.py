import nltk
from nltk.stem.snowball import EnglishStemmer
from nltk.tokenize.api import StringTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer
import nltk.data
import numpy as np
import copy
import operator
import re

damp = .85
conv = .05 # value to check if weights converge

def get_rakeweight_data(text):
	text = text.lower()
	# replace non-ascii and newline characters with space
	content = ''.join([i if ord(i) < 128 and i != '\n' else ' ' for i in text])

	# return pretrained sentence tokenizer
	sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

	# segment content into sentences
	sentences = sent_detector.tokenize(content)

	# tokenize words in sentences
	sentences = [word_tokenize(sent) for sent in sentences]

	# # remove stopwords
	# sentences = [list(t for t in sent if ( (len(t) > 1) and (t.lower() not in stopwords.words('english')) )) for sent in sentences]

	# get list of all tokens
	all_tokens = [t for sent in sentences for t in sent]

	# remove duplicates and sort
	all_tokens = sorted(list(set(all_tokens)))
	# print all_tokens

	return all_tokens, sentences

# a phrase are tokens in the same sentence that are grouped together by removing stopwords
def initializePhrases(sentences, tokens):
	keyphrases = []
	for sent in sentences:
		phrase = []
		for token in sent:
			if len(token) > 1 and token not in stopwords.words('english') and token.isalnum():
				pattern = re.compile('[\W_]+')
				token = pattern.sub('', token)
				phrase.append(token)
			else:
				if len(phrase) > 0:
					keyphrases.append(phrase)
				phrase = []
	return keyphrases


def addToMatrix(comatrix, comatrix_count, word1, word2):
	if word1 == word2:
		# if word1 in comatrix:
		# 	if word1 not in comatrix[word1]:
		# 		comatrix[word1][word1] = 1
		# else:
		# 	comatrix[word1] = {}
		# 	comatrix[word1][word1] = 1
		return
	else:

		if word1 in comatrix:
			# comatrix_count[word1] += 1
			if word2 in comatrix[word1]:
				comatrix[word1][word2] += 1
			else:
				comatrix[word1][word2] = 1
		else:
			# comatrix_count[word1] = 1
			comatrix[word1] = {}
			comatrix[word1][word2] = 1

		if word2 in comatrix:
			# comatrix_count[word2] += 1
			if word1 in comatrix[word2]:
				comatrix[word2][word1] += 1
			else:
				comatrix[word2][word1] = 1
		else:
			# comatrix_count[word2] = 1
			comatrix[word2] = {}
			comatrix[word2][word1] = 1



def initializeCoMatrix(keyphrases):
	comatrix = {}
	comatrix_count = {}


	# print keyphrases
	for phrase in keyphrases:
		# print phrase
		if len(phrase) > 1:
			# print phrase
			for i in range(1, len(phrase)):
				addToMatrix(comatrix, comatrix_count, phrase[0], phrase[i])
		else:
			addToMatrix(comatrix, comatrix_count, phrase[0], phrase[0])

	# print comatrix
	# print comatrix_count
	return comatrix, comatrix_count

def getKeywordWeights(comatrix, comatrix_count):
	keywords = {} #key: token, value: weight


	for t1 in comatrix:
		freq = 0
		deg = 0
		for t2 in comatrix[t1]:
			freq += 1
			deg += comatrix[t1][t2]
		keywords[t1] = float(deg) / freq
	return keywords

def getKeyphraseWeights(keyphrases, keywords):
	keyphrase_weights = {}
	for phrase in keyphrases:
		key = ""
		weight = 0
		length = 0
		for token in phrase:
			if key == "":
				key += token
			else:
				key += " " + token 
			if token in keywords:
				weight += keywords[token]
			length += 1

		keyphrase_weights[key] = weight / length


	return keyphrase_weights

def printScores(vertex_scores):
	for v in vertex_scores:
		print "token: " + v + " score: " + str(vertex_scores[v])

# uodates a vertice based on the TextRank algorithm with weighted edges and vertices
# W(Vi) = (1-d) + d * sum(weight of edge from i to j for all adjacent j/ sum(weight of edges from j to k for all adjacent k)) * W(j) 
# returns true if token node is converging, false otherwise
def updateNode(vertex_scores, temp_scores, token, comatrix, comatrix_count):
	total = 0
	if token not in comatrix:
		return True

	for edge in comatrix[token]:
		edge_weight = comatrix[token][edge]
		adj_weight = 0
		if edge in vertex_scores:
			adj_weight = vertex_scores[edge]
		# count = comatrix_count[edge]
		count = len(comatrix[edge])
		total += (adj_weight * edge_weight) / float(count)

	total *= damp
	total += 1 - damp


	if (abs(total - vertex_scores[token]) > conv):
		temp_scores[token] = total
		return False


	temp_scores[token] = total
	return True


def reweightTR(keyword_weights, comatrix, comatrix_count, tokens):

	has_converged = False
	counter = 0
	while not has_converged:
		if counter == 30:
			break
		has_converged = True
		print "ROUND " + str(counter)
		temp_scores = dict(keyword_weights)
		for token in keyword_weights:
			# if token is not None:
			c = updateNode(keyword_weights, temp_scores, token, comatrix, comatrix_count)
			if not c:
				has_converged = False
		keyword_weights = dict(temp_scores)
		# printScores(keyword_weights)

		if counter == 30:
			break
		counter += 1

	return keyword_weights


def main(text):
	tokens, sentences = get_rakeweight_data(text)
	keyphrases = initializePhrases(sentences, tokens)
	comatrix, comatrix_count = initializeCoMatrix(keyphrases)
	keyword_weights = getKeywordWeights(comatrix, comatrix_count)

	keyword_weights = reweightTR(keyword_weights, comatrix, comatrix_count, tokens) # THIS IS WHERE RAKE GETS REWEIGHTED
																					# REMOVE TO GET RAKE VALUES
	keyphrase_weights = getKeyphraseWeights(keyphrases, keyword_weights)
	sorted_x = sorted(keyphrase_weights.items(), key=operator.itemgetter(1), reverse=True)
	keywords = []

	counter = 0
	for pair in sorted_x:
		if counter > 25:
			break
		counter += 1
		keywords.append(pair[0])
		# if pair[1] > 0:
		# 	print "KEYWORD: " + pair[0] + " WEIGHT: " + str(pair[1])

	# print keywords
	return keywords

if __name__ == '__main__':
	text = "Harold ate an apple. Harold did not realize it was a magic apple. If he had known it was magic, he would have ate the whole apple."
	text = '''Compatibility of systems of linear constraints over the set of natural numbers.
Criteria of compatibility of a system of linear Diophantine equations, strict inequations,
and nonstrict inequations are considered. Upper bounds for components of a minimal set
of solutions and algorithms of construction of minimal generating sets of solutions for all
types of systems are given. These criteria and the corresponding algorithms for
constructing a minimal supporting set of solutions can be used in solving all the
considered types of systems and systems of mixed types.'''
	with open("data/maui-semeval2010-test/C-1.txt") as test_f:
		text = test_f.read();
	# print text
	main(text)







