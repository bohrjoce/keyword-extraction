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

damp = .85
co_matrix = {} # dictionary of dictionaries representing co-occurence matrix
conv = .01 # value to check if weights converge

def addToMatrix(word1, word2):
	if word1 == word2:
		return

	if word1 in co_matrix:
		if word2 in co_matrix[word1]:
			co_matrix[word1][word2] += 1
		else:
			co_matrix[word1][word2] = 1
	else: 
		co_matrix[word1] = {}
		co_matrix[word1][word2] = 1


def get_rakeweight_data(doc):

  # replace non-ascii and newline characters with space
  content = ''.join([i if ord(i) < 128 and i != '\n' else ' ' for i in doc])

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
  # stemmer = EnglishStemmer()
  # sentences = [list(stemmer.stem(t) for t in sent) for sent in sentences]

  # get list of all tokens
  all_tokens = [t for sent in sentences for t in sent]

  # remove duplicates and sort
  all_tokens = sorted(list(set(all_tokens)))

  # ---- replace rakeweight with differenct weighting scheme ----

  # construct co-occurrence matrix
  for sent in sentences:
  	for word1 in sent:
  		for word2 in sent:
  			if word1 != word2:
  				addToMatrix(word1, word2)

  return all_tokens



# takes indexes of tokens and stores them in vertices dict
# appends 
# def addEdge(token1, token2):
# 	if token1 not in vertices:
# 		vertices[token1] = []

# 	if token2 not in vertices[token1]:
# 		vertices[token1].append(token2)


# given a co-occurence matrix and list of tokens,
# intitializes vertex weights given RAKE algorithm
def getTokenWeight(vertex_scores, tokens):
	for t1 in tokens:
		degree = 0
		freq = 0
		if t1 in co_matrix:
			for t2 in co_matrix[t1]:
				# addEdge(t1, t2)
				degree += co_matrix[t1][t2]
				freq += 1
			vertex_scores[t1] = float(degree) / freq
	

# uodates a vertice based on the TextRank algorithm with weighted edges and vertices
# W(Vi) = (1-d) + d * sum(weight of edge from i to j for all adjacent j/ sum(weight of edges from j to k for all adjacent k)) * W(j) 
# returns true if token node is converging, false otherwise
def updateNode(vertex_scores, temp_scores, token):
	total = 0
	if token not in co_matrix:
		return True

	for edge in co_matrix[token]:
		edge_weight = co_matrix[token][edge]
		adj_weight = vertex_scores[edge]
		count = 0
		for e in co_matrix[edge]:
			count += co_matrix[edge][e]
		total += (adj_weight * edge_weight) / count
		# total += count * adj_weight / (edge_weight)

	total *= damp
	total += 1 - damp


	if (abs(total - vertex_scores[token]) > conv):
		# print token + " NOT CONVERGIN"
		temp_scores[token] = total
		return False


	temp_scores[token] = total
	return True

def printScores(vertex_scores):
	for v in vertex_scores:
		print "token: " + v + " score: " + str(vertex_scores[v])


def main():
	text = "The presidential race is coming up. Who do you think will win? It will be a close presidential race."
	
	tokens = get_rakeweight_data(text)
	vertex_scores = {} #key: token, #value: current score
	getTokenWeight(vertex_scores, tokens)

	# iterates through TextRank algorithm until it converges
	has_converged = False
	counter = 0
	while not has_converged:
		has_converged = True
		# print "ROUND " + str(counter)
		# printScores(vertex_scores)
		temp_scores = dict(vertex_scores)
		for t in tokens:
			c = updateNode(vertex_scores, temp_scores, t)
			if not c:
				has_converged = False
		# print "converged? " + str(has_converged)
		vertex_scores = dict(temp_scores)

		if counter == 30:
			break
		counter += 1

	dic = sorted(vertex_scores.items(), key = operator.itemgetter(1), reverse = True)
	num_words = len(vertex_scores)
	keywords = []
	# according to TextRank, # of keywords should be size of set divided by 3
	count = 1
	for i in dic: 
		if (count > num_words / 3):
			break
		keywords.append(i[0])
		count += 1

	print keywords	

if __name__ == '__main__':
	main()














