import nltk
from nltk.stem.snowball import EnglishStemmer
from nltk.tokenize.api import StringTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer
from feature_extract import stem_sen, remove_non_nva_sen
import nltk.data
import numpy as np
import copy
import operator
from postprocess import get_keyphrases
from postprocess import get_keyphrase_weights
from collections import defaultdict

damp = .85
co_matrix = {} # dictionary of dictionaries representing co-occurence matrix
co_matrix_count = {}
conv = .05 # value to check if weights converge

def addToMatrix(word1, word2):
	if word1 == word2:
		return

	if word1 in co_matrix:
		co_matrix_count[word1] += 1
		if word2 in co_matrix[word1]:
			co_matrix[word1][word2] += 1
		else:
			co_matrix[word1][word2] = 1
	else:
		co_matrix[word1] = {}
		co_matrix[word1][word2] = 1
		co_matrix_count[word1] = 1


def get_rakeweight_data(doc):

	global co_matrix
	global co_matrix_count
	co_matrix = {}
	co_matrix_count = {}


	# replace non-ascii and newline characters with space
	content = doc.lower()
	content = ''.join([i if ord(i) < 128 and i != '\n' else ' ' for i in content])

	sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
	sentences = sent_detector.tokenize(content)

  # split sentences, then split sentences into tokens
  # THIS IS FOR COMBINING SINGLE KEYWORDS LATER
	postprocess_sentences = [word_tokenize(sent) for sent in sentences]
	postprocess_sentences = [list(word.encode('ascii') for word in sent) for sent in postprocess_sentences]
#  stemmed_tokenized_content = [word for word in stemmed_tokenized_content if re.match('^[\w-]+$', word) is not None]
	postprocess_sentences = [list(word for word in sent if word.isalpha()) for sent in postprocess_sentences]

	sentences = remove_non_nva_sen(sentences)

  # do the transformation as follow:
  # replace each token in each sentence by their lemmatized->stemmed->tagged version.
  # also need to keep a mapping back from stemmed-tagged version to un-stemmed but lemmatized
	mapping_back = {}
	sentences, all_tokens, mapping_back = stem_sen(sentences)

	# get list of all tokens
	all_tokens = [t for sent in sentences for t in sent]

	# remove duplicates and sort
	all_tokens = sorted(list(set(all_tokens)))

	# remove stopwords
	sentences = [list(t for t in sent if ( (len(t) > 1) and (t.lower()not in stopwords.words('english')) )) for sent in sentences]

	# construct co-occurrence matrix
	for sent in sentences:
		for word1 in sent:
			for word2 in sent:
				if word1 != word2:
					addToMatrix(word1, word2)

	return all_tokens, mapping_back, postprocess_sentences

# given a co-occurence matrix and list of tokens,
# intitializes vertex weights given RAKE algorithm
def getTokenWeight(vertex_scores, tokens):
	for t1 in tokens:
		degree = 0
		freq = 0
		if t1 in co_matrix:
			for t2 in co_matrix[t1]:
				degree += co_matrix[t1][t2]
				freq += 1
			vertex_scores[t1] = float(degree) / float(freq)


# uodates a vertice based on the TextRank algorithm with weighted edges and vertices
# W(Vi) = (1-d) + d * sum(weight of edge from i to j for all adjacent j/ sum(weight of edges from j to k for all adjacent k)) * W(j)
# returns true if token node is converging, false otherwise
def updateNode(vertex_scores, temp_scores, token):
	total = 0
	if token not in co_matrix:
		return True

	for edge in co_matrix[token]:
		edge_weight = co_matrix[token][edge]
		adj_weight = 0
		if edge in vertex_scores:
			adj_weight = vertex_scores[edge]
		count = 0
		count = co_matrix_count[edge]
		total += (adj_weight * edge_weight) / float(count)

	total *= damp
	total += 1 - damp


	if (abs(total - vertex_scores[token]) > conv):
		temp_scores[token] = total
		return False


	temp_scores[token] = total
	return True

def printScores(vertex_scores):
	for v in vertex_scores:
		print "token: " + v + " score: " + str(vertex_scores[v])

def getRake(text):
	text = text.lower()

	tokens = get_rakeweight_data(text)
	vertex_scores = {} #key: token, #value: current score
	getTokenWeight(vertex_scores, tokens)
	# printScores(vertex_scores)

	dic = sorted(vertex_scores.items(), key = operator.itemgetter(1), reverse = True)
	num_words = len(vertex_scores)
	rake_keywords = []
	# according to TextRank, # of keywords should be size of set divided by 3
	count = 1
	for i in dic:
		if (count > (num_words / 3) + 1):
			break
		rake_keywords.append(i[0])
		count += 1

	print rake_keywords
	return rake_keywords

def main(text):
	text = text.lower()

	tokens, mapping_back, stemmed_sentences = get_rakeweight_data(text)
	vertex_scores = {} #key: token, #value: current score
	getTokenWeight(vertex_scores, tokens)
	# printScores(vertex_scores)

	# iterates through TextRank algorithm until it converges
	has_converged = False
	counter = 0
	while not has_converged:
		has_converged = True
		# printScores(vertex_scores)
		temp_scores = dict(vertex_scores)
		for t in tokens:
			# print "token: " + str(t)
			if t is not None:
				c = updateNode(vertex_scores, temp_scores, t)
				if not c:
					has_converged = False
		vertex_scores = dict(temp_scores)

		if counter == 30:
			break
		counter += 1

	# printScores(vertex_scores)
	num_words = len(vertex_scores)
	keywords = []
	tok_max = sorted(vertex_scores.iteritems(), key=lambda x:-x[1])[:24]
	keyword_weights = defaultdict(float)
	for tok, val in tok_max:
		keyword = mapping_back[tok]
		keywords.append(keyword)
		keyword_weights[keyword] += val

  # construct multiple keywords
	keyphrases,keyphrase_freq = get_keyphrases(keywords, stemmed_sentences)
	keyphrase_weights = get_keyphrase_weights(keyphrases, keyword_weights, keyphrase_freq)
	keyword_weights.update(keyphrase_weights)
	top_keywords = sorted(keyword_weights, key=keyword_weights.get, reverse=True)[:15]

#	print keywords
	return top_keywords

if __name__ == '__main__':
	text = "The presidential race is coming soon. Who do you think will win? It will be a close presidential race."
	text = "The presidential race is coming soon. It will be a close presidential race."
	text = '''Efficient discovery of grid services is essential for the success of
grid computing. The standardization of grids based on web
services has resulted in the need for scalable web service
discovery mechanisms to be deployed in grids Even though UDDI
has been the de facto industry standard for web-services
discovery, imposed requirements of tight-replication among
registries and lack of autonomous control has severely hindered
its widespread deployment and usage. With the advent of grid
computing the scalability issue of UDDI will become a roadblock
that will prevent its deployment in grids. In this paper we present
our distributed web-service discovery architecture, called DUDE
(Distributed UDDI Deployment Engine). DUDE leverages DHT
(Distributed Hash Tables) as a rendezvous mechanism between
multiple UDDI registries. DUDE enables consumers to query
multiple registries, still at the same time allowing organizations to
have autonomous control over their registries.. Based on
preliminary prototype on PlanetLab, we believe that DUDE
architecture can support effective distribution of UDDI registries
thereby making UDDI more robust and also addressing its scaling
issues. Furthermore, The DUDE architecture for scalable
distribution can be applied beyond UDDI to any Grid Service
Discovery mechanism.'''

	text = "Harold ate an apple. Harold did not realize it was a magic apple. If he had known it was magic, he would have ate the whole apple."
	# text = "Information Retrieval is very fun. I learn a lot from the class. There is much to learn from information retrieval."
	main(text)
	# getRake(text)







