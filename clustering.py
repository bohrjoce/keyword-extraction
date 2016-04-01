import sklearn.cluster
import numpy as np
def kcluster(num_cluster, weight_array, one_hot_tokens, num_key = 10):
	# num_key is the number of keyword that we extract from a cluster
	# we can find the union of the extracted keys from each cluster

	# in case we have less vector than cluste number
	num_cluster = min(num_cluster, len(weight_array))
	k_clusters = sklearn.cluster.k_means(weight_array, num_cluster)[0]
	union_array = []
	num_key = min(num_key, len(one_hot_tokens))
	for vec in k_clusters:
		tmp = sorted(range(len(vec)), key=lambda i: vec[i])[-num_key:]
		union_array = list(set(tmp) | set(union_array))
	res = []
	for ind in union_array:
		res.append(one_hot_tokens[ind])
	return res