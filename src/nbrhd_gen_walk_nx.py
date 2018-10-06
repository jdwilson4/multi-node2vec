'''
Neighborhood aliasing procedure used for fast random walks on multilayer networks. 

Details can be found in the paper: "Fast Embedding of Multilayer Networks: An Algorithm and Application to Group fMRI" 
by JD Wilson, M Baybay, R Sankar, and P Stillman

Preprint here: https://arxiv.org/pdf/1809.06437.pdf

Contributors:
- Melanie Baybay
University of San Francisco, Department of Computer Science
- Rishi Sankar
Henry M. Gunn High School
- James D. Wilson (maintainer)
University of San Francisco, Department of Mathematics and Statistics

Questions or Bugs? Contact James D. Wilson at jdwilson4@usfca.edu
'''


import numpy as np
import networkx as nx
import random
#import multiprocessing
import threading
import time

#is is_directed needed?

class NeighborhoodGen():
	def __init__(self, graph, p, q, thread_limit=1, is_directed=False, weighted=False):
		self.G = graph
		self.is_directed = is_directed
		self.p = p
		self.q = q
		self.weighted = weighted
		self.thread_limit = thread_limit

		self.preprocess_transition_probs()

	def multinode2vec_walk(self, w, walk_length, start_node, start_layer_id):
		'''
		Simulate a random walk starting from start node. (Generate one neighborhood)
		'''

		G = self.G
		alias_nodes = self.alias_nodes
		alias_edges = self.alias_edges

		walk = [start_node] #nbrhd
		cur_layer_id = start_layer_id
		force_switch = False
		while len(walk) < walk_length:
			cur = walk[-1]
			if not force_switch:
				prev_layer_id = cur_layer_id
			random.seed(1234)
			rval = random.random()
			if rval < w or force_switch: #then switch layer
				total_layers = len(G)
				rlay = random.randint(0, total_layers - 2)
				if rlay >= cur_layer_id:
					rlay += 1
				cur_layer_id = rlay
				force_switch = False
			cur_layer = G[cur_layer_id]
			try:
				cur_nbrs = sorted(cur_layer.neighbors(cur))
				if len(cur_nbrs) > 0:
					if len(walk) == 1 or prev_layer_id != cur_layer_id:
						walk.append(cur_nbrs[alias_draw(alias_nodes[cur_layer_id][cur][0], alias_nodes[cur_layer_id][cur][1])])
					else:
						prev = walk[-2]
						next = cur_nbrs[alias_draw(alias_edges[cur_layer_id][(prev, cur)][0],
							alias_edges[cur_layer_id][(prev, cur)][1])]
						walk.append(next)
				else:
					force_switch = True
					continue
			except Exception as e:
				force_switch = True
				continue

		return walk

	def simulate_walks(self, num_walks, walk_length):
		'''
		Repeatedly simulate random walks from each node.
		'''
		G = self.G
		walks = {}
		for layer in G:
			walks[layer] = []
			nodes = list(layer.nodes())
			print('Walk iteration:')
			for walk_iter in range(num_walks):
				print(str(walk_iter+1), '/', str(num_walks))
				random.shuffle(nodes)
				for node in nodes:
					walks[layer].append(self.node2vec_walk(walk_length=walk_length, start_node=node))

		return walks

	def get_alias_edge(self, src, dst, layer):
		'''
		Get the alias edge setup lists for a given edge.
		'''
		p = self.p
		q = self.q

		unnormalized_probs = []
		for dst_nbr in sorted(layer.neighbors(dst)):
			if dst_nbr == src:
				unnormalized_probs.append(layer[dst][dst_nbr]['weight']/p)
			elif layer.has_edge(dst_nbr, src):
				unnormalized_probs.append(layer[dst][dst_nbr]['weight'])
			else:
				unnormalized_probs.append(layer[dst][dst_nbr]['weight']/q)
		norm_const = sum(unnormalized_probs)
		normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

		return alias_setup(normalized_probs)

	def preprocess_transition_probs(self):
		'''
		Preprocessing of transition probabilities for guiding the random walks.
		'''
		G = self.G
		is_directed = self.is_directed

		self.alias_nodes = {}
		self.alias_edges = {}
		self.lock = threading.Lock()

		tlimit = self.thread_limit
		layer_count = len(self.G)
		counter = 0
		if tlimit == 1:
			for i in range(layer_count):
				self.preprocess_thread(self.G[i],i)
		else:
			while counter < layer_count:
				threads = []
				rem = layer_count - counter
				if rem >= tlimit:
					for i in range(tlimit):
						thread = threading.Thread(target=self.preprocess_thread, args=(self.G[counter],counter,))
						threads.append(thread)
						thread.start()
						counter += 1
				else:
					for i in range(rem):
						thread = threading.Thread(target=self.preprocess_thread, args=(self.G[counter],counter,))
						threads.append(thread)
						thread.start()
						counter += 1
				for thread in threads:
					thread.join()

		return

	def preprocess_thread(self, layer, counter):
		start_time = time.time()
		print("Starting thread for layer " + str(counter))
		alias_nodes = {}
		for node in layer.nodes():
			unnormalized_probs = [layer[node][nbr]['weight'] for nbr in sorted(layer.neighbors(node))]
			norm_const = sum(unnormalized_probs)
			normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
			alias_nodes[node] = alias_setup(normalized_probs)

		alias_edges = {}
		triads = {}

		if self.is_directed:
			for edge in layer.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1], layer)
		else:
			for edge in layer.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1], layer)
				alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0], layer)

		self.lock.acquire()
		try:
			self.alias_nodes[counter] = alias_nodes
			self.alias_edges[counter] = alias_edges
		finally:
			self.lock.release()

		print("Finished thread for layer " + str(counter) + " in " + str(time.time() - start_time) + " seconds.")

		return

def alias_setup(probs):
	'''
	Compute utility lists for non-uniform sampling from discrete distributions.
	Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
	for details
	'''
	K = len(probs)
	q = np.zeros(K)
	J = np.zeros(K, dtype=np.int)

	smaller = []
	larger = []
	for kk, prob in enumerate(probs):
		q[kk] = K*prob
		if q[kk] < 1.0:
			smaller.append(kk)
		else:
			larger.append(kk)

	while len(smaller) > 0 and len(larger) > 0:
		small = smaller.pop()
		large = larger.pop()

		J[small] = large
		q[large] = q[large] + q[small] - 1.0
		if q[large] < 1.0:
			smaller.append(large)
		else:
			larger.append(large)

	return J, q

def alias_draw(J, q):
	'''
	Draw sample from a non-uniform discrete distribution using alias sampling.
	'''
	K = len(J)

	kk = int(np.floor(np.random.rand()*K))
	if np.random.rand() < q[kk]:
		return kk
	else:
		return J[kk]
