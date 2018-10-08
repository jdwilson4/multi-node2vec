"""
Core functions of the multi-node2vec algorithm. 

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
"""
from gensim.models import word2vec as w2v
from .mltn2v_utils import *
from .nbrhd_gen_walk_nx import *
import time
import networkx as nx


# -------------------------------------------------------------------------------
# multinode2vec
# -------------------------------------------------------------------------------
def generate_features(nbrhds, d, out, nbrhd_size=-1, w2v_iter=1, workers=8, sg=1):
    """
    Generates d features for each unique node in a multilayer network based on
    its neighborhood.

    :param G_m: multilayer graph
    :param d: feature dimensionality
    :param out: absolute path for output file (no extension, file type)
    :param nbrhd_size: number of neighbors for each node. Default: max adjacency
    :param depth: minimum number of layers that a neighborhood persists
    :param n_samples: number of generated neighborhoods per node
    :param w2v_iter: number of word2vec training epochs
    :param workers: number of workers
    :param sg: sets word2vec architecture. 1 for Skip-Gram, 0 for CBOW
    :return: n x d network embedding
    """
    print("Total Neighborhoods: {}".format(len(nbrhds)))
    w2v_model = w2v.Word2Vec(nbrhds, size=d, window=nbrhd_size, min_count=0,
                             workers=workers, iter=w2v_iter, sg=sg)
    embfile = out + ".emb"
    splitpath = embfile.split('/')
    if len(splitpath) > 1:
    	dirs = embfile[:-len(splitpath[-1])]
    	if not os.path.exists(dirs):
    		os.makedirs(dirs)
    if not os.path.exists(embfile):
    	with open(embfile, 'w'): pass
    w2v_model.wv.save_word2vec_format(embfile)
    ftrs = emb_to_pandas(embfile)
    feature_matrix_to_csv(ftrs, out)
    return ftrs


# -------------------------------------------------------------------------------
# NEIGHBORHOODS
# -------------------------------------------------------------------------------
def extract_neighborhoods_walk(layers, nbrhd_size, wvals, p, q, is_directed=False, weighted=False):
    nxg = []
    for layer in layers:
        nxg.append(nx.convert_matrix.from_pandas_edgelist(layer,edge_attr='weight'))

    start = time.time()
    nbrhd_gen = NeighborhoodGen(nxg, p, q, is_directed=is_directed, weighted=weighted)
    print("Finished initialization of neighborhood generator in " + str(time.time() - start) + " seconds.")

    neighborhood_dict = {}
    for w in wvals:
        neighborhoods = []
        for i in range(len(nxg)):
            layer = nxg[i]
            for node in layer.nodes():
                for j in range(52):
                    neighborhoods.append(nbrhd_gen.multinode2vec_walk(w, nbrhd_size, node, i))
        print("Finished nbrhd generation for w=" + str(w))
        neighborhood_dict[w] = neighborhoods

    return neighborhood_dict

def extract_neighborhoods(layers, nbrhd_size, n_samples, weighted=False):
    """
    Extracts neighborhoods of length, nbrhd_size, for each node in each layer.
    :param layers: list of adjacency lists as pandas DataFrames with columns ["source", "target", "weight"]
    :param nbrhd_size: number of nodes per neighborhood
    :param n_samples: number of samples per node
    :param weighted: whether to select neighborhoods by highest weight
    :return: list of neighborhoods, represented as lists
    """
    neighborhoods = []
    if weighted:
        for layer in layers:
            for node in layer["source"].unique():
                neighbors = layer.loc[layer["source"] == node, "target"]
                neighbors.sort_values(by="weight", ascending=False, inplace=True)
                neighborhoods.extend(
                    extract_node_neighborhoods(node, neighbors, nbrhd_size, n_samples)
                )
    else:
        for layer in layers:
            for node in layer["source"].unique():
                neighbors = layer.loc[layer["source"] == node, "target"]
                neighborhoods.extend(
                    extract_node_neighborhoods(node, neighbors, nbrhd_size, n_samples)
                )
    return neighborhoods


def extract_node_neighborhoods(node, neighbors, nbrhd_size, n_samples):
    if len(neighbors) < nbrhd_size:
        print("[WARNING] Selected neighborhood size {} > node-{}'s degree {}. "
              "Setting neighborhood size to {} for node-{}."
              .format(nbrhd_size, node, len(neighbors), len(neighbors), node))
        nbrhd_size = len(neighbors)
    node_neighborhoods = []
    n = 0
    while n < n_samples:
        nbrhd = [node]
        nbrhd.extend(neighbors.sample(n=nbrhd_size-1).values)
        node_neighborhoods.append(nbrhd)
        n += 1
    return node_neighborhoods


# -------------------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------------------
def emb_to_pandas(emb_file):
    """
    Converts embedding file, as extracted from trained word2vec model, to a numpy n-dimensional array.

    :param emb_file: absolute path to word2vec embedding file
    :return: numpy ndarray, (N x d)
    """
    ftrs = pd.read_csv(emb_file, delim_whitespace=True, skiprows=1, header=None, index_col=0)
    ftrs.sort_index(inplace=True)
    return ftrs
