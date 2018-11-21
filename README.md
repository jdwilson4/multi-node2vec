# multi-node2vec
This is Python source code for the multi-node2vec algorithm. Multi-node2vec is a fast network embedding method for multilayer networks 
that identifies a continuous and low-dimensional representation for the unique nodes in the network. 

Details of the algorithm can be found in the paper: *Fast Embedding of Multilayer Networks: An Algorithm and Application to Group fMRI* 
by JD Wilson, M Baybay, R Sankar, and P Stillman. 

**Preprint**: https://arxiv.org/pdf/1809.06437.pdf

__Contributors__:
- Melanie Baybay
University of San Francisco, Department of Computer Science
- Rishi Sankar
Henry M. Gunn High School
- James D. Wilson (maintainer)
University of San Francisco, Department of Mathematics and Statistics

**Questions or Bugs?** Contact James D. Wilson at jdwilson4@usfca.edu

# Description

## The Mathematical Objective

 A multilayer network of length *m* is a collection of networks or graphs {G<sub>1</sub>, ..., G<sub>m</sub>}, where the graph G<sub>j</sub> models the relational structure of the *j*th layer of the network. Each layer G<sub>j</sub> = (V<sub>j</sub>, W<sub>j</sub>) is described by the vertex set V<sub>j</sub> that describes the units, or actors, of the layer, and the edge weights W<sub>j</sub> that describes the strength of relationship between the nodes. Layers in the multilayer sequence may be heterogeneous across vertices, edges, and size. Denote the set of unique nodes in {G<sub>1</sub>, ..., G<sub>m</sub>} by **N**, and let 
 *N* = |**N**| denote the number of nodes in that set. 
 
The aim of the **multi-node2vec** is to learn an interpretable low-dimensional feature representation of **N**. In particular, it seeks a *D*-dimensional representation

**F**: **N** --> R<sup>*D*</sup>, 

where *D* < < N. The function **F** can be viewed as an *N* x *D* matrix whose rows {**f**<sub>v</sub>: v = 1, ..., N} represent the feature space of each node in **N**. 

## The Algorithm
The **multi-node2vec** algorithm estimates **F** through maximum likelihood estimation, and relies upon two core steps

1) __NeighborhoodSearch__: a collection of vertex neighborhoods from the observed multilayer graph, also known as a *BagofNodes*, is identified. This is done through a 2nd order random walk on the multilayer network.

2) __Optimization__: Given a *BagofNodes*, **F** is then estimated through the maximization of the log-likelihood of **F** | **N**. This is done through the application of stochastic gradient descent on a two-layer Skip-gram neural network model.

The following image provides a schematic:

![multi-node2vec schematic](https://github.com/jdwilson4/multi-node2vec/blob/master/mn2vec_toy.png)

# Running multi-node2vec

## Requirements
- numpy==1.12.1
- gensim==2.3.0


## Usage
```
python3 multi_node2vec.py [--dir [DIR]] [--output [OUTPUT]] [--d [D]] [--nbsize [NBSIZE]][--n_samples [N_SAMPLES]][--w2v_iter [W2V_ITER]] [--w2v_workers [W2V_WORKERS]] [--rvals [RVALS]] [--pvals [PVALS]] [--qvals [QVALS]]
```

***Arguments***

- --dir [directory name]   : Absolute path to directory of correlation/adjacency matrix files in csv format. Note that each .csv should contain an adjacency matrix with columns and rows labeled by the node ID.
- --output [filename]      : Absolute path to output file (no extension).
- --d [dimensions]         : Dimensionality. Default is 100.
- --walk_length [n]        : Length of each random walk for identifying multilayer neighborhoods. Default is 100. 
- --window_size [w]        : Size of context window used for Skip Gram optimization. Default is 10.
- --n_samples [samples]    : Number of times to sample a layer. Default is 1.
- --w2v_workers [workers]  : Number of parallel worker threads. Default is 8.
- --rvals [layer walk prob]: The unnormalized walk probability for traversing layers. Default is .25.
- --pvals [return prob]    : The unnormalized walk probability of returning to a previously seen node. Default is 1.
- --qvals [explore prob]   : The unnormalized walk probability of exploring new nodes. Default is 0.50. 

### Examples

__Quick Test example__

This example runs **multi-node2vec** on a small test multilayer network with 2 layers and 264 nodes in each layer. It takes about 2 minutes to run on a personal computer using 8 cores.
```
python3 multi_node2vec.py --dir data/test --output results/test --d 100 --window_size 2 --n_samples 1 --rvals 0.25
```

__fMRI Case Study__

This example runs **multi-node2vec** on the multilayer network representing group fMRI of 74 healthy controls as run in the paper *Fast Embedding of Multilayer Networks: An Algorithm and Application to Group fMRI*. The model will generate
generate 100 features for each of 264 unique nodes using a walk parameter *r = 0.25*. The values of *p* (=1) and *q* (=0.50) are set to the default of what is available in the original **node2vec** specification. It takes about an hour to run on a personal computer using 8 cores.
```
python3 multi_node2vec.py --dir data/CONTROL_fmt --output results/control --d 100 --window_size 10 --n_samples 1 --rvals 0.25 --pvals 1 --qvals 0.5
```



