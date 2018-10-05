# multi-node2vec
This is Python source code for the multi-node2vec algorithm. Multi-node2vec is a fast network embedding method for multilayer networks 
that identifies a continuous and low-dimensional representation for the unique nodes in the network. 

# The Mathematical Objective

 A multilayer network of length *m* is a collection of networks or graphs {G<sub>1</sub>, ..., G<sub>m</sub>}, where the graph G<sub>j</sub> models the relational structure of the *j*th layer of the network. Each layer G<sub>j</sub> = (V<sub>j</sub>, W<sub>j</sub>) is described by the vertex set V<sub>j</sub> that describes the units, or actors, of the layer, and the edge weights W<sub>j</sub> that describes the strength of relationship between the nodes. Layers in the multilayer sequence may be heterogeneous across vertices, edges, and size. Denote the set of unique nodes in {G<sub>1</sub>, ..., G<sub>m</sub>} by **N**, and let 
 *N* = |**N**| denote the number of nodes in that set. 
 
The aim of the **multi-node2vec** is to learn an interpretable low-dimensional feature representation of **N**. In particular, it seeks a *D*-dimensional representation

**F**: **N** --> R<sup>*D*</sup>, 

where *D* < < N. The function **F** can be viewed as an *N* x *D* matrix whose rows {**f**<sub>v</sub>: v = 1, ..., N} represent the feature space of each node in **N**. 
