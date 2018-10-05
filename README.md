# multi-node2vec
This is Python source code for the multi-node2vec algorithm. Multi-node2vec is a fast network embedding method for multilayer networks 
that identifies a continuous and low-dimensional representation for the unique nodes in the network. 

# The Mathematical Objective

 A multilayer network of length $m$ is a collection of networks or graphs $\{G_1, \ldots, G_m\}$, where the graph $G_{\ell}$ models the relational structure of the $\ell$th layer of the network.
 Each layer $G_\ell = (V_{\ell}, W_\ell)$ is described by the vertex set $V_\ell$ that describes the units, or actors, of the layer, and the edge weights $W_\ell = \{w_{\ell}(u,v): u, v \in V_\ell\}$
 that describes the strength of relationship between the nodes. Layers may be viewed as ordered or unordered depending on the application; thus, dynamic networks are a special case of multilayer networks 
 with layers ordered through time. Layers in the multilayer sequence may be heterogeneous across vertices, edges, and size. Denote the set of unique nodes in $\{G_1, \ldots, G_m\}$ by $\mathcal{N}$, and let 
 $N = |\mathcal{N}|$ denote the number of nodes in that set. 
 
The aim of the **multi-node2vec** is to learn an interpretable low-dimensional feature representation of $\mathcal{N}$. In particular, it seeks a $D$-dimensional representation

\begin{equation} \mathbf{F}: \mathcal{N} \rightarrow \mathbb{R}^D, \end{equation}
where $D < < N$. The function $\mathbf{F}$ can be viewed as an $N \times D$ matrix whose rows $\{\mathbf{f}_v: v = 1, \ldots, N \}$ represent the feature space of each node in $\mathcal{N}$. 
