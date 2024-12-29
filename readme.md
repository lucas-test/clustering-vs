# 2-Club Cluster Edge Deletion with Vertex Splitting

We implement two new algorithms:

- 2CCEDVS for 2-Club Cluster Edge Deletion with Vertex Splitting on a graph
- 2CCED for 2-Clib Cluster Edge Deletion on a graph

We compare these algorithms on the following real data:

- CE GN
- CE GT
- CE HT
- CE LC
- DM HT

We compare our new algorithms with the following existing algorithms:

- OCluster

- MCL [https://github.com/micans/mcl]

- KaPoCE [https://github.com/kittobi1992/cluster_editing]

- ClusterOne [https://github.com/ntamas/cl1]


## Installation

Install networkx:

    pip install networkx


Build the solvers from the sources:

- MCL [https://github.com/micans/mcl]

- KaPoCE [https://github.com/kittobi1992/cluster_editing] (This algorithm works with node indices starting at 1)

- ClusterOne [https://github.com/ntamas/cl1] just download the .jar release file

And place them in the folder solvers/


## Run comparisons

Run the script `main.py` for comparing the different algorithms on one real instance.
Change the `base` variable to change the real instance.
For every algorithm, it computes the running time, the number of clusters, the overlapping and the average density of the clusters.