# 2-Club Cluster Edge Deletion with Vertex Splitting

We implement two new algorithms:

- 2CCEDVS for 2-Club Cluster Edge Deletion with Vertex Splitting on a graph
- 2CCED for 2-Club Cluster Edge Deletion on a graph

We compare these algorithms on the following real data:

- CE GN [https://networkrepository.com/bio-CE-GN.php]
- CE GT [https://networkrepository.com/bio-CE-GT.php]
- CE HT [https://networkrepository.com/bio-CE-HT.php]
- CE LC [https://networkrepository.com/bio-CE-LC.php]
- DM HT [https://networkrepository.com/bio-DM-HT.php]

We compare our new algorithms with the following existing algorithms:

- MCL [https://github.com/micans/mcl]

- KaPoCE [https://github.com/kittobi1992/cluster_editing]

- ClusterOne [https://github.com/ntamas/cl1]


## Installation

Install networkx (with pip or with apt):

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