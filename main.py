base = "CE-HT"


import src.algo_2CCEDVS as algo_2CCEDVS
import src.utils as utils
import time
import src.algo_mcl as algo_MCL
import src.algo_ocluster as algo_OCR
import src.algo_cluster_one as algo_CL1
import src.algo_KPCE as algo_KPCE
import src.algo_2CCED as algo_2CCED
from src.utils import compute_qualities 


def write_clusters_to_file(clusters, file_name='clusters.txt'):
    with open(file_name, 'w') as file:
        for cluster in clusters:
            file.write(' '.join(str(x) for x in cluster) + '\n')



file_name = f"real/bio-{base}.edges"
print(file_name)
G = utils.load_file(file_name, " ")
print("n=", G.number_of_nodes(), "m=",  G.number_of_edges())
print("Algo\tTime(s)\tClust\tOverl.\tAvgDensity")


# OClustR
start_time = time.time()
assignation, clusters = algo_OCR.solve(G, file_name, {})
write_clusters_to_file(clusters, f"real/{base}-OCR.clusters")
end_time = time.time()
q = compute_qualities(G, assignation, clusters)
q.insert(0, end_time - start_time)
q.insert(0, "OCR")
print("\t".join(map(lambda x: f"{x}" if isinstance(x, int) else ( f"{float(x):.3f}" if isinstance(x, float) else f"{x}")  , q)))


# KaPoCE
start_time = time.time()
assignation, clusters = algo_KPCE.solve(G, file_name, {})
write_clusters_to_file(clusters, f"real/{base}-KPCE.clusters")
end_time = time.time()
q = compute_qualities(G, assignation, clusters)
q.insert(0, end_time - start_time)
q.insert(0,"KPCE")
print("\t".join(map(lambda x: f"{x}" if isinstance(x, int) else ( f"{float(x):.3f}" if isinstance(x, float) else f"{x}")  , q)))

# 2CCEDVS
start_time = time.time()
assignation, clusters = algo_2CCEDVS.solve(G, file_name, {})
write_clusters_to_file(clusters, f"real/{base}-2CCEDVS.clusters")
end_time = time.time()
q = compute_qualities(G, assignation, clusters)
q.insert(0, end_time - start_time)
q.insert(0,"2CCEDVS")
print("\t".join(map(lambda x: f"{x}" if isinstance(x, int) else ( f"{float(x):.3f}" if isinstance(x, float) else f"{x}")  , q)))


# ClusterOne
start_time = time.time()
assignation, clusters = algo_CL1.solve(G, file_name, {})
write_clusters_to_file(clusters, f"real/{base}-CL1.clusters")
end_time = time.time()
q =  compute_qualities(G, assignation, clusters)
q.insert(0, end_time - start_time)
q.insert(0,"CL1")
print("\t".join(map(lambda x: f"{x}" if isinstance(x, int) else ( f"{float(x):.3f}" if isinstance(x, float) else f"{x}")  , q)))

# MCL
start_time = time.time()
assignation, clusters = algo_MCL.solve(G, file_name, 2)
end_time = time.time()
write_clusters_to_file(clusters, f"real/{base}-MCL.clusters")
q =  compute_qualities(G, assignation, clusters)
q.insert(0, end_time - start_time)
q.insert(0,"MCL")
print("\t".join(map(lambda x: f"{x}" if isinstance(x, int) else ( f"{float(x):.3f}" if isinstance(x, float) else f"{x}")  , q)))

#2CCED
start_time = time.time()
assignation, clusters = algo_2CCED.solve(G, file_name, {})
end_time = time.time()
write_clusters_to_file(clusters, f"real/{base}-2CCED.clusters")
q =  compute_qualities(G, assignation, clusters)
q.insert(0, end_time - start_time)
q.insert(0,"2CCED")
print("\t".join(map(lambda x: f"{x}" if isinstance(x, int) else ( f"{float(x):.3f}" if isinstance(x, float) else f"{x}")  , q)))
