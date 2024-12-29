import subprocess
from src.utils import assignation_from_clusters


def read_cluster_one_output(vertices, filepath):
    clusters = []
    with open(filepath, 'r') as f:
        for line in f:
            cluster = list(map(int, line.strip().split()))
            clusters.append(cluster)

    for v in vertices:
        found = False
        for cluster in clusters:
            if v in cluster:
                found = True
                break
        if found == False:
            clusters.append([v])
    return clusters


def execute_cluster_one(filepath):
    command = f"java -jar solvers/cluster_one-1.2.jar {filepath} -F plain 2> /dev/null > temp.cl1.out"
    # command = f"java -jar solvers/cluster_one-1.2.jar {filepath} -F plain > temp.cl1.out"
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to execute CL1 command for {filepath}: {e}")




def solve(G, nse_file_path, original_assignation):
    execute_cluster_one(nse_file_path)
    clusters_CL1 = read_cluster_one_output(G.nodes(), "temp.cl1.out")
    assignation_CL1 = assignation_from_clusters(clusters_CL1)
    return assignation_CL1, clusters_CL1

