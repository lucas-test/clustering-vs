from src.utils import assignation_from_clusters
import subprocess


def solve(G, nse_file_path, inflation):
    execute_mcl_command(nse_file_path, inflation)
    clusters = read_mcl_output("temp.out")
    MCL_assignation = assignation_from_clusters(clusters)
    return MCL_assignation, clusters



def read_mcl_output(filepath):
    clusters = []
    with open(filepath, 'r') as f:
        for line in f:
            cluster = list(map(int, line.strip().split()))
            clusters.append(cluster)
    return clusters




def execute_mcl_command(filepath, inflation):
    """Execute the MCL command for a given file."""
    command = f"./solvers/mcl {filepath} --abc -I {inflation} -V all -o temp.out" # -overlap split
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to execute MCL command for {filepath}: {e}")


