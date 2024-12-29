import networkx as nx

from src.utils import clusters_from_assignation

def graph_to_txt(graph, filename, offset=0):
    """
    Saves a NetworkX graph to a text file in the format:
    p cep n m
    <node1> <node2>
    ...
    """
    with open(filename, 'w') as f:
        f.write(f"p cep {graph.number_of_nodes()} {graph.number_of_edges()}\n")
        for edge in graph.edges(data=False):
            f.write(f"{edge[0]+offset} {edge[1]+offset}\n")

import subprocess


def apply_edit(G, edit_file_name):
    """
    Return the weighted graph
    and the communities dictionnary (vertex id -> list int)
    """
    assert isinstance(G, nx.Graph)

    with open(edit_file_name, 'r') as file:
        next(file)
        for line in file:
            u, v = map(float, line.strip().split(' '))
            u -= 1
            v -= 1
            if G.has_edge(u,v):
                G.remove_edge(u,v)
            else:
                G.add_edge(u,v, weight=1)




def get_assignation(G):
    """
    Assuming that G is an union of clusters (cliques)
    """
    assert isinstance(G, nx.Graph)
    c = 0
    assignation = {}
    for v in G.nodes():
        if v in assignation:
            continue
        for w in G.neighbors(v):
            if w in assignation:
                assignation[v] = assignation[w]
                break
        if v not in assignation:
            assignation[v] = [c]
            c += 1
    return assignation
    



def execute_KPCE(txt_filename, edit_filename):
    """
    Processes a NetworkX graph, saves it to a text file, and executes a shell command
    to pipe the content through a heuristic program.
    """
    command = f"cat {txt_filename} | ../solvers/kpce_heuristics > {edit_filename}"
    subprocess.run(command, shell=True, check=True)





def solve(G, nse_file_path, orig):
    H = G.__class__()
    H.add_nodes_from(G)
    H.add_edges_from((u, v, d) for u, v, d in G.edges(data=True))
    graph_to_txt(H, "temp.txt", 1)
    # execute_KPCE("temp.txt", "temp.edit")
    apply_edit(H, "temp.edit")
    assign_KPCE = get_assignation(H)
    clusters = clusters_from_assignation(assign_KPCE)
    return assign_KPCE, clusters