import networkx as nx

from src.utils import clusters_from_assignation





def has_diameter_at_most_2(G, nodes):
    subgraph = G.subgraph(nodes)
    if not nx.is_connected(subgraph):
        return False
    try:
        return max(nx.eccentricity(subgraph).values()) <= 2
    except nx.NetworkXError:
        return False


def try_split_vertex(G, v, indices):
    neighbors = list(G.neighbors(v))
    if len(neighbors) < 4:
        return False, None

    new_v = max(G.nodes()) + 1
    indices[new_v] = indices[v]
    G_copy = G.copy()
    G_copy.add_node(new_v)

    split_point = len(neighbors) // 2
    for neighbor in neighbors[:split_point]:
        G_copy.remove_edge(v, neighbor)
        G_copy.add_edge(new_v, neighbor)

    cluster1 = {v} | set(G_copy.neighbors(v))
    cluster2 = {new_v} | set(G_copy.neighbors(new_v))

    if has_diameter_at_most_2(G_copy, cluster1) or has_diameter_at_most_2(G_copy, cluster2):
        return True, G_copy
    else:
        del(indices[new_v])
        return False, None



# Main clustering algorithm

def solve_private(G_nx, with_splitting=True):
    assert isinstance(G_nx, nx.Graph)


    original_vertices = [v for v in G_nx.nodes()]

    indices = {}
    for v in G_nx.nodes():
        indices[v] = v

    if with_splitting:
        for v in G_nx.nodes():
            split_successful, new_graph = try_split_vertex(G_nx, v, indices)
            if split_successful:
                G_nx = new_graph


    # Calculate Square Clustering Coefficient
    square_cc = nx.square_clustering(G_nx)

    # Create clustering based on Square Clustering Coefficient
    sorted_vertices = sorted(square_cc.keys(), key=square_cc.get, reverse=True)

    clusters = {}
    unclustered_vertices = set(G_nx.nodes())
    while unclustered_vertices:
        for v in sorted_vertices:
            if v in unclustered_vertices:
                cluster = {v}
                candidates = set(nx.descendants_at_distance(G_nx, v, 1))
                candidates.update(nx.descendants_at_distance(G_nx, v, 2))

                for candidate in candidates:
                    if candidate in unclustered_vertices:
                        temp_cluster = cluster.union({candidate})
                        try:
                            if all(nx.shortest_path_length(G_nx.subgraph(temp_cluster), source=x, target=y) <= 2
                                for x in temp_cluster for y in temp_cluster):
                                cluster.add(candidate)
                        except nx.NetworkXNoPath:
                            continue

                clusters[v] = cluster
                unclustered_vertices -= cluster
                break

    final_clusters = []
    for v,cluster in clusters.items():
        cl = []
        for v in cluster:
            cl.append(indices[v])
        final_clusters.append(cl)


    assignation = {}
    for v in original_vertices:
        assignation[v] = []
        for i,cluster in enumerate(final_clusters):
            if v in cluster:
                assignation[v].append(i)


    return assignation







def solve(G, nse_file_path, orig):
    H = G.__class__()
    H.add_nodes_from(G)
    H.add_edges_from((u, v, d) for u, v, d in G.edges(data=True))
    a = solve_private(H, with_splitting=False)
    return a, clusters_from_assignation(a)
