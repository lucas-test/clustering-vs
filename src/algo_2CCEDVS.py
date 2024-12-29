
import numpy as np
import networkx as nx

from src.utils import  clusters_from_assignation, floyd_warshall


def printv(verbose, *args):
    if verbose:
        message = ' '.join(map(str, args))
        print(message)

def matrix_power(matrix, power):
    """
    Compute the power of a matrix using exponentiation by squaring.
    
    Parameters:
    - matrix: The input matrix as a NumPy array.
    - power: The exponent to raise the matrix to.
    
    Returns:
    - The matrix raised to the given power.
    """
    result = np.eye(matrix.shape[0]) 
    while power > 0:
        if power & 1:  
            result = np.dot(result, matrix)
        matrix = np.dot(matrix, matrix)  
        power >>= 1 
    return result


def transition_matrix_centered(vertex_neighbors_index, tm_common, neighbors, original_vertices):
    """
    tm_common is the transition matrix of the original_vertices
    """
    d = len(neighbors)
    tmc = np.zeros((d,d))
    for ni in range(d):
        i = neighbors[ni]
        i_index = original_vertices[i]
        s = 0
        for nj in range(d):
            j = neighbors[nj]
            j_index = original_vertices[j]
            tmc[ni][nj] = tm_common[i_index][j_index]
            s += tmc[ni][nj]
        tmc[ni][vertex_neighbors_index] += 1 - s
    return tmc




def compute_order(neighbors, vertex, tm_common, original_vertices):
    d = len(neighbors)

    vertex_id = neighbors.index(vertex)

    # Compute the centered transition matrix of size dxd
    tm = transition_matrix_centered(vertex_id, tm_common, neighbors, original_vertices)

    # Compute tm^8*v_0
    power = 8
    tm_powered = matrix_power(np.transpose(tm), power)
    v = np.zeros((d, 1))
    v[vertex_id] = 1
    v = np.dot(tm_powered, v)
    
    # Order neighbors by decreasing probability
    order = []
    for ni in range(d):
        i = neighbors[ni]
        order.append([i,v[ni]])
    order = sorted(order, key=lambda v: v[1], reverse=True)
    return order



def compute_transition_matrix(G, original_vertices):
    assert isinstance(G, nx.Graph)

    n = len(original_vertices)

    tm = np.zeros((n,n))

    degree = {}
    degree = {v: 0 for v in original_vertices}
    for v in original_vertices:
        for _, _, w in G.edges(v, data='weight', default=1):  # default weight is 1 if not specified
            degree[v] += w


    for i in original_vertices:
        for _, j, w in G.edges(i, data='weight', default=1):  # default weight is 1 if not specified
            if j in original_vertices:
                i_index = original_vertices[i]
                j_index = original_vertices[j]
                tm[i_index][j_index] = w / degree[i]

    return tm


def is_2club(G, cluster):
    dist = floyd_warshall(G, cluster)
    for x in cluster:
        for y in cluster:
            if dist[x][y] > 2:
                return False
    return True






def best(G, order):
    assert isinstance(G, nx.Graph)

    # print(order)
    best_weight = float("inf")
    best_cluster = []
    cluster = []
    problematic_pairs = []

    outdegree = {}
    c = 0

    for i,proba_i in order:

        # Remove all problematic pairs (x,y) which were at distance at least 2 if they are both adjacent to i
        for k in range(len(problematic_pairs)-1,-1,-1):
            x,y = problematic_pairs[k]
            if G.has_edge(x,i) and G.has_edge(y,i):
                del(problematic_pairs[k])
        
        # One liner of previous
        # problematic_pairs = [pair for pair in problematic_pairs if not (G.has_edge(pair[0], i) and G.has_edge(pair[1], i))]

        # Add (x,i) in problematic pairs if x is at dist at least 2 in cluster
        i_neighbors = set(G.neighbors(i))
        for v in cluster:
            if v not in i_neighbors:
                common = False
                for w in G.neighbors(v):
                    if w in i_neighbors and w in cluster:
                        common = True
                        break
                if common == False:
                    problematic_pairs.append((i,v))
        
        cluster.append(i)

        

   
        

        if len(cluster) == 1:
            outdegree[i] = 0.
            for _,j,w in G.edges(i, data="weight"):
                if j not in cluster:
                    outdegree[i] += w
            if outdegree[i] > 1:
                c += 1
            else:
                c += outdegree[i]
            best_weight = outdegree[i]
            best_cluster = [i]
            continue

        outdegree[i] = 0
        for _,j,w in G.edges(i, data="weight"):
            if j not in cluster:
                outdegree[i] += w
            else:
                if outdegree[j] > 1 and outdegree[j] -w <= 1:
                    c -= 1
                    c += outdegree[j]-w
                elif outdegree[j] <= 1:
                    c -= w
                outdegree[j] -= w
        
        if outdegree[i] > 1:
            c += 1
        else:
            c += outdegree[i]




        if len(problematic_pairs) > 0:
            continue

        cost = c*pow(len(cluster), -1)
        if cost == 0:
            return cost, cluster.copy()
        if cost < best_weight:
            best_weight = cost
            best_cluster = cluster.copy()
        # print(i, f"{c:.3}", f"{cost:.2}")
        
    return best_weight, best_cluster


def compute_2neighbors(G, v):
    assert isinstance(G, nx.Graph)

    neighbors = [v]
    for x in G.neighbors(v):
        if x not in neighbors:
            neighbors.append(x)     
        for y in G.neighbors(x):
            if y not in neighbors:
                neighbors.append(y)
    return neighbors


def print_order(order):
    s = ""
    for item in order:
        index = item[0]  # Extract the index
        prob_array = item[1]  # Extract the probability array
        modified_prob = np.floor(prob_array[0] * 100)
        s += (f"({index}, {int(modified_prob)}) ")
    print(s)
 


def solve_private(G, verbose = False):
    assert isinstance(G, nx.Graph)

    original_vertices = {v: i for i, v in enumerate(G.nodes())}
    n = len(G)+1
    assigned = set()
    c = 0

    nb_splits = 0

    assignation = {}
    for v in original_vertices:
        assignation[v] = []
    nb_clusters = 0
    clusters = []

    
    while True:
        # print("round", c)
        best_cost = float("inf")
        best_cluster = []

        tm = compute_transition_matrix(G, original_vertices)

        # Fast version we only check the minimal and the maximal degree vertices
        mindeg = float('inf')
        minv = None
        maxv = None
        maxdeg= 0
        for v in original_vertices:
            if v not in assigned:
                if G.degree(v) > maxdeg:
                    maxdeg = G.degree(v)
                    maxv = v
                if G.degree(v) < mindeg:
                    mindeg = G.degree(v)
                    minv = v

        if minv:
            X1 = compute_2neighbors(G,minv)
            order1 = compute_order(X1, minv, tm, original_vertices)
            cost1, cluster1 = best(G, order1)
            if cost1 < best_cost:
                best_cost = cost1
                best_cluster = cluster1.copy()
        


        if maxv:
            X1 = compute_2neighbors(G,maxv)
            order1 = compute_order(X1, maxv, tm, original_vertices)
            cost1, cluster1 = best(G, order1)
            if cost1 < best_cost:
                best_cost = cost1
                best_cluster = cluster1.copy()


        # For every non assigned original vertex, compute the best cluster from this vertex
        # Record the best one in term of cost
        # for v in original_vertices:
        #     if v not in assigned:
        #         # print("-----------")
        #         # print(v)
        #         X = compute_2neighbors(G,v)
        #         # print("neighbors:",X)
        #         order = compute_order(X, v, tm, original_vertices)
        #         # print_order( order)
        #         cost, cluster = best(G, order)
        #         # print(cost, cluster)
        #         if cost == 0.:
        #             best_cost = 0.
        #             best_cluster = cluster.copy()
        #             break
        #         if cost < best_cost:
        #             best_cost = cost
        #             best_cluster = cluster.copy()

        if best_cluster == []:
            break
        else:
            printv(verbose, f"New cluster: cost: {best_cost:.2} size: {len(best_cluster)}", sorted(best_cluster), len(assigned))
            cluster_with_splits = best_cluster.copy()

            if len(best_cluster) == 1:
                for v in best_cluster:
                    neighbors = list(G.edges(v, data="weight"))
                    for _,j,w in neighbors:
                        # print("del", v, j, " weight: ", w)
                        G.remove_edge(v,j)
                        c += w

            for v in best_cluster:
                assignation[v].append(nb_clusters)

                # Compute the weighted degree of v out of best_cluster
                d = 0
                for _,j,w in G.edges(v, data="weight"):
                    if j not in cluster_with_splits :
                        # print(v, j, w)
                        d += w
                # print("check", v, "degree", d)
                if d > 1:
                    # print("spl", v, "outdegree: ", d)
                    nb_splits += 1
                    G.add_node(n)
                    neighbors = list(G.edges(v, data="weight"))
                    for _,j,w in neighbors:
                        # if j in best_cluster or j not in original_vertices:
                        if j in cluster_with_splits:
                            G.add_edge(n, j, weight = w)
                            G.remove_edge(v, j)
                    cluster_with_splits.append(n)
                    # print(n, G.degree(v), G.degree(n), G.degree(v) + G.degree(n))
                    n += 1
                    c += 1 # splitting
                else:
                    assigned.add(v)
                    neighbors = list(G.edges(v, data="weight"))
                    for _,j,w in neighbors:
                        # if j not in best_cluster and j in original_vertices:
                        if j not in cluster_with_splits:
                            # print("del", v, j, " weight: ", w)
                            G.remove_edge(v,j)
                            c += w
            nb_clusters += 1
            clusters.append(best_cluster)


    printv(verbose, f"Nb clusters: {len(clusters)}")
    printv(verbose, f"Nb_operations: {c}\nNb splits: {nb_splits}")


    # avg_intra, avg_inter, intra_inf, inter_inf, total_intra, total_inter = compute_overall_avg_distances(G,
    #                                                                                                     clusters)

    # # Write final results
    # printv(verbose, f"Final Clustering Results:\n")
    # printv(verbose, f"Number of clusters: {len(clusters)}\n")
    # printv(verbose, f"Average intra-cluster distance: {format_distance(avg_intra)}\n")
    # printv(verbose, f"Average inter-cluster distance: {format_distance(avg_inter)}\n")
    # printv(verbose, f"Intra-cluster infinite pairs: {intra_inf} out of {total_intra}\n")
    # printv(verbose, f"Inter-cluster infinite pairs: {inter_inf} out of {total_inter}\n")
    
    return assignation





def solve(G, nse_file_path, orig):
    H = G.__class__()
    H.add_nodes_from(G)
    H.add_edges_from((u, v, d) for u, v, d in G.edges(data=True)) 
    assignation = solve_private(H, verbose=False)
    clusters = clusters_from_assignation(assignation)
    return assignation, clusters
