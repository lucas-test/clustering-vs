import statistics
import networkx as nx



def load_file(file_name, delimiter="\t"):
    """
    Return the weighted graph
    """
    G = nx.Graph()

    with open(file_name, 'r') as file:
        next(file)
        for line in file:
            node1, node2, weight = map(float, line.strip().split(delimiter))
            G.add_edge(int(node1), int(node2), weight=1) 
            G.nodes[int(node1)].setdefault('peso', 1) # for OCR
            G.nodes[int(node2)].setdefault('peso', 1)
            G.nodes[int(node1)].setdefault('dicionario', {})
            G.nodes[int(node2)].setdefault('dicionario', {})

    return G





def floyd_warshall(G, cluster):
    """
    Floyd Warhsall to compute the distances in the induced subgraph by cluster
    """
    n = len(G)
    dist = {}
    for i in cluster:
        dist[i] = {}
        for j in cluster:
            if G.has_edge(i,j):
                dist[i][j] = 1
            else:
                dist[i][j] = float('inf')
        dist[i][i] =0
    
    for k in cluster:
        for i in cluster:
            for j in cluster:
                dist[j][i] = min(dist[j][i], dist[j][k] + dist[k][i])
    return dist





def compute_qualities(G, assignation, clusters):
    """
    Return [number of clusters, overlapping, average density of the clusters]
    """
    ol = overlapping(assignation)
    return [len(clusters), ol, get_average_density(G, clusters)]




def total_internal_edge_weight(G, cluster):
    assert isinstance(G, nx.Graph)
    sum = 0.0
    for v in cluster:
        G.edges(v, data="weight")
        for _, u, w in G.edges(v, data='weight', default=1):
            if u in cluster:
                sum += w
    return sum

def total_external_edge_weight(G, cluster):
    assert isinstance(G, nx.Graph)
    sum = 0.0
    for v in cluster:
        G.edges(v, data="weight")
        for _, u, w in G.edges(v, data='weight', default=1):
            if u not in cluster:
                sum += w
    return sum

def get_density(G, cluster):
    assert isinstance(G, nx.Graph)
    if len(cluster) < 2:
        return 0.0
    return 2.0 * total_internal_edge_weight(G, cluster) / (len(cluster) * (len(cluster) - 1))




def get_average_density(G, clusters):
    assert isinstance(G, nx.Graph)
    sum = 0.0
    for cluster in clusters:
        sum += get_density(G, cluster)
    return sum/len(clusters)



# protected double getSignificanceReal() {
# 		double[] inWeights = new double[this.size()];
# 		double[] outWeights = new double[this.size()];
# 		IntHashSet memberHashSet = this.getMemberHashSet();
# 		int j;
		
# 		Arrays.fill(inWeights, 0.0);
# 		Arrays.fill(outWeights, 0.0);
		
# 		j = 0;
# 		for (int i: members) {
# 			int[] edgeIdxs = this.graph.getAdjacentEdgeIndicesArray(i, Directedness.ALL);
# 			for (int edgeIdx: edgeIdxs) {
# 				double weight = this.graph.getEdgeWeight(edgeIdx);
# 				int endpoint = this.graph.getEdgeEndpoint(edgeIdx, i);
# 				if (memberHashSet.contains(endpoint)) {
# 					/* This is an internal edge */
# 					inWeights[j] += weight;
# 				} else {
# 					/* This is a boundary edge */
# 					outWeights[j] += weight;
# 				}
# 			}
# 			j++;
# 		}
		
# 		/* Internal edges were found twice, divide the result by two */
# 		MannWhitneyTest test = new MannWhitneyTest(inWeights, outWeights, H1.GREATER_THAN);
# 		return test.getSP();
# 	}



def compute_overall_avg_distances(G, clusters):
    all_nodes = set(G.nodes())
    intra_distances = []
    inter_distances = []
    intra_inf_count = 0
    inter_inf_count = 0

    i = 0
    for cluster in clusters:
        print(i)
        i += 1
        subgraph = G.subgraph(cluster)
        for u in cluster:
            for v in cluster:
                if u != v:
                    try:
                        distance = nx.shortest_path_length(subgraph, u, v)
                        intra_distances.append(distance)
                    except nx.NetworkXNoPath:
                        intra_inf_count += 1

        other_nodes = all_nodes - set(cluster)
        for u in cluster:
            for v in other_nodes:
                try:
                    distance = nx.shortest_path_length(G, u, v)
                    inter_distances.append(distance)
                except nx.NetworkXNoPath:
                    inter_inf_count += 1

    finite_intra = [d for d in intra_distances if d != float('inf')]
    finite_inter = [d for d in inter_distances if d != float('inf')]

    avg_intra = statistics.mean(finite_intra) if finite_intra else float('inf')
    avg_inter = statistics.mean(finite_inter) if finite_inter else float('inf')

    return avg_intra, avg_inter, intra_inf_count, inter_inf_count, len(intra_distances), len(inter_distances)

def overlapping(assignation):
    s = 0
    for _,l in assignation.items():
        s += len(l)
    return s/len(assignation)


def assignation_from_clusters(clusters):
    vertices = set()
    for cluster in clusters:
        for v  in cluster:
            if v not in vertices:
                vertices.add(v)

    assignation = {}
    for v in vertices:
        assignation[v] = []
        for i,cluster in enumerate(clusters):
            if v in cluster:
                assignation[v].append(i)
    return assignation


def clusters_from_assignation(assignation):
    clusters = {}
    for v,l in assignation.items():
        for c in l:
            if c not in clusters:
                clusters[c] = []
            clusters[c].append(v)
    clusters2 = []
    for key,cluster in clusters.items():
        clusters2.append(cluster)
    return clusters2


def nb_overlapping_vertices(assignation):
    c = 0
    for v,l in assignation.items():
        if len(l) > 1:
            c += 1
    return c

def p_out(G, assignation):
    assert isinstance(G, nx.Graph)
    p = 0
    c = 0
    # Compute the number of edges between two clusters (assignation is map(node, List(int))
    for u, v in G.edges():
        clusters_u = set(assignation[u])
        clusters_v = set(assignation[v])
        if not clusters_u.intersection(clusters_v):
            p += 1

    clusters_id = []
    for v,l in assignation.items():
         for k in l:
            if k not in clusters_id:
                clusters_id.append(k)
    
    for k1 in clusters_id:
        for k2 in clusters_id:
            if k1 < k2:
                for u in G.nodes():
                    if k1 in assignation[u]:
                        clusters_u = set(assignation[u])
                        for v in G.nodes():
                            if k2 in assignation[v]:
                                clusters_v = set(assignation[v])
                                if u != v and G.has_edge(u,v) == False:
                                    c += 1

    

    n = G.number_of_nodes()
    total_possible_edges = n * (n - 1) / 2
    return p / (p + c)


def sparsity(G, clusters):
    assert isinstance(G, nx.Graph)
    assert isinstance(clusters, list), "assignation must be a list"

    density = 0
    s = 0 # sparsity

    for cluster in clusters:
        media_int = 0
        media_est = 0
        
        for j in cluster:
            kinj = 0
            for v in G.neighbors(j):
                if v in cluster:
                    kinj += 1

            media_int += kinj
            media_est += G.degree(j) - kinj
        
        size = len(cluster)
        pair_num = size * (size - 1)
        pair_num_e = (len(G) - size) * size
        
        if pair_num != 0:
            density += media_int / pair_num
        if pair_num_e != 0:
            s += media_est / pair_num_e

    density /= len(clusters)
    s /= len(clusters)
    return s





