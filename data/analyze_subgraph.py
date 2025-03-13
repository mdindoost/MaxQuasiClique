import networkx as nx
import matplotlib.pyplot as plt

def analyze_subgraph(edges_file, subgraph_nodes_file):
    # 1. Read the edge list and build the full graph
    G = nx.Graph()
    with open(edges_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            src, dst = line.split(',')
            # Optionally convert to int if your data is integer-based:
            # src, dst = int(src), int(dst)
            G.add_edge(src, dst)

    # 2. Read the list of subgraph nodes
    subgraph_nodes = set()
    with open(subgraph_nodes_file, 'r') as f:
        for line in f:
            node = line.strip()
            if node:
                # Optionally convert to int if your data is integer-based:
                # node = int(node)
                subgraph_nodes.add(node)

    # 3. Construct the induced subgraph
    H = G.subgraph(subgraph_nodes).copy()

    print("Number of nodes in subgraph:", H.number_of_nodes())
    print("Number of edges in subgraph:", H.number_of_edges())

    # 4a. Check connectivity using nx.is_connected()
    if H.number_of_nodes() > 1:
        if nx.is_connected(H):
            print("The subgraph is a single connected component (is_connected=True).")
        else:
            print("The subgraph is NOT connected (is_connected=False).")
    elif H.number_of_nodes() == 1:
        print("Subgraph has exactly one node, so it's (trivially) connected.")
    else:
        print("Subgraph has no nodes, cannot check connectivity.")

    # 4b. Enumerate all connected components and print them
    connected_components = list(nx.connected_components(H))
    print(f"Number of connected components (by enumeration): {len(connected_components)}")
    for i, comp in enumerate(connected_components, start=1):
        print(f"  Component {i} has {len(comp)} node(s): {comp}")

    # 5. Compute degrees, clustering coefficients, etc.
    degrees = dict(H.degree())
    clustering_coeffs = nx.clustering(H)

    if degrees:
        degree_values = list(degrees.values())
        min_degree = min(degree_values)
        max_degree = max(degree_values)
        mean_degree = sum(degree_values) / len(degree_values)

        print("\nDegree statistics:")
        print("  Min degree:", min_degree)
        print("  Max degree:", max_degree)
        print("  Mean degree:", mean_degree)

        # Node(s) with min degree (in case there's more than one)
        min_degree_nodes = [n for n, d in degrees.items() if d == min_degree]
        print("\nNode(s) with min degree:", min_degree_nodes)

        # === Additional check for the min-degree nodes ===
        # For each node with min degree, gather any neighbors it might have in the subgraph
        print("\nChecking neighbors in subgraph for min-degree nodes:")
        for node in min_degree_nodes:
            neighbors_in_subgraph = list(H.neighbors(node))
            if not neighbors_in_subgraph:
                print(f"  {node} is isolated in the subgraph (no neighbors).")
            else:
                print(f"  {node} has {len(neighbors_in_subgraph)} neighbor(s) in the subgraph: {neighbors_in_subgraph}")

        # Average clustering coefficient for the whole subgraph
        avg_clustering = nx.average_clustering(H)
        print("\nClustering:")
        print("  Average clustering coefficient:", avg_clustering)

        # (Optional) show a histogram of the degree distribution
        plt.hist(degree_values, bins=range(min_degree, max_degree+2))
        plt.title("Degree Distribution of Subgraph")
        plt.xlabel("Degree")
        plt.ylabel("Frequency")
        plt.show()

        # Print each node's degree & clustering
        print("\nNode-level metrics:")
        for node in H.nodes():
            print(f"Node: {node}, Degree: {degrees[node]}, Clustering: {clustering_coeffs[node]}")
    else:
        print("No degrees to compute (subgraph may be empty).")

if __name__ == "__main__":
    # Update with your actual file paths
    edges_file = "edges.csv"
    subgraph_nodes_file = "max_clique_submission_benchmark_162.csv"
    analyze_subgraph(edges_file, subgraph_nodes_file)
