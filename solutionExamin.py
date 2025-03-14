import networkx as nx
import matplotlib.pyplot as plt
import statistics

def read_graph(edges_file):
    """
    Reads an edge list from a text file.
    Lines starting with '#' are ignored.
    Each valid line should have two numbers separated by whitespace.
    """
    G = nx.Graph()
    with open(edges_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            u, v = parts
            G.add_edge(int(u), int(v))
    return G

def read_nodes(nodes_file):
    """
    Reads node IDs from a file, one per line.
    """
    nodes = set()
    with open(nodes_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                nodes.add(int(line))
    return nodes

def compute_degree_stats(G):
    """
    Returns the min, max, and median of node degrees for graph G.
    """
    degrees = [deg for _, deg in G.degree()]
    if degrees:
        return min(degrees), max(degrees), statistics.median(degrees)
    else:
        return None, None, None

def compute_diameter(G):
    """
    Computes the diameter of G.
    If G is not connected, computes the diameter of each connected component
    and returns the maximum diameter.
    """
    if nx.is_connected(G):
        return nx.diameter(G)
    else:
        diameters = []
        for comp in nx.connected_components(G):
            H = G.subgraph(comp)
            diameters.append(nx.diameter(H))
        return max(diameters) if diameters else None

def find_top3_longest_paths(G):
    """
    Finds the top 3 longest shortest paths (by distance) in graph G.
    Note: This uses an all-pairs shortest path calculation and may be expensive
    for very large graphs.
    """
    lengths = dict(nx.all_pairs_shortest_path_length(G))
    pairs = []
    for u, dists in lengths.items():
        for v, dist in dists.items():
            if u < v:  # avoid duplicate pairs (u,v) and (v,u)
                pairs.append((u, v, dist))
    # Sort pairs by distance descending
    pairs_sorted = sorted(pairs, key=lambda x: x[2], reverse=True)
    return pairs_sorted[:3]

def main():
    # Update these filenames to match your files
    edges_file = "/home/mohammad/MaxQuasiClique/data/flywire_edges_converted.txt"  # file with graph edges
    subgraph_file = "solution_in_progress.txt"  # file with subgraph nodes

    # 1. Read the graph from file
    G = read_graph(edges_file)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # 2. Read the subgraph nodes and create induced subgraph H
    sub_nodes = read_nodes(subgraph_file)
    H = G.subgraph(sub_nodes).copy()
    print(f"Subgraph: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges")
    
    # 3. Compute the top 3 longest shortest paths in the subgraph
    top3 = find_top3_longest_paths(H)
    print("\nTop 3 longest shortest paths in the subgraph:")
    for u, v, dist in top3:
        print(f"Nodes {u} to {v} with distance {dist}")
    
    # 4. Compute and print the diameter of the subgraph
    diam_H = compute_diameter(H) if H.number_of_nodes() > 0 else None
    print(f"\nDiameter of subgraph: {diam_H}")
    
    # 5. Compute degree statistics for G and H
    min_deg_G, max_deg_G, med_deg_G = compute_degree_stats(G)
    min_deg_H, max_deg_H, med_deg_H = compute_degree_stats(H)
    print("\nGraph node degree stats: min =", min_deg_G, ", max =", max_deg_G, ", median =", med_deg_G)
    print("Subgraph node degree stats: min =", min_deg_H, ", max =", max_deg_H, ", median =", med_deg_H)
    
    # 6. Remove the subgraph from the original graph and analyze the remainder
    R = G.copy()
    R.remove_nodes_from(sub_nodes)
    print(f"\nAfter removing subgraph, remaining graph has {R.number_of_nodes()} nodes and {R.number_of_edges()} edges")
    
    if R.number_of_nodes() > 0:
        min_deg_R, max_deg_R, med_deg_R = compute_degree_stats(R)
        if nx.is_connected(R):
            diam_R = nx.diameter(R)
        else:
            diam_R = "Graph not connected"
        print("Remaining graph node degree stats: min =", min_deg_R, ", max =", max_deg_R, ", median =", med_deg_R)
        print("Diameter of remaining graph:", diam_R)
    else:
        print("Remaining graph is empty.")
    
    # 7. Draw the subgraph and highlight specific nodes, lowest degree nodes, and highest degree nodes.
    # Define the nodes to highlight:
    highlight_nodes = {2904, 4160, 4565, 7450, 8788, 10375, 17862, 37758}
    # Check which of these nodes are actually present in H
    present_highlights = highlight_nodes.intersection(H.nodes())
    absent_highlights = highlight_nodes - set(H.nodes())
    if absent_highlights:
        print("Warning: The following nodes are not in the subgraph and won't be highlighted:", absent_highlights)
    
    # Compute the minimum and maximum degree in H
    degrees_H = dict(H.degree())
    min_degree = min(degrees_H.values())
    max_degree = max(degrees_H.values())
    
    # Use a layout for drawing
    pos = nx.spring_layout(H)
    node_colors = []
    for node in H.nodes():
        if node in present_highlights:
            node_colors.append("red")
        elif degrees_H[node] == max_degree:
            node_colors.append("green")
        elif degrees_H[node] == min_degree:
            node_colors.append("blue")
        else:
            node_colors.append("lightblue")
    
    # Draw the subgraph with labels using a smaller font size for node labels
    nx.draw(H, pos, with_labels=True, node_color=node_colors, node_size=300, font_size=6)
    plt.title("Subgraph with Highlighted, Lowest & Highest Degree Nodes")
    plt.show()

if __name__ == "__main__":
    main()
