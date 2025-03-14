import networkx as nx

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

def main():
    # File paths
    graph_file = "/home/mohammad/MaxQuasiClique/data/flywire_edges_converted.txt"
    solution_file1 = "solution_50_175_3.txt"  # Solution A
    solution_file2 = "solution_50_175_1.txt"  # Solution B

    # Load full graph and solution node sets
    G = read_graph(graph_file)
    sol1 = read_nodes(solution_file1)
    sol2 = read_nodes(solution_file2)

    # Compute differences:
    diff1 = sol1 - sol2  # Nodes in solution1 (A) but not in solution2 (B)
    diff2 = sol2 - sol1  # Nodes in solution2 (B) but not in solution1 (A)

    print("Nodes in solA but not in solB:")
    for node in sorted(diff1):
        full_degree = G.degree(node)
        # Degree in its subgraph: count neighbors that are in solution1 (A)
        subgraph_degree = sum(1 for neighbor in G.neighbors(node) if neighbor in sol1)
        # Edges connecting to solution2 (B)
        edges_to_sol2 = sum(1 for neighbor in G.neighbors(node) if neighbor in sol2)
        print(f"  Node {node}: degree(g) {full_degree}, degree(subg) {subgraph_degree}, edges to solB: {edges_to_sol2}")

    print("\nNodes in solB but not in solA:")
    for node in sorted(diff2):
        full_degree = G.degree(node)
        # Degree in its subgraph: count neighbors that are in solution2 (B)
        subgraph_degree = sum(1 for neighbor in G.neighbors(node) if neighbor in sol2)
        # Edges connecting to solution1 (A)
        edges_to_sol1 = sum(1 for neighbor in G.neighbors(node) if neighbor in sol1)
        print(f"  Node {node}: degree in graph {full_degree}, degree in its subgraph {subgraph_degree}, edges to solution_50_175_1.txt: {edges_to_sol1}")

if __name__ == "__main__":
    main()
