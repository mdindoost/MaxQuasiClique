#!/usr/bin/env python3
"""
Extract insights from a known solution for the MaxQuasiClique challenge.
This script works with minimal information - just the node IDs of the solution.
"""

import sys
import os
import csv
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import numpy as np

def read_solution_from_input(input_content):
    """Read solution nodes from input content (pasted in terminal)."""
    solution_nodes = []
    lines = input_content.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if line:
            try:
                node_id = int(line)
                solution_nodes.append(node_id)
            except ValueError:
                print(f"Warning: Invalid node ID: {line}")
    
    return solution_nodes

def find_solution_nodes_in_graph(solution_nodes, edge_file):
    """Find the solution nodes in the graph and analyze their properties."""
    print(f"Looking for {len(solution_nodes)} solution nodes in {edge_file}...")
    
    # Track node degrees and connections within solution
    node_degrees = defaultdict(int)
    solution_connections = defaultdict(set)
    solution_set = set(solution_nodes)
    
    # Count edges
    edge_count = 0
    line_count = 0
    solution_edges = 0
    
    # Process the edge file
    with open(edge_file, 'r') as f:
        for line in f:
            line_count += 1
            if line_count % 1000000 == 0:
                print(f"  Processed {line_count:,} lines, found {solution_edges} solution edges")
            
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        u = int(parts[0])
                        v = int(parts[1])
                        
                        # Increment node degrees
                        node_degrees[u] += 1
                        node_degrees[v] += 1
                        
                        # Check if both nodes are in solution
                        if u in solution_set and v in solution_set:
                            solution_connections[u].add(v)
                            solution_connections[v].add(u)
                            solution_edges += 1
                        
                        edge_count += 1
                    except ValueError:
                        continue
    
    print(f"Finished processing {line_count:,} lines")
    print(f"Found {solution_edges} edges between solution nodes")
    
    # Verify quasi-clique property
    possible_edges = len(solution_nodes) * (len(solution_nodes) - 1) // 2
    density = solution_edges / possible_edges if possible_edges > 0 else 0
    
    print(f"\n=== Solution Properties ===")
    print(f"Nodes: {len(solution_nodes)}")
    print(f"Edges within solution: {solution_edges:,} / {possible_edges:,} possible")
    print(f"Density: {density:.4f}")
    print(f"Is valid quasi-clique? {'Yes' if density > 0.5 else 'No'}")
    
    # Analyze node degrees
    solution_degrees = {node: len(connections) for node, connections in solution_connections.items()}
    avg_solution_degree = sum(solution_degrees.values()) / len(solution_degrees) if solution_degrees else 0
    
    print(f"Average connections within solution: {avg_solution_degree:.2f}")
    
    # Get total degrees in the whole graph
    total_degrees = {node: node_degrees[node] for node in solution_nodes}
    avg_total_degree = sum(total_degrees.values()) / len(total_degrees) if total_degrees else 0
    
    print(f"Average total degree in graph: {avg_total_degree:.2f}")
    
    # Identify potential seed nodes (highest connectivity within solution)
    sorted_nodes = sorted(solution_degrees.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 nodes by connections within solution:")
    for i, (node, degree) in enumerate(sorted_nodes[:10]):
        total_degree = total_degrees.get(node, 0)
        print(f"  {i+1}. Node {node}: {degree} internal connections, {total_degree} total connections")
    
    return solution_connections, solution_degrees, total_degrees

def analyze_id_patterns(solution_nodes):
    """Analyze patterns in the node IDs."""
    print("\n=== Node ID Analysis ===")
    
    # Convert to numpy array for easier analysis
    nodes = np.array(solution_nodes)
    
    # Basic statistics
    min_id = np.min(nodes)
    max_id = np.max(nodes)
    mean_id = np.mean(nodes)
    median_id = np.median(nodes)
    std_id = np.std(nodes)
    
    print(f"ID Range: {min_id} to {max_id}")
    print(f"Range size: {max_id - min_id:,}")
    print(f"Mean ID: {mean_id:.2f}")
    print(f"Median ID: {median_id:.2f}")
    print(f"Standard deviation: {std_id:.2f}")
    
    # Check for patterns in the most significant digits
    digits = [int(str(node)[:3]) for node in nodes]
    digit_counts = pd.Series(digits).value_counts()
    
    print("\nMost common 3-digit prefixes:")
    for prefix, count in digit_counts.head(5).items():
        print(f"  Prefix {prefix}: {count} nodes ({count/len(nodes)*100:.1f}%)")
    
    # Create ID buckets to check distribution
    bucket_size = 10**5  # Adjust based on ID range
    buckets = defaultdict(int)
    
    for node in nodes:
        bucket = node // bucket_size
        buckets[bucket] += 1
    
    sorted_buckets = sorted(buckets.items())
    
    print("\nID distribution by buckets:")
    for bucket, count in sorted(buckets.items(), key=lambda x: x[1], reverse=True)[:5]:
        bucket_start = bucket * bucket_size
        bucket_end = (bucket + 1) * bucket_size - 1
        print(f"  Range {bucket_start:,} to {bucket_end:,}: {count} nodes ({count/len(nodes)*100:.1f}%)")
    
    return

def analyze_solution_structure(solution_connections):
    """Analyze the structure of the solution using the connection information."""
    print("\n=== Solution Structure Analysis ===")
    
    # Create a graph from the solution connections
    G = nx.Graph()
    for node, neighbors in solution_connections.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)
    
    # Basic properties
    n = G.number_of_nodes()
    m = G.number_of_edges()
    print(f"Graph created with {n} nodes and {m} edges")
    
    # Connected components
    components = list(nx.connected_components(G))
    print(f"Number of connected components: {len(components)}")
    for i, comp in enumerate(components):
        if i < 5:  # Show only first 5 components
            print(f"  Component {i+1}: {len(comp)} nodes")
    
    # Clustering coefficient
    try:
        avg_clustering = nx.average_clustering(G)
        print(f"Average clustering coefficient: {avg_clustering:.4f}")
    except:
        print("Could not compute clustering coefficient")
    
    # Try to identify communities
    try:
        communities = nx.community.louvain_communities(G)
        print(f"Identified {len(communities)} communities using Louvain method")
        for i, community in enumerate(communities):
            if i < 5:  # Show only first 5 communities
                print(f"  Community {i+1}: {len(community)} nodes")
    except:
        print("Could not identify communities")
    
    # Visualize small graphs
    if n <= 500:
        plt.figure(figsize=(10, 10))
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx(G, pos=pos, with_labels=False, node_size=50, 
                         node_color='lightblue', edge_color='gray', alpha=0.8)
        plt.title(f"Solution Graph ({n} nodes, {m} edges)")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("solution_graph.png", dpi=300)
        print("Graph visualization saved to solution_graph.png")
    
    return G

def extract_recommended_seeds(solution_nodes, solution_degrees, total_degrees, n=20):
    """Extract recommended seed nodes based on analysis."""
    print("\n=== Recommended Seed Nodes ===")
    
    # Combine metrics: internal connectivity and total degree
    combined_score = {}
    for node in solution_nodes:
        internal_score = solution_degrees.get(node, 0) / max(solution_degrees.values()) if solution_degrees else 0
        total_score = total_degrees.get(node, 0) / max(total_degrees.values()) if total_degrees else 0
        # Weight internal connectivity more
        combined_score[node] = 0.7 * internal_score + 0.3 * total_score
    
    # Sort by combined score
    sorted_nodes = sorted(combined_score.items(), key=lambda x: x[1], reverse=True)
    
    print(f"Top {n} recommended seed nodes:")
    for i, (node, score) in enumerate(sorted_nodes[:n]):
        print(f"  {node}")
    
    # Create a seed file
    with open("recommended_seeds.txt", "w") as f:
        for node, _ in sorted_nodes[:n]:
            f.write(f"{node}\n")
    
    print(f"Recommended seeds saved to recommended_seeds.txt")
    
    return [node for node, _ in sorted_nodes[:n]]

def map_solution_to_sequential_ids(solution_nodes, mapping_file):
    """Map solution nodes to sequential IDs if mapping file is available."""
    if not os.path.exists(mapping_file):
        print(f"Mapping file {mapping_file} not found, skipping ID mapping")
        return None
    
    print(f"\n=== Mapping to Sequential IDs ===")
    orig_to_seq = {}
    
    with open(mapping_file, 'r') as f:
        reader = csv.reader(f)
        # Skip header
        next(reader, None)
        
        for row in reader:
            if len(row) >= 2:
                try:
                    orig_id = int(row[0])
                    seq_id = int(row[1])
                    orig_to_seq[orig_id] = seq_id
                except ValueError:
                    continue
    
    # Map solution nodes to sequential IDs
    mapped_nodes = []
    unmapped_nodes = []
    
    for node in solution_nodes:
        if node in orig_to_seq:
            mapped_nodes.append(orig_to_seq[node])
        else:
            unmapped_nodes.append(node)
    
    print(f"Mapped {len(mapped_nodes)} nodes to sequential IDs")
    if unmapped_nodes:
        print(f"Could not map {len(unmapped_nodes)} nodes")
    
    if mapped_nodes:
        with open("solution_sequential_ids.txt", "w") as f:
            for node in mapped_nodes:
                f.write(f"{node}\n")
        print("Sequential IDs saved to solution_sequential_ids.txt")
    
    return mapped_nodes

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python extract_solution_insights.py <solution_file> [<edge_file>] [<mapping_file>]")
        print("  OR")
        print("  python extract_solution_insights.py --paste (to paste node IDs)")
        sys.exit(1)
    
    # Get solution nodes
    if sys.argv[1] == "--paste":
        print("Paste the solution nodes (one per line), then press Ctrl+D (Unix) or Ctrl+Z (Windows) followed by Enter:")
        input_content = sys.stdin.read()
        solution_nodes = read_solution_from_input(input_content)
    else:
        solution_file = sys.argv[1]
        with open(solution_file, 'r') as f:
            content = f.read()
        solution_nodes = read_solution_from_input(content)
    
    print(f"Loaded {len(solution_nodes)} nodes from solution")
    
    # Optional edge file for deeper analysis
    edge_file = sys.argv[2] if len(sys.argv) > 2 else "data/flywire_edges.txt"
    
    if os.path.exists(edge_file):
        solution_connections, solution_degrees, total_degrees = find_solution_nodes_in_graph(solution_nodes, edge_file)
        G = analyze_solution_structure(solution_connections)
        extract_recommended_seeds(solution_nodes, solution_degrees, total_degrees)
    else:
        print(f"Edge file {edge_file} not found, skipping graph analysis")
        analyze_id_patterns(solution_nodes)
    
    # Optional mapping file to convert to sequential IDs
    mapping_file = sys.argv[3] if len(sys.argv) > 3 else "data/id_mapping.csv"
    map_solution_to_sequential_ids(solution_nodes, mapping_file)
    
    # Final recommendations
    print("\n=== Final Recommendations ===")
    print("1. Try increasing the number of seeds to 1000+")
    print("2. Use the recommended seed nodes identified above")
    print("3. Adjust the merging parameters to be more aggressive")
    print("4. Decrease the alpha parameter decay rate")
    print("5. Use as many threads as available for parallel processing")

if __name__ == "__main__":
    main()