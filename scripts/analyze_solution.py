#!/usr/bin/env python3
"""
Analyze a known solution for the MaxQuasiClique challenge on the FlyWire connectome data.
This script reads the solution, identifies its structure, and provides helpful insights.
"""

import sys
import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict, Counter

def load_solution(solution_file):
    """Load the solution nodes from a file."""
    solution_nodes = []
    with open(solution_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                try:
                    solution_nodes.append(int(line))
                except ValueError:
                    print(f"Warning: Invalid node ID in solution file: {line}")
    
    print(f"Loaded {len(solution_nodes)} nodes from solution")
    return solution_nodes

def load_edge_list(edge_file):
    """Load the edge list from a file."""
    edges = []
    node_set = set()
    
    print(f"Loading edges from {edge_file}...")
    line_count = 0
    
    with open(edge_file, 'r') as f:
        for line in f:
            line_count += 1
            if line_count % 1000000 == 0:
                print(f"  Processed {line_count:,} lines...")
            
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        u = int(parts[0])
                        v = int(parts[1])
                        edges.append((u, v))
                        node_set.add(u)
                        node_set.add(v)
                    except ValueError:
                        continue
    
    print(f"Loaded {len(edges):,} edges connecting {len(node_set):,} nodes")
    return edges, node_set

def load_id_mapping(mapping_file):
    """Load the mapping between original and sequential IDs."""
    orig_to_seq = {}
    seq_to_orig = {}
    
    with open(mapping_file, 'r') as f:
        reader = csv.reader(f)
        # Skip header
        next(reader, None)
        
        for row in reader:
            if len(row) >= 2:
                orig_id = int(row[0])
                seq_id = int(row[1])
                orig_to_seq[orig_id] = seq_id
                seq_to_orig[seq_id] = orig_id
    
    print(f"Loaded ID mapping with {len(orig_to_seq):,} entries")
    return orig_to_seq, seq_to_orig

def build_solution_graph(edges, solution_nodes):
    """Build a graph of the solution."""
    G = nx.Graph()
    G.add_nodes_from(solution_nodes)
    
    # Create a set for faster lookups
    solution_set = set(solution_nodes)
    
    # Add edges between solution nodes
    for u, v in edges:
        if u in solution_set and v in solution_set:
            G.add_edge(u, v)
    
    return G

def analyze_solution_structure(G):
    """Analyze the structure of the solution."""
    print("\n=== Solution Structure Analysis ===")
    
    # Basic stats
    n = G.number_of_nodes()
    m = G.number_of_edges()
    possible_edges = n * (n - 1) // 2
    density = m / possible_edges if possible_edges > 0 else 0
    
    print(f"Nodes: {n}")
    print(f"Edges: {m:,} / {possible_edges:,} possible")
    print(f"Density: {density:.4f}")
    print(f"Is valid quasi-clique? {'Yes' if density > 0.5 else 'No'}")
    
    # Degree distribution
    degrees = [d for _, d in G.degree()]
    avg_degree = sum(degrees) / len(degrees) if degrees else 0
    min_degree = min(degrees) if degrees else 0
    max_degree = max(degrees) if degrees else 0
    
    print(f"Average degree: {avg_degree:.2f}")
    print(f"Min degree: {min_degree}")
    print(f"Max degree: {max_degree}")
    
    # Clustering coefficients
    try:
        avg_clustering = nx.average_clustering(G)
        print(f"Average clustering coefficient: {avg_clustering:.4f}")
    except:
        print("Could not compute clustering coefficient")
    
    # Connected components
    components = list(nx.connected_components(G))
    print(f"Number of connected components: {len(components)}")
    for i, comp in enumerate(components):
        print(f"  Component {i+1}: {len(comp)} nodes")
    
    # Diameter (of largest component)
    largest_component = max(components, key=len) if components else set()
    largest_component_graph = G.subgraph(largest_component)
    
    try:
        diameter = nx.diameter(largest_component_graph)
        print(f"Diameter of largest component: {diameter}")
    except (nx.NetworkXError, nx.NetworkXNotImplemented):
        print("Could not compute diameter (not connected or too large)")
    
    return {
        "nodes": n,
        "edges": m,
        "possible_edges": possible_edges,
        "density": density,
        "avg_degree": avg_degree,
        "min_degree": min_degree,
        "max_degree": max_degree,
        "components": len(components)
    }

def analyze_communities(G, solution_nodes, community_map=None):
    """Analyze the community structure of the solution."""
    print("\n=== Community Analysis ===")
    
    if community_map is None:
        # If no community map is provided, detect communities
        print("No community mapping provided, detecting communities...")
        try:
            communities = nx.community.louvain_communities(G)
            community_map = {}
            for i, community in enumerate(communities):
                for node in community:
                    community_map[node] = i
        except:
            print("Could not detect communities")
            return
    
    # Count nodes per community
    community_counts = defaultdict(int)
    for node in solution_nodes:
        if node in community_map:
            community_counts[community_map[node]] += 1
    
    # Sort communities by size
    sorted_communities = sorted(community_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"Solution spans {len(community_counts)} communities")
    print("\nLargest communities in solution:")
    for i, (comm_id, count) in enumerate(sorted_communities[:10]):
        print(f"  Community {comm_id}: {count} nodes ({count/len(solution_nodes)*100:.1f}%)")
    
    # Check for bridges between communities
    community_bridges = defaultdict(int)
    for u, v in G.edges():
        if u in community_map and v in community_map:
            if community_map[u] != community_map[v]:
                pair = tuple(sorted([community_map[u], community_map[v]]))
                community_bridges[pair] += 1
    
    print("\nBridges between communities:")
    sorted_bridges = sorted(community_bridges.items(), key=lambda x: x[1], reverse=True)
    for i, (comm_pair, count) in enumerate(sorted_bridges[:10]):
        print(f"  Between communities {comm_pair[0]} and {comm_pair[1]}: {count} edges")
    
    return community_counts

def analyze_node_importance(G, solution_nodes):
    """Analyze which nodes are most important in the solution."""
    print("\n=== Node Importance Analysis ===")
    
    # Calculate centrality measures
    degree_centrality = nx.degree_centrality(G)
    
    try:
        betweenness_centrality = nx.betweenness_centrality(G, k=min(100, len(solution_nodes)))
    except:
        betweenness_centrality = {}
        print("Could not compute betweenness centrality")
    
    try:
        closeness_centrality = nx.closeness_centrality(G)
    except:
        closeness_centrality = {}
        print("Could not compute closeness centrality")
    
    # Find top nodes by different measures
    top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:20]
    top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:20]
    top_closeness = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:20]
    
    print("\nTop nodes by degree centrality:")
    for node, value in top_degree[:10]:
        print(f"  Node {node}: {value:.4f}")
    
    if betweenness_centrality:
        print("\nTop nodes by betweenness centrality (potential bridges):")
        for node, value in top_betweenness[:10]:
            print(f"  Node {node}: {value:.4f}")
    
    if closeness_centrality:
        print("\nTop nodes by closeness centrality:")
        for node, value in top_closeness[:10]:
            print(f"  Node {node}: {value:.4f}")
    
    # Find potential seed nodes
    potential_seeds = set()
    for node, _ in top_degree[:5]:
        potential_seeds.add(node)
    for node, _ in top_betweenness[:5]:
        potential_seeds.add(node)
    for node, _ in top_closeness[:5]:
        potential_seeds.add(node)
    
    print("\nRecommended seed nodes for algorithm:")
    for node in potential_seeds:
        print(f"  {node}")
    
    return potential_seeds

def visualize_solution(G, output_file="solution_visualization.png"):
    """Create a visualization of the solution graph."""
    print("\n=== Creating Visualization ===")
    
    # Limit size for visualization
    if G.number_of_nodes() > 500:
        print(f"Graph too large ({G.number_of_nodes()} nodes), visualizing a sample of 500 nodes")
        nodes = list(G.nodes())[:500]
        G = G.subgraph(nodes)
    
    plt.figure(figsize=(12, 12))
    
    # Choose layout based on graph size
    if G.number_of_nodes() < 100:
        pos = nx.spring_layout(G, seed=42)
    else:
        pos = nx.kamada_kawai_layout(G)
    
    # Draw the graph
    nx.draw_networkx(
        G, 
        pos=pos,
        node_color='lightblue',
        edge_color='gray',
        alpha=0.8,
        with_labels=False,
        node_size=50
    )
    
    plt.title(f"Solution Graph ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Visualization saved to {output_file}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_solution.py <solution_file> [<edge_file>] [<id_mapping_file>]")
        print("\nExample:")
        print("  python analyze_solution.py known_solution.txt data/flywire_edges.txt data/id_mapping.csv")
        sys.exit(1)
    
    solution_file = sys.argv[1]
    edge_file = sys.argv[2] if len(sys.argv) > 2 else "data/flywire_edges.txt"
    mapping_file = sys.argv[3] if len(sys.argv) > 3 else "data/id_mapping.csv"
    
    # Load the solution
    solution_nodes = load_solution(solution_file)
    
    # Load the edge list
    edges, all_nodes = load_edge_list(edge_file)
    
    # Check if all solution nodes are in the graph
    solution_set = set(solution_nodes)
    missing_nodes = solution_set - all_nodes
    if missing_nodes:
        print(f"Warning: {len(missing_nodes)} solution nodes are not in the graph")
    
    # Build the solution graph
    G = build_solution_graph(edges, solution_nodes)
    print(f"Built solution graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Analyze the solution structure
    analyze_solution_structure(G)
    
    # Analyze communities 
    analyze_communities(G, solution_nodes)
    
    # Analyze node importance
    analyze_node_importance(G, solution_nodes)
    
    # Visualize the solution
    visualize_solution(G, "solution_visualization.png")
    
    # Generate recommendations
    print("\n=== Algorithm Improvement Recommendations ===")
    print("1. Use the recommended seed nodes identified above")
    print("2. Adjust the merging threshold to allow more aggressive merging between communities")
    print("3. Increase the number of seeds to at least 1000 for better exploration")
    print("4. Modify the alpha parameter to decrease more slowly for larger solutions")
    print("5. Add a targeted refinement phase to identify and connect potential bridges between communities")

if __name__ == "__main__":
    main()