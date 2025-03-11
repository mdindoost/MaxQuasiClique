#!/usr/bin/env python3
"""
Generate a test graph that approximates some properties of connectome networks.
This creates a graph with community structure and specific connection patterns.
"""

import random
import argparse
import numpy as np

def generate_connectome_like_graph(num_neurons, output_file, seed=None):
    """Generate a graph that has some properties similar to connectome data."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    print(f"Generating graph with {num_neurons} neurons...")
    
    # Create communities (brain regions)
    num_regions = max(3, num_neurons // 100)
    regions = []
    neurons_per_region = {}
    
    # Assign neurons to regions with varying sizes
    remaining_neurons = num_neurons
    for i in range(num_regions - 1):
        # Each region gets a random proportion of remaining neurons
        size = max(5, int(random.betavariate(2, 2) * remaining_neurons))
        size = min(size, remaining_neurons - (num_regions - i - 1) * 5)
        regions.append(f"R{i}")
        neurons_per_region[f"R{i}"] = list(range(num_neurons - remaining_neurons, 
                                              num_neurons - remaining_neurons + size))
        remaining_neurons -= size
    
    # Last region gets the remainder
    regions.append(f"R{num_regions-1}")
    neurons_per_region[f"R{num_regions-1}"] = list(range(num_neurons - remaining_neurons, num_neurons))
    
    # Print region sizes
    for region, neurons in neurons_per_region.items():
        print(f"  {region}: {len(neurons)} neurons")
    
    # Create edges with higher probability within regions and lower between regions
    edges = set()
    
    # Parameters for connection probabilities
    within_region_base_p = 0.1  # Base probability for connections within a region
    between_region_base_p = 0.01  # Base probability for connections between regions
    
    # Add connections within regions (higher probability)
    for region, neurons in neurons_per_region.items():
        # Higher connectivity in smaller regions
        region_size = len(neurons)
        within_p = within_region_base_p * (1 + 1/np.sqrt(region_size))
        
        # Create rich-club structure within regions
        for i, u in enumerate(neurons):
            # Neurons with lower IDs have more connections (rich-club property)
            hub_factor = 1 + 2 * (1 - i/region_size)**2
            
            for j, v in enumerate(neurons):
                if i < j:  # Only consider each pair once
                    # Connection probability increases for hub-to-hub connections
                    p = within_p * hub_factor * (1 + 2 * (1 - j/region_size)**2)
                    p = min(p, 0.9)  # Cap probability
                    
                    if random.random() < p:
                        edges.add((u, v))
    
    # Add connections between regions (lower probability)
    for i, region1 in enumerate(regions):
        for j, region2 in enumerate(regions):
            if i < j:  # Only consider each pair once
                # Distance-based probability (farther regions less likely to connect)
                distance_factor = 1 / (1 + abs(i - j))
                between_p = between_region_base_p * distance_factor
                
                # Create connections between regions
                for u in neurons_per_region[region1]:
                    for v in neurons_per_region[region2]:
                        if random.random() < between_p:
                            edges.add((u, v))
    
    # Add a few dense clusters (potential quasi-cliques)
    num_clusters = max(2, num_neurons // 500)
    for _ in range(num_clusters):
        # Create a dense subgraph
        cluster_size = random.randint(10, min(50, num_neurons // 20))
        cluster_nodes = random.sample(range(num_neurons), cluster_size)
        
        # Add edges with high probability (0.6-0.8)
        p_dense = random.uniform(0.6, 0.8)
        for i, u in enumerate(cluster_nodes):
            for j, v in enumerate(cluster_nodes):
                if i < j and random.random() < p_dense:
                    edges.add((u, v))
    
    # Write edges to file
    with open(output_file, 'w') as f:
        f.write(f"# Test graph with {num_neurons} neurons and {len(edges)} edges\n")
        for u, v in sorted(edges):
            f.write(f"{u} {v}\n")
    
    print(f"Generated graph with {num_neurons} neurons and {len(edges)} edges")
    print(f"Saved to {output_file}")
    
    # Calculate and print some statistics
    avg_degree = 2 * len(edges) / num_neurons
    print(f"Average degree: {avg_degree:.2f}")
    
    # Estimate clustering coefficient (expensive to compute exactly)
    sample_size = min(500, num_neurons)
    sample_nodes = random.sample(range(num_neurons), sample_size)
    
    # Create adjacency list for samples
    adj_list = {node: set() for node in sample_nodes}
    for u, v in edges:
        if u in sample_nodes:
            adj_list[u].add(v)
        if v in sample_nodes:
            adj_list[v].add(u)
    
    # Compute average clustering coefficient
    clustering_sum = 0
    for node in sample_nodes:
        neighbors = adj_list[node]
        k = len(neighbors)
        if k < 2:
            continue
        
        # Count connections between neighbors
        connections = 0
        neighbors_list = list(neighbors)
        for i, u in enumerate(neighbors_list):
            for v in neighbors_list[i+1:]:
                if (u, v) in edges or (v, u) in edges:
                    connections += 1
        
        # Calculate local clustering coefficient
        possible_connections = k * (k - 1) / 2
        clustering = connections / possible_connections if possible_connections > 0 else 0
        clustering_sum += clustering
    
    avg_clustering = clustering_sum / len(sample_nodes)
    print(f"Estimated average clustering coefficient (from {sample_size} samples): {avg_clustering:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a test graph with connectome-like properties")
    parser.add_argument("--neurons", type=int, default=1000, help="Number of neurons (vertices)")
    parser.add_argument("--output", type=str, default="test_graph.txt", help="Output file path")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    generate_connectome_like_graph(args.neurons, args.output, args.seed)