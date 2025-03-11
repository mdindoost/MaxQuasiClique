#!/usr/bin/env python3
"""
Run the MaxQuasiClique algorithm on the FlyWire connectome data and analyze the results.
"""

import subprocess
import argparse
import os
import time
import json
import sys
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def run_algorithm(executable, input_file, num_seeds, num_threads=None, timeout=None):
    """Run the MaxQuasiClique algorithm and return the output."""
    print(f"Running {executable} on {input_file} with {num_seeds} seeds...")
    
    thread_arg = f" {num_threads}" if num_threads else ""
    command = f"./{executable} {input_file} {num_seeds}{thread_arg}"
    
    start_time = time.time()
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True
        )
        stdout, stderr = process.communicate(timeout=timeout)
        
        if process.returncode != 0:
            print(f"Error running algorithm: {stderr}")
            return None
        
        return stdout
    except subprocess.TimeoutExpired:
        process.kill()
        print(f"Timeout expired after {timeout} seconds")
        return None
    finally:
        end_time = time.time()
        print(f"Execution time: {end_time - start_time:.2f} seconds")

def parse_solution(solution_file):
    """Parse the solution file containing vertex IDs."""
    if not os.path.exists(solution_file):
        print(f"Error: Solution file {solution_file} not found")
        return None
    
    with open(solution_file, 'r') as f:
        vertices = [int(line.strip()) for line in f if line.strip() and not line.startswith('#')]
    
    print(f"Loaded solution with {len(vertices)} vertices")
    return vertices

def build_solution_graph(edge_list_file, solution_vertices):
    """Build a graph from the edge list file, filtered to the solution vertices."""
    G = nx.Graph()
    G.add_nodes_from(solution_vertices)
    
    # Create a set for faster lookups
    vertex_set = set(solution_vertices)
    
    # Read edges from file and add those connecting vertices in the solution
    with open(edge_list_file, 'r') as f:
        for i, line in enumerate(f):
            if i == 0 and line.startswith('#'):
                continue  # Skip comment line
            
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    u = int(parts[0])
                    v = int(parts[1])
                    
                    if u in vertex_set and v in vertex_set:
                        G.add_edge(u, v)
                except ValueError:
                    continue
            
            # Progress reporting
            if i % 1000000 == 0 and i > 0:
                print(f"Processed {i} edges while building solution graph")
    
    return G

def analyze_solution(G, output_prefix):
    """Analyze properties of the solution graph."""
    # Basic metrics
    num_vertices = G.number_of_nodes()
    num_edges = G.number_of_edges()
    possible_edges = (num_vertices * (num_vertices - 1)) // 2
    density = num_edges / possible_edges if possible_edges > 0 else 0
    
    # Additional network metrics
    try:
        avg_degree = sum(dict(G.degree()).values()) / num_vertices if num_vertices > 0 else 0
    except:
        avg_degree = 0
        print("Warning: Could not compute average degree")
    
    try:
        avg_clustering = nx.average_clustering(G)
    except:
        avg_clustering = 0
        print("Warning: Could not compute average clustering coefficient")
    
    # Compute connected components
    try:
        components = list(nx.connected_components(G))
        largest_component = max(components, key=len) if components else set()
        largest_component_size = len(largest_component)
    except:
        components = []
        largest_component_size = 0
        print("Warning: Could not compute connected components")
    
    # Compute diameter of largest component (if not too large)
    diameter = float('inf')
    if largest_component_size > 0 and largest_component_size <= 1000:
        try:
            largest_component_graph = G.subgraph(largest_component)
            diameter = nx.diameter(largest_component_graph)
        except nx.NetworkXError:
            diameter = float('inf')  # Not connected
        except:
            print("Warning: Could not compute diameter")
    
    # Store results
    results = {
        "num_vertices": num_vertices,
        "num_edges": num_edges,
        "possible_edges": possible_edges,
        "density": density,
        "avg_degree": avg_degree,
        "avg_clustering": avg_clustering,
        "num_components": len(components) if 'components' in locals() else 0,
        "largest_component_size": largest_component_size,
        "diameter": diameter if diameter != float('inf') else "N/A",
        "is_quasi_clique": num_edges > possible_edges / 2
    }
    
    # Print analysis
    print("\n=== Solution Analysis ===")
    print(f"Vertices: {num_vertices}")
    print(f"Edges: {num_edges}/{possible_edges}")
    print(f"Density: {density:.4f}")
    print(f"Average degree: {avg_degree:.2f}")
    print(f"Average clustering coefficient: {avg_clustering:.4f}")
    print(f"Number of connected components: {results['num_components']}")
    print(f"Largest component size: {largest_component_size}")
    print(f"Diameter of largest component: {results['diameter']}")
    print(f"Is quasi-clique: {'Yes' if results['is_quasi_clique'] else 'No'}")
    
    # Save results
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    with open(f"{output_prefix}_analysis.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def visualize_solution(G, output_prefix):
    """Create visualizations of the solution graph."""
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    
    # Plot the graph
    plt.figure(figsize=(12, 10))
    
    # Limit visualization for very large graphs
    if G.number_of_nodes() > 1000:
        print("Graph too large for full visualization, plotting a sample...")
        sampled_nodes = list(G.nodes())[:1000]  # Take the first 1000 nodes
        G = G.subgraph(sampled_nodes)
    
    # Use different layout algorithms based on graph size
    try:
        if G.number_of_nodes() < 100:
            pos = nx.spring_layout(G, seed=42)
        elif G.number_of_nodes() < 500:
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.random_layout(G, seed=42)
        
        # Draw the network
        nx.draw_networkx(
            G,
            pos=pos,
            node_color='lightblue',
            node_size=50,
            edge_color='gray',
            with_labels=False,
            alpha=0.8
        )
        
        plt.title(f"Quasi-Clique Solution ({G.number_of_nodes()} vertices, {G.number_of_edges()} edges)")
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_graph.png", dpi=300)
        plt.close()
    except Exception as e:
        print(f"Warning: Could not create graph visualization: {str(e)}")
    
    try:
        # Plot degree distribution
        plt.figure(figsize=(10, 6))
        degrees = [d for n, d in G.degree()]
        plt.hist(degrees, bins=min(20, len(set(degrees))), alpha=0.7, color='blue')
        plt.xlabel('Degree')
        plt.ylabel('Frequency')
        plt.title('Degree Distribution')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_prefix}_degree_dist.png", dpi=300)
        plt.close()
    except Exception as e:
        print(f"Warning: Could not create degree distribution plot: {str(e)}")
    
    try:
        # Plot clustering coefficient distribution
        plt.figure(figsize=(10, 6))
        clustering = nx.clustering(G)
        clustering_values = list(clustering.values())
        plt.hist(clustering_values, bins=min(20, len(set(clustering_values))), alpha=0.7, color='green')
        plt.xlabel('Clustering Coefficient')
        plt.ylabel('Frequency')
        plt.title('Clustering Coefficient Distribution')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_prefix}_clustering_dist.png", dpi=300)
        plt.close()
    except Exception as e:
        print(f"Warning: Could not create clustering coefficient distribution plot: {str(e)}")
    
    print(f"Visualizations saved to {output_prefix}_*.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run and analyze the MaxQuasiClique algorithm on FlyWire data")
    parser.add_argument("--executable", type=str, default="build/optimized_solver",
                        help="Path to the solver executable")
    parser.add_argument("--input", type=str, default="data/flywire_edges.txt",
                        help="Input edge list file")
    parser.add_argument("--seeds", type=int, default=50,
                        help="Number of seeds to use")
    parser.add_argument("--threads", type=int, default=8,
                        help="Number of threads to use")
    parser.add_argument("--solution", type=str, default="solution.txt",
                        help="Path to solution file (if already generated)")
    parser.add_argument("--skip-run", action="store_true",
                        help="Skip running the algorithm and only analyze existing solution")
    parser.add_argument("--output-prefix", type=str, default="results/flywire_solution",
                        help="Prefix for output files")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="Timeout in seconds")
    args = parser.parse_args()
    
    # Run the algorithm if not skipped
    if not args.skip_run:
        output = run_algorithm(
            args.executable, 
            args.input, 
            args.seeds, 
            args.threads, 
            args.timeout
        )
        if output:
            print("Algorithm completed successfully")
    
    # Parse the solution
    solution_vertices = parse_solution(args.solution)
    if not solution_vertices:
        print("No solution found. Exiting.")
        sys.exit(1)
    
    # Build the solution graph
    solution_graph = build_solution_graph(args.input, solution_vertices)
    print(f"Built solution graph with {solution_graph.number_of_nodes()} vertices and {solution_graph.number_of_edges()} edges")
    
    # Analyze the solution
    analysis = analyze_solution(solution_graph, args.output_prefix)
    
    # Visualize the solution
    visualize_solution(solution_graph, args.output_prefix)
    
    print("\nAnalysis complete!")