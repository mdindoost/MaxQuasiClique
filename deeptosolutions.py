import csv
import os
import numpy as np
import networkx as nx
from collections import Counter, defaultdict

# Function to load ID mapping (long IDs to continuous IDs)
def load_id_mapping(mapping_file):
    id_mapping = {}
    with open(mapping_file, 'r') as f:
        reader = csv.reader(f)
        # Skip header row
        next(reader, None)
        
        for row in reader:
            if len(row) >= 2:
                original_id, continuous_id = row[0], row[1]
                try:
                    id_mapping[original_id] = int(continuous_id)
                except ValueError:
                    # Skip any non-integer values
                    continue
    return id_mapping

# Function to calculate diameter of a subgraph
def calculate_diameter(graph, nodes):
    subgraph = graph.subgraph(nodes)
    if not nx.is_connected(subgraph):
        return float('inf')  # Infinite diameter for disconnected graphs
    return nx.diameter(subgraph)

# Function to parse the input CSV file with mostly-cliques
def parse_mostly_cliques(input_file, id_mapping):
    subgraphs = []
    current_subgraph = None
    
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith("Mostly-clique"):
                if current_subgraph and len(current_subgraph['nodes']) > 0:
                    subgraphs.append(current_subgraph)
                current_subgraph = {'id': len(subgraphs) + 1, 'nodes': [], 'original_nodes': []}
            
            elif line.startswith("Number of edges:"):
                if current_subgraph:
                    current_subgraph['edges'] = int(line.split(":")[1].strip())
            
            elif line.startswith("k:"):
                if current_subgraph:
                    current_subgraph['k'] = int(line.split(":")[1].strip())
            
            elif line and current_subgraph and not line.startswith("Number") and not line.startswith("Mostly"):
                try:
                    original_id = line
                    current_subgraph['original_nodes'].append(original_id)
                    
                    # Convert to continuous ID if mapping exists
                    if original_id in id_mapping:
                        continuous_id = id_mapping[original_id]
                        current_subgraph['nodes'].append(continuous_id)
                    else:
                        print(f"Warning: No mapping found for ID {original_id}")
                except ValueError:
                    pass
    
    # Add the last subgraph
    if current_subgraph and len(current_subgraph['nodes']) > 0:
        subgraphs.append(current_subgraph)
    
    return subgraphs

# Function to analyze subgraphs and create graph data
def analyze_subgraphs(subgraphs, graph_file):
    # Create graph from edge file
    graph = nx.Graph()
    
    with open(graph_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                node1, node2 = int(parts[0]), int(parts[1])
                graph.add_edge(node1, node2)
    
    # Analyze each subgraph
    node_frequency = Counter()
    
    for sg in subgraphs:
        # Count node occurrences
        node_frequency.update(sg['nodes'])
        
        # Calculate density
        nodes = sg['nodes']
        n = len(nodes)
        max_edges = n * (n - 1) // 2
        sg['density'] = sg['edges'] / max_edges if max_edges > 0 else 0
        
        # Calculate diameter (this can be slow for large graphs)
        try:
            sg['diameter'] = calculate_diameter(graph, nodes)
        except Exception as e:
            print(f"Error calculating diameter for subgraph {sg['id']}: {e}")
            sg['diameter'] = -1
        
        # Find most common node in this subgraph
        if len(nodes) > 0:
            most_common_node = max(nodes, key=lambda n: graph.degree(n))
            sg['most_common_node'] = most_common_node
        else:
            sg['most_common_node'] = -1
    
    # Calculate node frequencies
    total_subgraphs = len(subgraphs)
    node_stats = []
    
    for node, freq in node_frequency.items():
        percentage = (freq / total_subgraphs) * 100
        node_stats.append({
            'node_id': node,
            'frequency': freq,
            'percentage': percentage
        })
    
    # Sort by frequency (descending)
    node_stats.sort(key=lambda x: x['frequency'], reverse=True)
    
    # Identify core nodes (>50% occurrence)
    core_nodes = [stats['node_id'] for stats in node_stats if stats['percentage'] > 50]
    
    return subgraphs, node_stats, core_nodes

# Function to write output files
def write_output_files(subgraphs, node_stats, core_nodes, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "converted_subgraphs"), exist_ok=True)
    
    # Write subgraph summary
    with open(os.path.join(output_dir, "subgraph_summary.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'Nodes', 'Edges', 'Density', 'Diameter', 'Most_Common_Node'])
        
        for sg in subgraphs:
            writer.writerow([
                sg['id'],
                len(sg['nodes']),
                sg['edges'],
                sg['density'],
                sg['diameter'],
                sg['most_common_node']
            ])
    
    # Write node frequency
    with open(os.path.join(output_dir, "node_frequency.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Node_ID', 'Frequency', 'Percentage_of_Subgraphs'])
        
        for stats in node_stats:
            writer.writerow([
                stats['node_id'],
                stats['frequency'],
                stats['percentage']
            ])
    
    # Write core nodes
    with open(os.path.join(output_dir, "core_nodes.txt"), 'w') as f:
        for node in core_nodes:
            f.write(f"{node}\n")
    
    # Write individual subgraphs
    for sg in subgraphs:
        with open(os.path.join(output_dir, "converted_subgraphs", f"subgraph_{sg['id']}.txt"), 'w') as f:
            for node in sg['nodes']:
                f.write(f"{node}\n")

# Main function
def main():
    # Configuration
    input_file = "data/144-175.csv"  # Your CSV file with mostly-cliques
    mapping_file = "data/id_mapping.csv"  # Your ID mapping file
    graph_file = "data/flywire_edges_converted.txt"  # Your edge list file with continuous IDs
    output_dir = "analysis_results"
    
    # Load ID mapping
    print("Loading ID mapping...")
    id_mapping = load_id_mapping(mapping_file)
    
    # Parse input file
    print("Parsing mostly-cliques file...")
    subgraphs = parse_mostly_cliques(input_file, id_mapping)
    print(f"Found {len(subgraphs)} subgraphs")
    
    # Analyze subgraphs
    print("Analyzing subgraphs...")
    subgraphs, node_stats, core_nodes = analyze_subgraphs(subgraphs, graph_file)
    
    # Write output files
    print("Writing output files...")
    write_output_files(subgraphs, node_stats, core_nodes, output_dir)
    
    print("Analysis complete!")
    print(f"Core nodes found: {len(core_nodes)}")
    if node_stats:
        print(f"Most frequent node: {node_stats[0]['node_id']} (appears in {node_stats[0]['percentage']:.2f}% of subgraphs)")

if __name__ == "__main__":
    main()