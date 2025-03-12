#!/usr/bin/env python3
"""
Generate a seed file for the two-phase algorithm based on the known solution.
"""

import sys
import os
import csv

def create_seed_file(input_content, output_file="known_solution_seeds.txt", mapping_file="data/id_mapping.csv"):
    """Create a seed file from the input content, optionally mapping to sequential IDs."""
    # Extract node IDs
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
    
    print(f"Found {len(solution_nodes)} node IDs")
    
    # If mapping file exists, map to sequential IDs
    if os.path.exists(mapping_file):
        print(f"Mapping node IDs using {mapping_file}...")
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
        
        # Map nodes
        mapped_nodes = []
        unmapped_nodes = []
        
        for node in solution_nodes:
            if node in orig_to_seq:
                mapped_nodes.append(orig_to_seq[node])
            else:
                unmapped_nodes.append(node)
        
        print(f"Mapped {len(mapped_nodes)} nodes to sequential IDs")
        if unmapped_nodes:
            print(f"Warning: Could not map {len(unmapped_nodes)} nodes")
            print(f"First few unmapped nodes: {unmapped_nodes[:5]}")
        
        # Write mapped nodes to seed file
        with open(output_file, 'w') as f:
            for node in mapped_nodes:
                f.write(f"{node}\n")
        
        print(f"Wrote {len(mapped_nodes)} seed nodes to {output_file}")
    else:
        # No mapping available, write original IDs
        print(f"No mapping file found, using original node IDs")
        with open(output_file, 'w') as f:
            for node in solution_nodes:
                f.write(f"{node}\n")
        
        print(f"Wrote {len(solution_nodes)} seed nodes to {output_file}")

def main():
    # Check if input is provided as a file or should be read from stdin
    if len(sys.argv) > 1:
        # Read from file
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "known_solution_seeds.txt"
        mapping_file = sys.argv[3] if len(sys.argv) > 3 else "data/id_mapping.csv"
        
        with open(input_file, 'r') as f:
            content = f.read()
        
        create_seed_file(content, output_file, mapping_file)
    else:
        # Read from stdin
        print("Paste the known solution nodes (one per line), then press Ctrl+D (Unix) or Ctrl+Z (Windows) followed by Enter:")
        content = sys.stdin.read()
        create_seed_file(content)

if __name__ == "__main__":
    main()