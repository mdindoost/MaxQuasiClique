#!/usr/bin/env python3
"""
Map sequential IDs in solution back to original FlyWire neuron IDs.
"""

import argparse
import csv
import json

def load_id_mapping(mapping_file):
    """Load the ID mapping from file."""
    mapping = {}
    reverse_mapping = {}
    
    with open(mapping_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        
        for row in reader:
            if len(row) >= 2:
                original_id = row[0]
                sequential_id = int(row[1])
                mapping[original_id] = sequential_id
                reverse_mapping[sequential_id] = original_id
    
    print(f"Loaded mapping with {len(mapping)} IDs")
    return mapping, reverse_mapping

def map_solution_to_original_ids(solution_file, mapping_file, output_file):
    """Map sequential IDs in solution to original IDs."""
    # Load the ID mapping
    _, reverse_mapping = load_id_mapping(mapping_file)
    
    # Load the solution
    solution_ids = []
    with open(solution_file, 'r') as f:
        for line in f:
            try:
                sequential_id = int(line.strip())
                solution_ids.append(sequential_id)
            except ValueError:
                continue
    
    print(f"Loaded solution with {len(solution_ids)} vertices")
    
    # Map to original IDs
    original_ids = []
    for seq_id in solution_ids:
        if seq_id in reverse_mapping:
            original_ids.append(reverse_mapping[seq_id])
        else:
            print(f"Warning: No mapping found for ID {seq_id}")
    
    print(f"Mapped {len(original_ids)} vertices to original IDs")
    
    # Write the mapped solution
    with open(output_file, 'w') as f:
        f.write("# FlyWire neuron IDs in the quasi-clique solution\n")
        for orig_id in original_ids:
            f.write(f"{orig_id}\n")
    
    print(f"Wrote mapped solution to {output_file}")

def create_summary(solution_file, mapping_file, output_file):
    """Create a summary of the solution with original IDs."""
    # Load the ID mapping
    _, reverse_mapping = load_id_mapping(mapping_file)
    
    # Load the solution
    solution_ids = []
    with open(solution_file, 'r') as f:
        for line in f:
            try:
                sequential_id = int(line.strip())
                solution_ids.append(sequential_id)
            except ValueError:
                continue
    
    # Map to original IDs and create summary
    original_ids = [reverse_mapping.get(seq_id, "UNKNOWN") for seq_id in solution_ids]
    
    summary = {
        "solution_size": len(solution_ids),
        "mapped_ids": len([id for id in original_ids if id != "UNKNOWN"]),
        "original_neuron_ids": original_ids
    }
    
    # Write the summary
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Wrote summary to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map solution IDs to original FlyWire neuron IDs")
    parser.add_argument("--solution", default="solution.txt", help="Solution file with sequential IDs")
    parser.add_argument("--mapping", default="data/id_mapping.csv", help="ID mapping file")
    parser.add_argument("--output", default="results/flywire_solution_original_ids.txt", help="Output file for original IDs")
    parser.add_argument("--summary", default="results/flywire_solution_summary.json", help="Summary file")
    args = parser.parse_args()
    
    map_solution_to_original_ids(args.solution, args.mapping, args.output)
    create_summary(args.solution, args.mapping, args.summary)