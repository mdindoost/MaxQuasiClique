#!/usr/bin/env python3
"""
Convert large neuron IDs to sequential small integers for the MaxQuasiClique algorithm.
"""

import argparse
import csv
from collections import defaultdict

def convert_ids(input_file, output_file, mapping_file=None):
    """Convert large IDs to sequential integers and maintain a mapping."""
    # Dictionary to map original IDs to new sequential IDs
    id_mapping = {}
    next_id = 0
    
    print(f"Reading file: {input_file}")
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        # Write a header comment
        outfile.write(f"# Converted from {input_file} - Large IDs mapped to sequential integers\n")
        
        # Process each line
        line_count = 0
        for line in infile:
            line_count += 1
            
            # Skip comment lines
            if line.startswith('#'):
                continue
                
            # Parse the line
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    orig_src = parts[0]
                    orig_dst = parts[1]
                    
                    # Map IDs if not already mapped
                    if orig_src not in id_mapping:
                        id_mapping[orig_src] = next_id
                        next_id += 1
                    
                    if orig_dst not in id_mapping:
                        id_mapping[orig_dst] = next_id
                        next_id += 1
                    
                    # Write the new IDs
                    outfile.write(f"{id_mapping[orig_src]} {id_mapping[orig_dst]}\n")
                    
                except Exception as e:
                    print(f"Error processing line {line_count}: {line.strip()} - {str(e)}")
                    continue
            
            # Print progress
            if line_count % 100000 == 0:
                print(f"Processed {line_count} lines, mapped {len(id_mapping)} unique IDs")
    
    print(f"Conversion complete. Mapped {len(id_mapping)} unique IDs.")
    
    # Save the mapping if requested
    if mapping_file:
        print(f"Writing ID mapping to {mapping_file}")
        with open(mapping_file, 'w') as mapfile:
            mapfile.write("original_id,sequential_id\n")
            for orig_id, seq_id in id_mapping.items():
                mapfile.write(f"{orig_id},{seq_id}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert large neuron IDs to sequential integers")
    parser.add_argument("--input", default="data/flywire_edges.txt", help="Input edge list file")
    parser.add_argument("--output", default="data/flywire_edges_converted.txt", help="Output edge list file")
    parser.add_argument("--mapping", default="data/id_mapping.csv", help="Output ID mapping file")
    args = parser.parse_args()
    
    convert_ids(args.input, args.output, args.mapping)