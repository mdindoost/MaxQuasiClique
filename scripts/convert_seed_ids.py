#!/usr/bin/env python3
"""
Convert original FlyWire neuron IDs to sequential IDs for seeding the algorithm.
"""

import argparse
import csv

def convert_seed_ids(original_ids_file, mapping_file, output_file):
    """
    Convert original IDs to sequential IDs using the mapping file.
    """
    # Load the ID mapping
    id_mapping = {}
    with open(mapping_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        
        for row in reader:
            if len(row) >= 2:
                original_id = row[0]
                sequential_id = int(row[1])
                id_mapping[original_id] = sequential_id
    
    print(f"Loaded mapping with {len(id_mapping)} IDs")
    
    # Load original IDs
    original_ids = []
    with open(original_ids_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                original_ids.append(line)
    
    print(f"Loaded {len(original_ids)} original IDs from {original_ids_file}")
    
    # Convert to sequential IDs
    sequential_ids = []
    not_found = 0
    
    for original_id in original_ids:
        if original_id in id_mapping:
            sequential_ids.append(id_mapping[original_id])
        else:
            not_found += 1
    
    if not_found > 0:
        print(f"Warning: {not_found} original IDs not found in the mapping")
    
    print(f"Successfully mapped {len(sequential_ids)} IDs")
    
    # Write sequential IDs to output file
    with open(output_file, 'w') as f:
        f.write("# Sequential IDs converted from original FlyWire neuron IDs\n")
        for sequential_id in sequential_ids:
            f.write(f"{sequential_id}\n")
    
    print(f"Sequential IDs written to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert original FlyWire neuron IDs to sequential IDs")
    parser.add_argument("--input", default="data/known_solution.txt", help="Input file with original IDs")
    parser.add_argument("--mapping", default="data/id_mapping.csv", help="ID mapping file")
    parser.add_argument("--output", default="data/sequential_seeds.txt", help="Output file for sequential IDs")
    args = parser.parse_args()
    
    convert_seed_ids(args.input, args.mapping, args.output)