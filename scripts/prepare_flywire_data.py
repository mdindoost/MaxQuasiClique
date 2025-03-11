#!/usr/bin/env python3
"""
Prepare FlyWire connectome data for the MaxQuasiClique algorithm.
This script converts the edge.csv file to the format expected by our algorithm.
"""

import csv
import argparse
import os
import sys

def convert_csv_to_edgelist(input_file, output_file):
    """Convert a CSV file to a simple edge list format."""
    print(f"Converting {input_file} to edge list format...")
    
    # Check if the file exists
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} does not exist")
        return False
    
    # Determine total number of lines for progress reporting
    total_lines = 0
    try:
        with open(input_file, 'r') as f:
            for _ in f:
                total_lines += 1
        print(f"Processing {total_lines} lines...")
    except Exception as e:
        print(f"Error counting lines: {str(e)}")
        total_lines = "unknown"
    
    # Process the file
    edge_count = 0
    try:
        with open(input_file, 'r') as csv_file, open(output_file, 'w') as out_file:
            # Write header comment
            out_file.write(f"# FlyWire connectome edge list converted from {input_file}\n")
            
            # Check if the file has a header
            csv_reader = csv.reader(csv_file)
            first_row = next(csv_reader, None)
            
            if first_row and any(h.lower() in ['source', 'target', 'from', 'to', 'node1', 'node2'] for h in first_row):
                # File has a header
                print(f"Detected header: {first_row}")
                
                # Find source and target column indices
                source_idx = None
                target_idx = None
                
                for i, header in enumerate(first_row):
                    header_lower = header.lower()
                    if header_lower in ['source', 'from', 'node1']:
                        source_idx = i
                    elif header_lower in ['target', 'to', 'node2']:
                        target_idx = i
                
                if source_idx is None or target_idx is None:
                    # If not found by name, assume first two columns
                    print("Could not identify source and target columns by name, assuming first two columns")
                    source_idx = 0
                    target_idx = 1
                
                print(f"Using columns: {first_row[source_idx]} (source) and {first_row[target_idx]} (target)")
                
                # Process the remaining rows
                csv_file.seek(0)  # Go back to the beginning of the file
                next(csv_reader)  # Skip the header
                
                for i, row in enumerate(csv_reader):
                    if len(row) > max(source_idx, target_idx):
                        try:
                            source = row[source_idx]
                            target = row[target_idx]
                            out_file.write(f"{source} {target}\n")
                            edge_count += 1
                        except (ValueError, IndexError) as e:
                            print(f"Error processing row {i}: {e}")
                    
                    # Progress reporting
                    if i % 100000 == 0 and i > 0:
                        print(f"Processed {i} rows, {edge_count} edges extracted")
            else:
                # No header - assume first two columns are source and target
                print("No header detected, assuming first two columns are source and target")
                
                # If first_row was not a header, write it
                if first_row and len(first_row) >= 2:
                    try:
                        source = first_row[0]
                        target = first_row[1]
                        out_file.write(f"{source} {target}\n")
                        edge_count += 1
                    except ValueError:
                        print("Warning: First row doesn't contain valid IDs, skipping")
                
                # Process the rest of the file
                line_count = 1
                for row in csv_reader:
                    if len(row) >= 2:
                        try:
                            source = row[0]
                            target = row[1]
                            out_file.write(f"{source} {target}\n")
                            edge_count += 1
                        except ValueError:
                            pass  # Skip rows with non-numeric data
                    
                    # Progress reporting
                    line_count += 1
                    if line_count % 100000 == 0:
                        print(f"Processed {line_count} rows, {edge_count} edges extracted")
        
        print(f"Conversion complete. Wrote {edge_count} edges to {output_file}")
        return True
    
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert FlyWire edge.csv to edge list format")
    parser.add_argument("--input", type=str, default="data/edges.csv",
                        help="Input CSV file path (default: data/edges.csv)")
    parser.add_argument("--output", type=str, default="data/flywire_edges.txt",
                        help="Output edge list file path (default: data/flywire_edges.txt)")
    args = parser.parse_args()
    
    success = convert_csv_to_edgelist(args.input, args.output)
    if success:
        print("\nNext steps:")
        print("1. Run the algorithm on the converted data:")
        print(f"   ./build/optimized_solver {args.output} 50 8")
        print("2. Or for the basic implementation (may be slow for large graphs):")
        print(f"   ./build/max_quasi_clique {args.output} 10")
    else:
        sys.exit(1)