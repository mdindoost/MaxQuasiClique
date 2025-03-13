import csv

def map_ids(clique_file, mapping_file, output_file=None):
    """
    Reads a list of node IDs from `clique_file` (one per line).
    Reads (original_id, sequential_id) pairs from `mapping_file`.
    Outputs or prints the corresponding sequential IDs for the clique nodes.
    """
    # QUESTION: 
    # 1) Are all IDs guaranteed to exist in the mapping file? 
    #    - If not, do we skip or raise an error?
    # 2) Are IDs guaranteed to be integers in both files, or should we treat them as strings?
    #
    # For now, let's assume all IDs are integers, and any missing mapping means we'll print a warning.

    # 1. Read the mapping into a dictionary
    original_to_sequential = {}
    with open(mapping_file, 'r', newline='') as mf:
        reader = csv.DictReader(mf)
        for row in reader:
            # Convert to int if your IDs are integers
            orig_id = int(row['original_id'])
            seq_id = int(row['sequential_id'])
            original_to_sequential[orig_id] = seq_id

    # 2. Read the clique file and map each ID
    mapped_ids = []
    with open(clique_file, 'r') as cf:
        for line in cf:
            line = line.strip()
            if not line:
                continue
            # Convert to int if your IDs are integers
            node_id = int(line)
            if node_id in original_to_sequential:
                mapped_ids.append(original_to_sequential[node_id])
            else:
                # If not found, decide how to handle
                print(f"Warning: ID {node_id} not found in mapping file.")
    
    # 3. Print or save the mapped IDs
    if output_file:
        with open(output_file, 'w') as out:
            for mid in mapped_ids:
                out.write(str(mid) + "\n")
        print(f"Mapped IDs saved to {output_file}")
    else:
        print("Mapped IDs (in order):")
        for mid in mapped_ids:
            print(mid)

# Example usage:
if __name__ == "__main__":
    clique_file = "max_clique_submission_benchmark_162.csv"
    mapping_file = "id_mapping.csv"
    output_file = "mapped_benchmark_162_ids.txt"   # or None if you just want to print

    map_ids(clique_file, mapping_file, output_file)
