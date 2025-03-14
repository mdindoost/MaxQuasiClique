def common_nodes(file1, file2):
    """
    Reads two files of node IDs (one per line) and prints how many are in common.
    Also prints the list of common nodes.
    """

    # QUESTION to clarify:
    # 1) Are these node IDs strings or integers?
    #    - If integers, we might want to convert them: int(line.strip()).
    #    - Otherwise, we keep them as strings.
    # 
    # 2) Should we ignore duplicates within the same file?
    #    - By using sets, duplicates in the same file are automatically ignored.
    #
    # 3) Do we need to handle empty lines or other formatting quirks?

    # For now, let's assume they are integers and ignore duplicates.
    with open(file1, 'r') as f1:
        nodes1 = set()
        for line in f1:
            line = line.strip()
            if line:
                # If node IDs are integers:
                node_id = int(line)
                nodes1.add(node_id)

    with open(file2, 'r') as f2:
        nodes2 = set()
        for line in f2:
            line = line.strip()
            if line:
                # If node IDs are integers:
                node_id = int(line)
                nodes2.add(node_id)

    # Compute the intersection
    common = nodes1.intersection(nodes2)
    print(f"Number of nodes in common: {len(common)}")

    # Print them if you want to see the actual overlap
    if common:
        print("Common node IDs:")
        for node in sorted(common):
            print(node)

# Example usage:
if __name__ == "__main__":
    # Update with the actual paths of your two files:
    file_a = "/home/mohammad/MaxQuasiClique/results/three175.csv"
    file_b = "solution_50_175.txt"

    common_nodes(file_a, file_b)
