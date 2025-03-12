import csv
import gzip
import os

# Load edges
edges = set()
with open("data/edges.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        edges.add((int(row[0]), int(row[1])))
        edges.add((int(row[1]), int(row[0])))  # Add both directions since it's an undirected graph

# Load your solution
nodes = []
with open("results/flywire_solution_original_ids.txt", "r") as f:
    for line in f:
        if line.strip() and not line.startswith("#"):
            nodes.append(int(line.strip()))

# Verify no duplicate nodes
assert len(nodes) == len(set(nodes)), "Duplicate nodes detected"

# Verify that edge density is above 0.5
edge_count, pair_count = 0, 0
for i, n1 in enumerate(nodes):
    for n2 in nodes[i + 1:]:
        pair_count += 1
        if (n1, n2) in edges:
            edge_count += 1

density = edge_count / pair_count if pair_count > 0 else 0
print(f"Solution has {len(nodes)} nodes")
print(f"Edge count: {edge_count}")
print(f"Possible pairs: {pair_count}")
print(f"Density: {density:.4f}")
assert density > 0.5, "Insufficient density"

print("Verification passed! Your solution is a valid quasi-clique.")