# Save this as convert_simple.py
with open('data/edges.csv', 'r') as infile, open('data/simple_edges.txt', 'w') as outfile:
    for line in infile:
        if line.startswith('#'):  # Skip comments
            continue
        parts = line.strip().split(',')
        if len(parts) >= 2:
            try:
                source = int(parts[0])
                target = int(parts[1])
                outfile.write(f"{source} {target}\n")
            except ValueError:
                # Skip header or non-numeric lines
                pass