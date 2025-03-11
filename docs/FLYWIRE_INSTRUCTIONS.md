# Running MaxQuasiClique on FlyWire Connectome Data

This guide provides step-by-step instructions for running the MaxQuasiClique algorithm on the FlyWire connectome data.

## Prerequisites

Ensure you have the following installed:
- C++17 compatible compiler (GCC 7+, Clang 5+, or MSVC 2017+)
- Python 3.6+ with the following packages:
  - networkx
  - matplotlib
  - numpy
  - tabulate
  - tqdm
- Make or CMake for building

## Data Preparation

1. Place your FlyWire edge.csv file in the `data` directory:
   ```
   mkdir -p data
   cp /path/to/edge.csv data/
   ```

2. The edge.csv file should contain the connectivity information between neurons. It should have at minimum two columns for source and target neuron IDs.

## Automated Execution

For the simplest approach, use the provided shell script:

```bash
# Make the script executable
chmod +x run_flywire.sh

# Run with default settings
./run_flywire.sh

# Or with custom parameters
./run_flywire.sh --input data/edge.csv --seeds 50 --threads 8 --timeout 3600
```

The script will:
1. Build the necessary executables
2. Convert the edge.csv file to the required format
3. Run the algorithm
4. Analyze and visualize the results

## Manual Execution

If you prefer to run each step manually:

### 1. Build the executables

```bash
# Using Make
make optimized_solver

# Or using CMake
mkdir -p build
cd build
cmake ..
make
cd ..
```

### 2. Prepare the data

```bash
python3 prepare_flywire_data.py --input data/edge.csv --output data/flywire_edges.txt
```

### 3. Run the algorithm

```bash
./build/optimized_solver data/flywire_edges.txt 50 8
```

Parameters:
- First parameter: Path to the edge list file
- Second parameter: Number of seed vertices to try (50 recommended for large graphs)
- Third parameter: Number of threads to use (adjust based on your system)

### 4. Analyze the results

```bash
python3 run_flywire_analysis.py --executable build/optimized_solver \
    --input data/flywire_edges.txt \
    --seeds 50 \
    --threads 8 \
    --solution solution.txt \
    --skip-run \
    --output-prefix results/flywire_solution
```

## Interpreting Results

After running the algorithm, you'll find:

1. **Solution File**: `solution.txt` contains the vertex IDs in the found quasi-clique.

2. **Analysis File**: `results/flywire_solution_analysis.json` contains metrics about the solution:
   - `num_vertices`: Number of vertices in the solution
   - `num_edges`: Number of edges in the solution
   - `density`: Edge density (ratio of existing edges to possible edges)
   - `avg_degree`: Average degree of vertices in the solution
   - `avg_clustering`: Average clustering coefficient
   - `is_quasi_clique`: Boolean indicating if the solution is a valid quasi-clique

3. **Visualizations**:
   - `results/flywire_solution_graph.png`: Visualization of the quasi-clique structure
   - `results/flywire_solution_degree_dist.png`: Degree distribution histogram
   - `results/flywire_solution_clustering_dist.png`: Clustering coefficient distribution

## Performance Considerations

- For large connectome graphs (>100,000 vertices), the algorithm may take several hours to run.
- Adjust the number of threads based on your system's capabilities.
- If memory usage is a concern, consider reducing the number of seed vertices.
- The visualization may be limited for very large solutions (>1000 vertices).

## Troubleshooting

If you encounter issues:

1. **Memory Errors**: Try running with fewer threads or seeds.
2. **Timeout**: Increase the timeout value in the shell script.
3. **CSV Parsing Errors**: Check if your edge.csv file has the expected format. The script attempts to automatically detect column headers, but manual adjustment might be needed.
4. **No Solution Found**: The graph might not contain large quasi-cliques. Try adjusting parameters or check if the graph is correctly formatted.

## Further Analysis

For deeper analysis of the results:

1. **Neuron Connectivity**: Map vertex IDs back to neuron IDs in the original data.
2. **Community Structure**: Use the quasi-clique to identify functional communities in the connectome.
3. **Comparative Analysis**: Compare with other neuronal circuits or previous FlyWire analysis.