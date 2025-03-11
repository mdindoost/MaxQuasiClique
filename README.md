# MaxQuasiClique_Challenge-
 find large quasi-cliques in the undirected graph representing the FlyWire connectome.

# MaxQuasiClique

This repository contains implementations of the Density Gradient with Clustering Coefficient algorithm for finding large quasi-cliques in undirected graphs. The repository includes both a basic version and an optimized version with multi-threading support for handling larger graphs.

## Problem Description

A quasi-clique is a subgraph where more than half of all possible edges exist between its vertices. Formally, for a subgraph with k vertices, it must contain more than (k choose 2) / 2 edges.

The goal is to find the largest possible quasi-clique in a given graph.

## Algorithm

The implemented algorithm uses a density gradient approach combined with clustering coefficients to efficiently find large quasi-cliques:

1. Pre-compute clustering coefficients for all vertices
2. Sort vertices by potential (combination of degree and clustering)
3. Try multiple seed vertices as starting points
4. For each seed, grow the solution by iteratively adding vertices that:
   - Connect well to the current solution
   - Have good clustering properties
   - Maintain the quasi-clique property
5. Return the largest valid quasi-clique found

## Implementations

The repository contains two implementations of the algorithm:

1. **Basic Implementation** (`main.cpp`): A straightforward implementation of the algorithm that is easier to understand and modify.

2. **Optimized Implementation** (`optimized_solver.cpp`): An advanced implementation with the following optimizations:
   - Multi-threading support for parallel processing
   - Efficient memory usage with compressed data structures
   - Incremental computation of subgraph properties
   - Optimized edge counting and boundary vertex identification
   - Progress reporting and performance metrics

## Building the Project

### Prerequisites
- C++17 compatible compiler (GCC 7+, Clang 5+, or MSVC 2017+)
- Python 3.6+ (for generating test graphs and running benchmarks)
- Optional: matplotlib, numpy, and tabulate (for visualization and benchmarking)

### Build Instructions

**Using Make:**

```bash
# Clone the repository
git clone https://github.com/yourusername/MaxQuasiClique.git
cd MaxQuasiClique

# Build all targets
make all

# The executables will be in the build/ directory
```

**Using CMake:**

```bash
# Clone the repository
git clone https://github.com/yourusername/MaxQuasiClique.git
cd MaxQuasiClique

# Create build directory
mkdir build
cd build

# Configure and build
cmake ..
make
```

## Usage

### Basic Implementation

```bash
./build/max_quasi_clique <graph_file> [num_seeds]
```

Parameters:
- `graph_file`: Path to the input graph file (edge list format)
- `num_seeds` (optional): Number of seed vertices to try (default: 20)

### Optimized Implementation

```bash
./build/optimized_solver <graph_file> [num_seeds] [num_threads]
```

Parameters:
- `graph_file`: Path to the input graph file (edge list format)
- `num_seeds` (optional): Number of seed vertices to try (default: 20)
- `num_threads` (optional): Number of threads to use (default: use all available)

### Input Format

The input graph should be in edge list format, where each line represents an edge with space-separated vertex IDs:

```
v1 v2
v3 v4
...
```

### Output

The program outputs:
- Progress information during execution
- Final solution size
- Verification of the quasi-clique property
- List of vertices in the solution
- Execution time

The solution is also saved to a file named `solution.txt`.

## Examples

### Running the Basic Implementation

```bash
# Build and run the basic implementation on the sample graph
make run

# Or manually:
./build/max_quasi_clique sample_graph.txt 10
```

### Running the Optimized Implementation

```bash
# Build and run the optimized implementation on the sample graph with 4 threads
make run-opt

# Or manually:
./build/optimized_solver sample_graph.txt 10 4
```

### Generating Test Graphs

```bash
# Generate test graphs of different sizes
make graphs

# Or manually:
python generate_test_graph.py --neurons 1000 --output test_graph_1000.txt
```

### Running Benchmarks

```bash
# Run benchmarks comparing both implementations
make benchmark

# Or manually:
python benchmark.py --executable=build/max_quasi_clique --optimized=build/optimized_solver --sizes=1000,2000,5000 --seeds=5 --threads=4
```

## Performance Considerations

### Basic Implementation
- Pre-computing clustering coefficients can be time-consuming for very large graphs
- The algorithm tries multiple seed vertices sequentially
- Suitable for graphs up to a few thousand vertices

### Optimized Implementation
- Uses parallel processing for both clustering coefficient computation and seed expansion
- More memory-efficient data structures for large graphs
- Adaptive strategies for boundary vertex evaluation
- Fast subgraph property verification
- Progress reporting and performance metrics
- Suitable for graphs with tens of thousands of vertices

### General Tips
- The number of seeds to try can be adjusted to trade off between solution quality and runtime
- For very large graphs, consider using the optimized implementation with a limited number of seeds
- Increasing the number of threads generally improves performance up to the number of physical CPU cores

## License

[MIT License](LICENSE)
