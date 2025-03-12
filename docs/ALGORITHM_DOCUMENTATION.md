# MaxQuasiClique Algorithms for FlyWire Connectome Analysis

This document provides detailed information about the algorithms implemented in the MaxQuasiClique project for analyzing the FlyWire connectome data.

## Overview

The goal of this project is to find large quasi-cliques in the FlyWire connectome graph. A quasi-clique is a subgraph where more than half of all possible edges exist between its vertices. Formally, for a subgraph with k vertices, it must contain more than (k choose 2) / 2 edges.

Finding large quasi-cliques in neuronal networks can reveal densely connected groups of neurons that may form functional circuits in the brain.

## Implemented Algorithms

We have implemented two main algorithms for finding large quasi-cliques:

1. **Original Density Gradient Algorithm**
2. **Two-Phase Algorithm with Community Detection**

### 1. Original Density Gradient Algorithm

This algorithm uses a greedy approach with clustering coefficient awareness to find large quasi-cliques.

#### Key Components:

- **Clustering Coefficient Integration**: The algorithm considers not just the direct connections of neurons but also how interconnected their neighborhoods are.
- **Adaptive Weighting**: As the solution grows, the algorithm dynamically adjusts the balance between direct connections and clustering properties.
- **Multi-Seed Exploration**: Multiple starting points are used to explore diverse regions of the graph.
- **Parallel Processing**: The algorithm uses multi-threading to explore multiple seed points simultaneously.

#### Strengths:
- Simple, intuitive approach
- Efficient for medium-sized graphs
- Good at finding local dense structures

#### Limitations:
- May get stuck in local optima
- Doesn't explicitly consider global graph structure
- Limited ability to find quasi-cliques that span multiple dense regions

### 2. Two-Phase Algorithm with Community Detection

This advanced algorithm adds community detection and a two-phase approach to overcome the limitations of the original algorithm.

#### Key Components:

- **Phase 1: Community-Aware Multiple Region Discovery**
  - **Community Detection**: Uses the Louvain method to identify natural communities in the connectome
  - **Community-Aware Seed Selection**: Seeds are chosen proportionally from different communities and boundary regions
  - **Multiple Solution Finding**: Finds multiple candidate quasi-cliques in parallel

- **Phase 2: Solution Refinement and Merging**
  - **Solution Merging**: Compatible solutions are merged to create larger quasi-cliques
  - **Jaccard Similarity**: Used to identify promising merge candidates
  - **Iterative Merging**: Continues until no further improvements are possible

#### Strengths:
- Better exploration of the global graph structure
- Ability to find quasi-cliques that span multiple communities
- More likely to find larger solutions
- Well-suited for networks with community structure (like neuronal networks)

#### Limitations:
- More computationally intensive than the original algorithm
- Requires more memory
- More complex implementation

## Running the Algorithms

We provide a unified script `run_flywire_with_two_phase.sh` that can run either algorithm on the FlyWire connectome data.

### Prerequisites

- C++17 compatible compiler (GCC 7+, Clang 5+, or MSVC 2017+)
- Python 3.6+ with:
  - networkx
  - matplotlib
  - numpy
  - pandas
  - tqdm

### Usage

```bash
./run_flywire_with_two_phase.sh [options]
```

#### Options:

- `--input FILE`: Input CSV file path (default: data/edges.csv)
- `--seeds N`: Number of seed vertices to try (default: 100)
- `--threads N`: Number of threads to use (default: auto-detected)
- `--algorithm TYPE`: Algorithm to use, either "original" or "two-phase" (default: "two-phase")

### Examples:

```bash
# Run with two-phase algorithm (default)
./run_flywire_with_two_phase.sh

# Run with original algorithm
./run_flywire_with_two_phase.sh --algorithm original

# Run with specific number of seeds
./run_flywire_with_two_phase.sh --seeds 200

# Run with specific number of threads
./run_flywire_with_two_phase.sh --threads 8
```

## Understanding the Output

The algorithm produces several output files:

- **`solution.txt`**: Contains the sequential IDs of neurons in the quasi-clique
- **`results/flywire_solution_original_ids.txt`**: Contains the original FlyWire neuron IDs
- **`results/flywire_solution_summary.json`**: Summary information about the solution
- **`results/flywire_solution_analysis.json`**: Detailed analysis of the solution
- **`results/flywire_solution_graph.png`**: Visualization of the quasi-clique structure
- **`results/flywire_solution_degree_dist.png`**: Degree distribution within the quasi-clique
- **`results/flywire_solution_clustering_dist.png`**: Clustering coefficient distribution

## Implementation Details

### Source Files:

- **`src/optimized_solver_fixed.cpp`**: Implementation of the original density gradient algorithm
- **`src/two_phase_solver.cpp`**: Implementation of the two-phase algorithm with community detection

### Helper Scripts:

- **`scripts/prepare_flywire_data.py`**: Converts the FlyWire CSV to edge list format
- **`scripts/convert_large_ids.py`**: Converts large neuron IDs to sequential integers
- **`scripts/map_results_to_original_ids.py`**: Maps results back to original neuron IDs
- **`scripts/run_flywire_analysis.py`**: Analyzes and visualizes the results

## Algorithm Parameters and Tuning

Both algorithms have several parameters that can be tuned:

### Original Algorithm:

- **Number of Seeds**: Controls the breadth of exploration
- **Alpha Parameter**: Balance between direct connections and clustering (internally adaptive)

### Two-Phase Algorithm:

- **Number of Seeds**: Controls the breadth of exploration
- **Community Detection Passes**: Affects the granularity of detected communities
- **Alpha Parameter**: Balance between direct connections and clustering (internally adaptive)
- **Jaccard Similarity Thresholds**: Controls which solutions are considered for merging

## Benchmark and Comparison

When comparing the two algorithms on the FlyWire connectome, we generally observe:

- **Two-Phase Algorithm**: Tends to find larger quasi-cliques, especially in graphs with strong community structure.
- **Original Algorithm**: Faster for smaller graphs and may be sufficient for simple structures.

For the FlyWire connectome specifically, the two-phase algorithm is recommended as it better accounts for the natural community structure of neuronal networks.

## Troubleshooting

### Common Issues:

1. **Out of Memory Errors**: 
   - Reduce the number of threads
   - Try running with fewer seeds

2. **Long Runtime**:
   - The algorithm can be stopped with Ctrl+C and will save the best solution found so far
   - Try reducing the number of seeds
   - Consider using the original algorithm for a faster, though potentially smaller, solution

3. **Empty Solution**:
   - Check if the input file format is correct
   - Try increasing the number of seeds
   - Verify that the graph has sufficient density to form quasi-cliques

## Biological Interpretation

When analyzing results from the FlyWire connectome, consider:

1. **Functional Relevance**: Quasi-cliques may represent functional modules in the fly brain
2. **Region Overlap**: Check if the neurons in your solution correspond to known anatomical regions
3. **Hub Neurons**: Highly connected neurons within the quasi-clique may play important roles in information processing

## References and Further Reading

1. B. Balasundaram, S. Butenko, and I. V. Hicks. "Clique relaxations in social network analysis: The maximum k-plex problem." Operations Research, 59(1):133–142, 2011.
2. V. D. Blondel, J.-L. Guillaume, R. Lambiotte, and E. Lefebvre. "Fast unfolding of communities in large networks." Journal of Statistical Mechanics: Theory and Experiment, 2008(10):P10008, 2008.
3. FlyWire Project: [https://flywire.ai/](https://flywire.ai/)