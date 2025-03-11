# Finding Quasi-Cliques in the FlyWire Connectome

This document provides instructions for running the MaxQuasiClique algorithm on the FlyWire connectome data with special handling for the large neuron IDs.

## The Challenge

The FlyWire connectome uses very large integer IDs for neurons (e.g., 720575940621287977). These IDs are outside the range of standard 32-bit integers, making direct processing with our C++ implementation challenging.

## Solution

We've added a two-phase approach:

1. **ID Conversion**: Convert the large neuron IDs to sequential integers for processing
2. **Result Mapping**: Map the solution back to the original FlyWire neuron IDs

## Files

- `scripts/convert_large_ids.py`: Converts large neuron IDs to sequential integers
- `scripts/map_results_to_original_ids.py`: Maps results back to original neuron IDs
- `run_flywire_with_conversion.sh`: All-in-one script implementing the full workflow

## Usage

### Automated Approach (Recommended)

```bash
# Make the script executable
chmod +x run_flywire_with_conversion.sh

# Run with default settings
./run_flywire_with_conversion.sh

# Or with custom parameters
./run_flywire_with_conversion.sh --input data/edges.csv --seeds 50 --threads 8 --timeout 3600
```

### Manual Approach

If you prefer to run each step separately:

1. **Convert to edge list format** (if needed)
   ```bash
   python3 scripts/prepare_flywire_data.py --input data/edges.csv --output data/flywire_edges.txt
   ```

2. **Convert large IDs to sequential integers**
   ```bash
   python3 scripts/convert_large_ids.py --input data/flywire_edges.txt \
       --output data/flywire_edges_converted.txt \
       --mapping data/id_mapping.csv
   ```

3. **Run the algorithm**
   ```bash
   ./build/optimized_solver data/flywire_edges_converted.txt 50 8
   ```

4. **Map solution back to original IDs**
   ```bash
   python3 scripts/map_results_to_original_ids.py --solution solution.txt \
       --mapping data/id_mapping.csv \
       --output results/flywire_solution_original_ids.txt \
       --summary results/flywire_solution_summary.json
   ```

5. **Analyze the results**
   ```bash
   python3 scripts/run_flywire_analysis.py --input data/flywire_edges_converted.txt \
       --solution solution.txt \
       --skip-run \
       --output-prefix results/flywire_solution
   ```

## Understanding the Results

After running the algorithm, you'll find:

1. **`solution.txt`**: Contains the sequential IDs of neurons in the quasi-clique

2. **`results/flywire_solution_original_ids.txt`**: Contains the original FlyWire neuron IDs corresponding to the quasi-clique

3. **`results/flywire_solution_summary.json`**: A summary of the solution, including:
   - `solution_size`: Total number of neurons in the quasi-clique
   - `mapped_ids`: Number of IDs successfully mapped back to original IDs
   - `original_neuron_ids`: List of the original FlyWire neuron IDs

4. **Analysis and visualization files**: Various files in the `results/` directory that provide statistics and visualizations of the quasi-clique structure

## Performance Notes

- The conversion process adds a small overhead but is necessary for handling the large neuron IDs
- The ID mapping is stored for reuse, so subsequent runs will be faster
- For large connectomes, the algorithm may take several hours to run

## Troubleshooting

If you encounter issues:

- **Memory errors**: Try reducing the number of seeds or threads
- **Conversion errors**: Check if the CSV format matches the expected format (source ID, target ID)
- **Algorithm timeout**: Increase the timeout value or reduce the problem size

## Next Steps

After finding a quasi-clique in the FlyWire connectome, you might:

1. **Analyze neuron relationships**: Study the connectivity patterns within the quasi-clique
2. **Biological interpretation**: Relate the quasi-clique to known neuronal circuits or functions
3. **Compare with other metrics**: Analyze how the quasi-clique relates to other network properties

## References

- FlyWire Project: [https://flywire.ai/](https://flywire.ai/)
- Quasi-Clique Definition: A subgraph with k vertices is a quasi-clique if it contains more than (k choose 2) / 2 edges