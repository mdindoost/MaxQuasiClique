#!/bin/bash
# Run the MaxQuasiClique algorithm on the FlyWire connectome data
# This script handles ID conversion to deal with large neuron IDs

# Default parameters
INPUT_FILE="data/edges.csv"
OUTPUT_DIR="results"
NUM_SEEDS=50
NUM_THREADS=8
TIMEOUT=3600  # 1 hour timeout

# Create directories if they don't exist
mkdir -p data
mkdir -p $OUTPUT_DIR
mkdir -p build
mkdir -p scripts

# Print banner
echo "===================================================="
echo "        MaxQuasiClique on FlyWire Connectome        "
echo "===================================================="
echo

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --input)
        INPUT_FILE="$2"
        shift
        shift
        ;;
        --seeds)
        NUM_SEEDS="$2"
        shift
        shift
        ;;
        --threads)
        NUM_THREADS="$2"
        shift
        shift
        ;;
        --timeout)
        TIMEOUT="$2"
        shift
        shift
        ;;
        *)
        echo "Unknown option: $1"
        echo "Usage: $0 [--input INPUT_FILE] [--seeds NUM_SEEDS] [--threads NUM_THREADS] [--timeout SECONDS]"
        exit 1
        ;;
    esac
done

echo "Input file: $INPUT_FILE"
echo "Number of seeds: $NUM_SEEDS"
echo "Number of threads: $NUM_THREADS"
echo "Timeout: $TIMEOUT seconds"
echo

# Step 1: Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file $INPUT_FILE not found"
    exit 1
fi

# Step 2: Build the executables if they don't exist
echo "Building executables..."
if [ ! -f "build/optimized_solver" ]; then
    make optimized_solver
    if [ $? -ne 0 ]; then
        echo "Error: Failed to build optimized_solver"
        exit 1
    fi
fi
echo "Build complete."
echo

# Step 3: Prepare the data - Convert to edge list format
echo "Converting CSV to edge list format..."
EDGE_LIST_FILE="data/flywire_edges.txt"

# Skip if already exists
if [ ! -f "$EDGE_LIST_FILE" ]; then
    python3 scripts/prepare_flywire_data.py --input "$INPUT_FILE" --output "$EDGE_LIST_FILE"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to prepare data"
        exit 1
    fi
else
    echo "Edge list file already exists, skipping conversion."
fi
echo

# Step 4: Convert large IDs to sequential integers
echo "Converting large neuron IDs to sequential integers..."
CONVERTED_EDGE_FILE="data/flywire_edges_converted.txt"
ID_MAPPING_FILE="data/id_mapping.csv"

# Skip if already exists
if [ ! -f "$CONVERTED_EDGE_FILE" ]; then
    python3 scripts/convert_large_ids.py --input "$EDGE_LIST_FILE" --output "$CONVERTED_EDGE_FILE" --mapping "$ID_MAPPING_FILE"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to convert IDs"
        exit 1
    fi
else
    echo "Converted edge file already exists, skipping conversion."
fi
echo

# Step 5: Run the algorithm on converted data
echo "Running MaxQuasiClique algorithm on FlyWire data..."
echo "This may take a while depending on the size of the graph..."
./build/optimized_solver "$CONVERTED_EDGE_FILE" $NUM_SEEDS $NUM_THREADS &
PID=$!

# Monitor progress and enforce timeout
(
    SECONDS=0
    while kill -0 $PID 2>/dev/null; do
        if [ $SECONDS -ge $TIMEOUT ]; then
            echo "Timeout reached ($TIMEOUT seconds). Terminating..."
            kill -9 $PID
            break
        fi
        
        if [ $((SECONDS % 60)) -eq 0 ]; then
            echo "Still running... (elapsed: $SECONDS seconds)"
        fi
        
        sleep 5
    done
) &
MONITOR_PID=$!

# Wait for the algorithm to finish
wait $PID
ALGORITHM_STATUS=$?
kill $MONITOR_PID 2>/dev/null

if [ $ALGORITHM_STATUS -ne 0 ]; then
    echo "Algorithm failed or was terminated."
    echo "Check if a partial solution was generated."
fi
echo

# Step 6: Map solution back to original IDs
echo "Mapping solution back to original FlyWire neuron IDs..."
ORIGINAL_IDS_FILE="$OUTPUT_DIR/flywire_solution_original_ids.txt"
SUMMARY_FILE="$OUTPUT_DIR/flywire_solution_summary.json"

python3 scripts/map_results_to_original_ids.py --solution "solution.txt" --mapping "$ID_MAPPING_FILE" --output "$ORIGINAL_IDS_FILE" --summary "$SUMMARY_FILE"
echo

# Step 7: Analyze the results using sequential IDs
echo "Analyzing results with sequential IDs..."
python3 scripts/run_flywire_analysis.py --executable "build/optimized_solver" \
    --input "$CONVERTED_EDGE_FILE" \
    --seeds $NUM_SEEDS \
    --threads $NUM_THREADS \
    --solution "solution.txt" \
    --skip-run \
    --output-prefix "$OUTPUT_DIR/flywire_solution"
echo

# Step 8: Print summary
echo "===================================================="
echo "                   Summary                          "
echo "===================================================="
if [ -f "$SUMMARY_FILE" ]; then
    # Extract key metrics from the JSON file
    VERTICES=$(grep "solution_size" "$SUMMARY_FILE" | cut -d ':' -f 2 | tr -d ',' | tr -d ' ')
    
    echo "Found a quasi-clique with $VERTICES neurons"
    echo
    echo "Results have been saved to:"
    echo "- Sequential ID solution: solution.txt"
    echo "- Original neuron IDs: $ORIGINAL_IDS_FILE"
    echo "- Analysis: $OUTPUT_DIR/flywire_solution_analysis.json"
    echo "- Summary: $SUMMARY_FILE"
    echo "- Visualizations: $OUTPUT_DIR/flywire_solution_*.png"
else
    echo "No summary results found. Check for errors above."
fi
echo "===================================================="