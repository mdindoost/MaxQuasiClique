#!/bin/bash
# Run the MaxQuasiClique algorithm on the FlyWire connectome data
# This script automates the entire process from data preparation to analysis

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

# Step 3: Prepare the data
echo "Preparing FlyWire data..."
EDGE_LIST_FILE="data/flywire_edges.txt"
python3 prepare_flywire_data.py --input "$INPUT_FILE" --output "$EDGE_LIST_FILE"
if [ $? -ne 0 ]; then
    echo "Error: Failed to prepare data"
    exit 1
fi
echo

# Step 4: Run the algorithm
echo "Running MaxQuasiClique algorithm on FlyWire data..."
echo "This may take a while depending on the size of the graph..."
./build/optimized_solver "$EDGE_LIST_FILE" $NUM_SEEDS $NUM_THREADS &
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

# Step 5: Analyze the results
echo "Analyzing results..."
python3 run_flywire_analysis.py --executable "build/optimized_solver" \
    --input "$EDGE_LIST_FILE" \
    --seeds $NUM_SEEDS \
    --threads $NUM_THREADS \
    --solution "solution.txt" \
    --skip-run \
    --output-prefix "$OUTPUT_DIR/flywire_solution"
echo

# Step 6: Print summary
echo "===================================================="
echo "                   Summary                          "
echo "===================================================="
if [ -f "$OUTPUT_DIR/flywire_solution_analysis.json" ]; then
    # Extract key metrics from the JSON file
    VERTICES=$(grep "num_vertices" "$OUTPUT_DIR/flywire_solution_analysis.json" | cut -d ':' -f 2 | tr -d ',' | tr -d ' ')
    DENSITY=$(grep "density" "$OUTPUT_DIR/flywire_solution_analysis.json" | head -1 | cut -d ':' -f 2 | tr -d ',' | tr -d ' ')
    IS_QUASI=$(grep "is_quasi_clique" "$OUTPUT_DIR/flywire_solution_analysis.json" | cut -d ':' -f 2 | tr -d ',' | tr -d ' ')
    
    echo "Found a quasi-clique with $VERTICES vertices"
    echo "Density: $DENSITY"
    echo "Is valid quasi-clique: $IS_QUASI"
    echo
    echo "Results have been saved to:"
    echo "- Solution: solution.txt"
    echo "- Analysis: $OUTPUT_DIR/flywire_solution_analysis.json"
    echo "- Visualizations: $OUTPUT_DIR/flywire_solution_*.png"
else
    echo "No analysis results found. Check for errors above."
fi
echo "===================================================="