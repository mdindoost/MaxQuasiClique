#!/bin/bash
# Run MaxQuasiClique algorithms on the FlyWire connectome data
# This script handles ID conversion and uses the modular two-phase algorithm

# Colors for output formatting
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default parameters
INPUT_FILE="data/edges.csv"
OUTPUT_DIR="results"
NUM_SEEDS=5000  # Increase default seeds for better exploration?!
INITIAL_SOLUTION=""

# Auto-detect number of threads (use 75% of available cores to avoid overloading the system)
AVAILABLE_THREADS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
RECOMMENDED_THREADS=$(( AVAILABLE_THREADS * 3 / 4 ))
if [ $RECOMMENDED_THREADS -lt 1 ]; then
    RECOMMENDED_THREADS=1
fi
NUM_THREADS=$RECOMMENDED_THREADS

# Function to print section headers
print_section() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
}

# Function to print progress messages
print_progress() {
    echo -e "${GREEN}[PROGRESS]${NC} $1"
}

# Function to print warnings
print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to print errors
print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to print commands being executed
print_command() {
    echo -e "${BLUE}[EXECUTING]${NC} $1"
}

# Function to handle graceful termination
cleanup() {
    print_warning "Script interrupted. Cleaning up..."
    
    # Check if the algorithm is still running
    if [ -n "$PID" ] && ps -p $PID > /dev/null; then
        print_progress "Sending termination signal to algorithm..."
        kill -TERM $PID
        
        # Give it some time to finish gracefully
        for i in {1..5}; do
            if ! ps -p $PID > /dev/null; then
                break
            fi
            sleep 1
        done
        
        # Force kill if still running
        if ps -p $PID > /dev/null; then
            print_warning "Algorithm not responding. Force killing..."
            kill -9 $PID
        fi
    fi
    
    print_progress "Checking for in-progress solution..."
    if [ -f "solution_in_progress.txt" ]; then
        print_progress "Found in-progress solution. Saving it as final solution."
        cp solution_in_progress.txt solution.txt
    fi
    
    print_section "CLEANUP COMPLETE"
    exit 1
}

# Register the cleanup function for signal handling
trap cleanup SIGINT SIGTERM

# Create directories if they don't exist
mkdir -p data
mkdir -p $OUTPUT_DIR
mkdir -p build
mkdir -p scripts

# Print banner
echo -e "${BLUE}====================================================${NC}"
echo -e "${BLUE}        MaxQuasiClique on FlyWire Connectome        ${NC}"
echo -e "${BLUE}====================================================${NC}"
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
        --initial-solution)
        INITIAL_SOLUTION="$2"
        shift
        shift
        ;;
    *)
        print_error "Unknown option: $1"
        echo "Usage: $0 [--input INPUT_FILE] [--seeds NUM_SEEDS] [--threads NUM_THREADS]"
        exit 1
        ;;
    esac
done

print_progress "Configuration:"
echo "- Input file: $INPUT_FILE"
echo "- Number of seeds: $NUM_SEEDS"
echo "- Detected CPU cores: $AVAILABLE_THREADS"
echo "- Using threads: $NUM_THREADS"
if [ -n "$INITIAL_SOLUTION" ]; then
    echo "- Initial solution: $INITIAL_SOLUTION"
fi
echo "- No timeout (run until completion)"
echo

# Step 1: Check if input file exists
print_section "CHECKING INPUT FILE"
if [ ! -f "$INPUT_FILE" ]; then
    print_error "Input file $INPUT_FILE not found"
    exit 1
fi
print_progress "Input file exists: $INPUT_FILE"
print_progress "File size: $(du -h "$INPUT_FILE" | cut -f1)"
echo "First few lines of the input file:"
head -n 5 "$INPUT_FILE"
echo

# Step 2: Build the modular implementation
print_section "BUILDING EXECUTABLE"

print_command "make clean && make"
make clean && make
if [ $? -ne 0 ]; then
    print_error "Failed to build the modular two-phase solver"
    exit 1
fi
print_progress "Successfully built two_phase_solver"
echo

# Step 3: Prepare the data - Convert to edge list format
print_section "CONVERTING CSV TO EDGE LIST"
EDGE_LIST_FILE="data/flywire_edges.txt"

# Skip if already exists
if [ ! -f "$EDGE_LIST_FILE" ]; then
    print_command "python3 scripts/prepare_flywire_data.py --input \"$INPUT_FILE\" --output \"$EDGE_LIST_FILE\""
    python3 scripts/prepare_flywire_data.py --input "$INPUT_FILE" --output "$EDGE_LIST_FILE"
    if [ $? -ne 0 ]; then
        print_error "Failed to prepare data"
        exit 1
    fi
    print_progress "Successfully converted CSV to edge list"
    print_progress "Output file: $EDGE_LIST_FILE"
    print_progress "File size: $(du -h "$EDGE_LIST_FILE" | cut -f1)"
else
    print_progress "Edge list file already exists, skipping conversion."
    print_progress "File: $EDGE_LIST_FILE"
    print_progress "File size: $(du -h "$EDGE_LIST_FILE" | cut -f1)"
fi
echo "First few lines of the edge list file:"
head -n 5 "$EDGE_LIST_FILE"
echo

# Step 4: Convert large IDs to sequential integers
print_section "CONVERTING LARGE IDs TO SEQUENTIAL INTEGERS"
CONVERTED_EDGE_FILE="data/flywire_edges_converted.txt"
ID_MAPPING_FILE="data/id_mapping.csv"

# Skip if already exists
if [ ! -f "$CONVERTED_EDGE_FILE" ]; then
    print_command "python3 scripts/convert_large_ids.py --input \"$EDGE_LIST_FILE\" --output \"$CONVERTED_EDGE_FILE\" --mapping \"$ID_MAPPING_FILE\""
    python3 scripts/convert_large_ids.py --input "$EDGE_LIST_FILE" --output "$CONVERTED_EDGE_FILE" --mapping "$ID_MAPPING_FILE"
    if [ $? -ne 0 ]; then
        print_error "Failed to convert IDs"
        exit 1
    fi
    print_progress "Successfully converted large IDs to sequential integers"
    print_progress "Output file: $CONVERTED_EDGE_FILE"
    print_progress "Mapping file: $ID_MAPPING_FILE"
    print_progress "File size: $(du -h "$CONVERTED_EDGE_FILE" | cut -f1)"
else
    print_progress "Converted edge file already exists, skipping conversion."
    print_progress "File: $CONVERTED_EDGE_FILE"
    print_progress "File size: $(du -h "$CONVERTED_EDGE_FILE" | cut -f1)"
fi
echo "First few lines of the converted edge list file:"
head -n 5 "$CONVERTED_EDGE_FILE"
echo

# Step 5: Run the algorithm on converted data
print_section "RUNNING MODULAR TWO-PHASE ALGORITHM"
print_progress "This may take a while depending on the size of the graph..."
print_progress "You can safely press Ctrl+C to stop - the best solution found so far will be saved"
if [ -n "$INITIAL_SOLUTION" ]; then
    print_command "./build/two_phase_solver \"$CONVERTED_EDGE_FILE\" $NUM_SEEDS $NUM_THREADS \"$INITIAL_SOLUTION\""
    ./build/two_phase_solver "$CONVERTED_EDGE_FILE" $NUM_SEEDS $NUM_THREADS "$INITIAL_SOLUTION" &
    PID=$!
else
    print_command "./build/two_phase_solver \"$CONVERTED_EDGE_FILE\" $NUM_SEEDS $NUM_THREADS"
    ./build/two_phase_solver "$CONVERTED_EDGE_FILE" $NUM_SEEDS $NUM_THREADS &
    PID=$!
fi

echo "Starting algorithm at $(date)"

# Wait for the algorithm to finish
wait $PID
ALGORITHM_STATUS=$?

echo "Algorithm finished at $(date) with status $ALGORITHM_STATUS"
if [ $ALGORITHM_STATUS -ne 0 ]; then
    print_warning "Algorithm exited with non-zero status: $ALGORITHM_STATUS"
else
    print_progress "Algorithm completed successfully."
fi

# Check if solution file exists and has content
if [ -f "solution.txt" ]; then
    SOLUTION_SIZE=$(wc -l < solution.txt)
    print_progress "Solution file created with $SOLUTION_SIZE vertices."
else
    print_warning "No solution file found. Checking for in-progress solution..."
    if [ -f "solution_in_progress.txt" ]; then
        print_progress "Found in-progress solution. Using it as final solution."
        cp solution_in_progress.txt solution.txt
        SOLUTION_SIZE=$(wc -l < solution.txt)
        print_progress "Solution file created with $SOLUTION_SIZE vertices."
    else
        print_warning "No solution found."
    fi
fi
echo

# Step 6: Map solution back to original IDs
print_section "MAPPING SOLUTION TO ORIGINAL IDs"
if [ -f "solution.txt" ] && [ -s "solution.txt" ]; then
    ORIGINAL_IDS_FILE="$OUTPUT_DIR/flywire_solution_original_ids.txt"
    SUMMARY_FILE="$OUTPUT_DIR/flywire_solution_summary.json"

    print_command "python3 scripts/map_results_to_original_ids.py --solution \"solution.txt\" --mapping \"$ID_MAPPING_FILE\" --output \"$ORIGINAL_IDS_FILE\" --summary \"$SUMMARY_FILE\""
    python3 scripts/map_results_to_original_ids.py --solution "solution.txt" --mapping "$ID_MAPPING_FILE" --output "$ORIGINAL_IDS_FILE" --summary "$SUMMARY_FILE"
    
    if [ $? -eq 0 ]; then
        print_progress "Successfully mapped solution to original IDs."
        print_progress "Original IDs file: $ORIGINAL_IDS_FILE"
        print_progress "Summary file: $SUMMARY_FILE"
    else
        print_error "Failed to map solution to original IDs."
    fi
else
    print_warning "Skipping mapping step - no valid solution file found."
fi
echo

# Step 7: Analyze the results using sequential IDs
print_section "ANALYZING RESULTS"
if [ -f "solution.txt" ] && [ -s "solution.txt" ]; then
    print_command "python3 scripts/run_flywire_analysis.py --executable \"build/two_phase_solver\" --input \"$CONVERTED_EDGE_FILE\" --seeds $NUM_SEEDS --threads $NUM_THREADS --solution \"solution.txt\" --skip-run --output-prefix \"$OUTPUT_DIR/flywire_solution\""
    python3 scripts/run_flywire_analysis.py --executable "build/two_phase_solver" \
        --input "$CONVERTED_EDGE_FILE" \
        --seeds $NUM_SEEDS \
        --threads $NUM_THREADS \
        --solution "solution.txt" \
        --skip-run \
        --output-prefix "$OUTPUT_DIR/flywire_solution"
    
    if [ $? -eq 0 ]; then
        print_progress "Successfully analyzed results."
        print_progress "Analysis files are in: $OUTPUT_DIR/"
    else
        print_error "Failed to analyze results."
    fi
else
    print_warning "Skipping analysis step - no valid solution file found."
fi
echo

# Step 8: Print summary
print_section "SUMMARY"
if [ -f "$SUMMARY_FILE" ]; then
    # Extract key metrics from the JSON file
    VERTICES=$(grep "solution_size" "$SUMMARY_FILE" | cut -d ':' -f 2 | tr -d ',' | tr -d ' ')
    
    echo -e "${GREEN}Found a quasi-clique with $VERTICES neurons${NC}"
    echo
    echo "Results have been saved to:"
    echo "- Sequential ID solution: solution.txt"
    echo "- Original neuron IDs: $ORIGINAL_IDS_FILE"
    echo "- Analysis: $OUTPUT_DIR/flywire_solution_analysis.json"
    echo "- Summary: $SUMMARY_FILE"
    echo "- Visualizations: $OUTPUT_DIR/flywire_solution_*.png"
else
    if [ -f "solution.txt" ] && [ -s "solution.txt" ]; then
        SOLUTION_SIZE=$(wc -l < solution.txt)
        echo -e "${YELLOW}Found a potential quasi-clique with $SOLUTION_SIZE neurons${NC}"
        echo "(But no summary file was generated)"
    else
        print_warning "No solution or summary found. Check for errors above."
    fi
fi
echo -e "${BLUE}====================================================${NC}"
echo "Process completed at $(date)"