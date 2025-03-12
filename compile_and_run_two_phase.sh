#!/bin/bash
# Compile and run the two-phase quasi-clique algorithm

# Colors for output formatting
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default parameters
INPUT_FILE="data/flywire_edges_converted.txt"
NUM_SEEDS=100
NUM_THREADS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

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
echo "- Using threads: $NUM_THREADS"
echo

# Create directories if they don't exist
mkdir -p build

# Step 1: Compile the two-phase algorithm
print_section "COMPILING TWO-PHASE ALGORITHM"
print_command "g++ -std=c++17 -Wall -Wextra -O3 -pthread src/two_phase_solver.cpp -o build/two_phase_solver"
g++ -std=c++17 -Wall -Wextra -O3 -pthread src/two_phase_solver.cpp -o build/two_phase_solver

if [ $? -ne 0 ]; then
    print_error "Failed to compile two_phase_solver"
    exit 1
fi

print_progress "Successfully compiled two_phase_solver"
echo

# Step 2: Check if input file exists
print_section "CHECKING INPUT FILE"
if [ ! -f "$INPUT_FILE" ]; then
    print_error "Input file $INPUT_FILE not found"
    exit 1
fi
print_progress "Input file exists: $INPUT_FILE"
print_progress "File size: $(du -h "$INPUT_FILE" | cut -f1)"
echo

# Step 3: Run the algorithm
print_section "RUNNING TWO-PHASE ALGORITHM"
print_progress "This may take a while depending on the size of the graph..."
print_progress "You can safely press Ctrl+C to stop - the best solution found so far will be saved"
print_command "./build/two_phase_solver \"$INPUT_FILE\" $NUM_SEEDS $NUM_THREADS"

echo "Starting algorithm at $(date)"
./build/two_phase_solver "$INPUT_FILE" $NUM_SEEDS $NUM_THREADS

ALGORITHM_STATUS=$?
echo "Algorithm finished at $(date) with status $ALGORITHM_STATUS"

if [ $ALGORITHM_STATUS -ne 0 ]; then
    print_warning "Algorithm exited with non-zero status: $ALGORITHM_STATUS"
else
    print_progress "Algorithm completed successfully."
fi

# Step 4: Check results
print_section "CHECKING RESULTS"

if [ -f "solution.txt" ]; then
    SOLUTION_SIZE=$(wc -l < solution.txt)
    print_progress "Solution file created with $SOLUTION_SIZE vertices."
    
    # Map back to original IDs if mapping file exists
    if [ -f "data/id_mapping.csv" ]; then
        print_progress "Mapping solution back to original FlyWire neuron IDs..."
        python3 scripts/map_results_to_original_ids.py --solution "solution.txt" \
            --mapping "data/id_mapping.csv" \
            --output "results/flywire_solution_original_ids.txt" \
            --summary "results/flywire_solution_summary.json"
    else
        print_warning "ID mapping file not found, skipping mapping to original IDs."
    fi
else
    print_warning "No solution file found."
fi

print_section "SUMMARY"
echo -e "${BLUE}====================================================${NC}"
echo "Process completed at $(date)"