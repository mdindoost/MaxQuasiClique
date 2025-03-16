#!/bin/bash
# Run CoreBased MaxQuasiClique algorithm on the FlyWire connectome data

# Colors for output formatting
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default parameters
INPUT_FILE="data/flywire_edges_converted.txt"
OUTPUT_DIR="results"
NUM_SEEDS=5000
INITIAL_SOLUTION=""
USE_CORE_BASED="true"  # Default to using core-based approach

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
echo -e "${BLUE}        CoreBased MaxQuasiClique on FlyWire       ${NC}"
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
        --use-core-based)
        USE_CORE_BASED="true"
        shift
        ;;
        --standard-approach)
        USE_CORE_BASED="false"
        shift
        ;;
    *)
        print_error "Unknown option: $1"
        echo "Usage: $0 [--input INPUT_FILE] [--seeds NUM_SEEDS] [--threads NUM_THREADS] [--initial-solution FILE] [--use-core-based | --standard-approach]"
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
if [ "$USE_CORE_BASED" = "true" ]; then
    echo "- Using core-based approach with analysis results"
else
    echo "- Using standard approach"
fi
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

# Step 2: Check if analysis results exist (for core-based approach)
if [ "$USE_CORE_BASED" = "true" ]; then
    print_section "CHECKING ANALYSIS RESULTS"
    if [ ! -d "analysis_results" ]; then
        print_warning "Analysis results directory not found"
        print_progress "Creating analysis_results directory"
        mkdir -p analysis_results
    fi
    
    if [ ! -f "analysis_results/core_nodes.txt" ]; then
        print_warning "Core nodes file not found. Reverting to standard approach."
        USE_CORE_BASED="false"
    else
        print_progress "Found core nodes file with $(wc -l < "analysis_results/core_nodes.txt") nodes"
    fi
    
    if [ ! -f "analysis_results/node_frequency.csv" ]; then
        print_warning "Node frequency file not found. Reverting to standard approach."
        USE_CORE_BASED="false"
    else
        print_progress "Found node frequency file"
    fi
    
    if [ "$USE_CORE_BASED" = "false" ]; then
        print_warning "Reverting to standard approach due to missing analysis files"
    fi
fi

# Step 3: Build the executable
print_section "BUILDING EXECUTABLE"

print_command "make clean && make"
make clean && make
if [ $? -ne 0 ]; then
    print_error "Failed to build the solver"
    exit 1
fi
print_progress "Successfully built two_phase_solver"
echo

# Step 4: Run the algorithm
print_section "RUNNING ALGORITHM"
print_progress "This may take a while depending on the size of the graph..."
print_progress "You can safely press Ctrl+C to stop - the best solution found so far will be saved"

# Set environment variable for core-based approach
if [ "$USE_CORE_BASED" = "true" ]; then
    export USE_CORE_BASED=1
else
    unset USE_CORE_BASED
fi

if [ -n "$INITIAL_SOLUTION" ]; then
    print_command "./build/two_phase_solver \"$INPUT_FILE\" $NUM_SEEDS $NUM_THREADS \"$INITIAL_SOLUTION\""
    ./build/two_phase_solver "$INPUT_FILE" $NUM_SEEDS $NUM_THREADS "$INITIAL_SOLUTION" &
    PID=$!
else
    print_command "./build/two_phase_solver \"$INPUT_FILE\" $NUM_SEEDS $NUM_THREADS"
./build/two_phase_solver "$INPUT_FILE" $NUM_SEEDS $NUM_THREADS &
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

# Step 5: Map solution back to original IDs (if needed)
if [ -f "solution.txt" ] && [ -s "solution.txt" ] && [ -f "data/id_mapping.csv" ]; then
    print_section "MAPPING SOLUTION TO ORIGINAL IDs"
    ORIGINAL_IDS_FILE="$OUTPUT_DIR/flywire_solution_original_ids.txt"
    SUMMARY_FILE="$OUTPUT_DIR/flywire_solution_summary.json"

    print_command "python3 scripts/map_results_to_original_ids.py --solution \"solution.txt\" --mapping \"data/id_mapping.csv\" --output \"$ORIGINAL_IDS_FILE\" --summary \"$SUMMARY_FILE\""
    python3 scripts/map_results_to_original_ids.py --solution "solution.txt" --mapping "data/id_mapping.csv" --output "$ORIGINAL_IDS_FILE" --summary "$SUMMARY_FILE"
    
    if [ $? -eq 0 ]; then
        print_progress "Successfully mapped solution to original IDs."
        print_progress "Original IDs file: $ORIGINAL_IDS_FILE"
        print_progress "Summary file: $SUMMARY_FILE"
    else
        print_error "Failed to map solution to original IDs."
    fi
fi

# Step 6: Print summary
print_section "SUMMARY"
if [ -f "solution.txt" ]; then
    VERTICES=$(wc -l < solution.txt)
    
    echo -e "${GREEN}Found a quasi-clique with $VERTICES neurons${NC}"
    if [ "$USE_CORE_BASED" = "true" ]; then
        echo -e "${GREEN}Using core-based approach with analysis results${NC}"
    fi
    echo
    echo "Results have been saved to:"
    echo "- Solution: solution.txt"
    if [ -f "$ORIGINAL_IDS_FILE" ]; then
        echo "- Original neuron IDs: $ORIGINAL_IDS_FILE"
    fi
    if [ -f "$SUMMARY_FILE" ]; then
        echo "- Analysis: $SUMMARY_FILE"
    fi
else
    print_warning "No solution found. Check for errors above."
fi
echo -e "${BLUE}====================================================${NC}"
echo "Process completed at $(date)"