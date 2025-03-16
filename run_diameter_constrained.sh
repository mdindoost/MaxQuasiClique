#!/bin/bash
# Run DiameterConstrained MaxQuasiClique algorithm on the FlyWire connectome data

# Colors for output formatting
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default parameters
INPUT_FILE="data/flywire_edges_converted.txt"
OUTPUT_DIR="results/diameter_constrained/"
MAX_DIAMETER=3
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
        cp solution_in_progress.txt "${OUTPUT_DIR}/final_solution.txt"
    fi
    
    print_section "CLEANUP COMPLETE"
    exit 1
}

# Register the cleanup function for signal handling
trap cleanup SIGINT SIGTERM

# Create directories if they don't exist
mkdir -p data
mkdir -p "$OUTPUT_DIR"
mkdir -p build
mkdir -p scripts

# Print banner
echo -e "${BLUE}====================================================${NC}"
echo -e "${BLUE}    Diameter-Constrained MaxQuasiClique Solver     ${NC}"
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
        --threads)
        NUM_THREADS="$2"
        shift
        shift
        ;;
        --max-diameter)
        MAX_DIAMETER="$2"
        shift
        shift
        ;;
        --initial-solution)
        INITIAL_SOLUTION="$2"
        shift
        shift
        ;;
        --output-dir)
        OUTPUT_DIR="$2"
        shift
        shift
        ;;
    *)
        print_error "Unknown option: $1"
        echo "Usage: $0 [--input INPUT_FILE] [--threads NUM_THREADS] [--max-diameter MAX_DIAMETER] [--initial-solution FILE] [--output-dir OUTPUT_DIR]"
        exit 1
        ;;
    esac
done

print_progress "Configuration:"
echo "- Input file: $INPUT_FILE"
echo "- Max diameter: $MAX_DIAMETER"
echo "- Detected CPU cores: $AVAILABLE_THREADS"
echo "- Using threads: $NUM_THREADS"
echo "- Output directory: $OUTPUT_DIR"
if [ -n "$INITIAL_SOLUTION" ]; then
    echo "- Initial solution: $INITIAL_SOLUTION"
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

# Step 2: Build the diameter-constrained implementation
print_section "BUILDING EXECUTABLE"

print_command "make clean && make diameter"
make clean && make diameter
if [ $? -ne 0 ]; then
    print_error "Failed to build the diameter-constrained solver"
    exit 1
fi
print_progress "Successfully built diameter_constrained_solver"
echo

# Step 3: Run the algorithm
print_section "RUNNING DIAMETER-CONSTRAINED ALGORITHM"
print_progress "This may take a while depending on the size of the graph..."
print_progress "You can safely press Ctrl+C to stop - the best solution found so far will be saved"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Build command
COMMAND="./build/diameter_constrained_solver --input \"$INPUT_FILE\" --threads $NUM_THREADS --max-diameter $MAX_DIAMETER --output-dir \"$OUTPUT_DIR\""
if [ -n "$INITIAL_SOLUTION" ]; then
    COMMAND="$COMMAND --initial-solution \"$INITIAL_SOLUTION\""
fi

print_command "$COMMAND"
eval $COMMAND &
PID=$!

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
if [ -f "${OUTPUT_DIR}/quasi_clique_diameter_${MAX_DIAMETER}.txt" ]; then
    SOLUTION_SIZE=$(wc -l < "${OUTPUT_DIR}/quasi_clique_diameter_${MAX_DIAMETER}.txt")
    print_progress "Solution file created with $SOLUTION_SIZE vertices."
else
    print_warning "No solution file found in output directory. Checking for in-progress solution..."
    if [ -f "solution_in_progress.txt" ]; then
        print_progress "Found in-progress solution. Using it as final solution."
        mkdir -p "$OUTPUT_DIR"
        cp solution_in_progress.txt "${OUTPUT_DIR}/quasi_clique_diameter_${MAX_DIAMETER}.txt"
        SOLUTION_SIZE=$(wc -l < "${OUTPUT_DIR}/quasi_clique_diameter_${MAX_DIAMETER}.txt")
        print_progress "Solution file created with $SOLUTION_SIZE vertices."
    else
        print_warning "No solution found."
    fi
fi
echo

# Step 4: Print summary
print_section "SUMMARY"
if [ -f "${OUTPUT_DIR}/quasi_clique_diameter_${MAX_DIAMETER}.txt" ]; then
    VERTICES=$(wc -l < "${OUTPUT_DIR}/quasi_clique_diameter_${MAX_DIAMETER}.txt")
    
    echo -e "${GREEN}Found a quasi-clique with $VERTICES neurons and diameter <= $MAX_DIAMETER${NC}"
    echo
    echo "Results have been saved to:"
    echo "- Final solution: ${OUTPUT_DIR}/quasi_clique_diameter_${MAX_DIAMETER}.txt"
    echo "- Also copied to: solution.txt (for compatibility with existing tools)"
    
    # Copy for compatibility with existing tools
    cp "${OUTPUT_DIR}/quasi_clique_diameter_${MAX_DIAMETER}.txt" solution.txt
else
    if [ -f "solution.txt" ] && [ -s "solution.txt" ]; then
        SOLUTION_SIZE=$(wc -l < "solution.txt")
        echo -e "${YELLOW}Found a potential quasi-clique with $SOLUTION_SIZE neurons${NC}"
        echo "(But no specific solution file was generated)"
    else
        print_warning "No solution or summary found. Check for errors above."
    fi
fi
echo -e "${BLUE}====================================================${NC}"
echo "Process completed at $(date)"