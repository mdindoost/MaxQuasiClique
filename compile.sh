#!/bin/bash

# Colors for output formatting
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Compiling modular two-phase solver...${NC}"

# Create the combined file from parts
echo -e "${YELLOW}Creating combined TwoPhaseQuasiCliqueSolver...${NC}"

# Create src directory if it doesn't exist
mkdir -p src

# Check if part files exist
if [[ -f src/TwoPhaseQuasiCliqueSolver.cpp && -f src/TwoPhaseQuasiCliqueSolver_part2.cpp && -f src/TwoPhaseQuasiCliqueSolver_part3.cpp ]]; then
    cat src/TwoPhaseQuasiCliqueSolver.cpp src/TwoPhaseQuasiCliqueSolver_part2.cpp src/TwoPhaseQuasiCliqueSolver_part3.cpp > src/TwoPhaseQuasiCliqueSolver_combined.cpp
    echo -e "${GREEN}Successfully created combined file.${NC}"
else
    echo -e "${RED}Error: One or more part files not found!${NC}"
    exit 1
fi

# Create build directory
mkdir -p build

# Compile the source files
echo -e "${YELLOW}Compiling source files...${NC}"
g++ -std=c++17 -Wall -Wextra -O3 -pthread -c src/Graph.cpp -o build/Graph.o
g++ -std=c++17 -Wall -Wextra -O3 -pthread -c src/ThreadPool.cpp -o build/ThreadPool.o
g++ -std=c++17 -Wall -Wextra -O3 -pthread -c src/CommunityDetector.cpp -o build/CommunityDetector.o
g++ -std=c++17 -Wall -Wextra -O3 -pthread -c src/TwoPhaseQuasiCliqueSolver_combined.cpp -o build/TwoPhaseQuasiCliqueSolver.o
g++ -std=c++17 -Wall -Wextra -O3 -pthread -c src/main.cpp -o build/main.o

# Link all object files
echo -e "${YELLOW}Linking...${NC}"
g++ -std=c++17 -Wall -Wextra -O3 -pthread build/Graph.o build/ThreadPool.o build/CommunityDetector.o build/TwoPhaseQuasiCliqueSolver.o build/main.o -o build/two_phase_solver

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Compilation successful! Executable created at build/two_phase_solver${NC}"
    echo "To run: ./build/two_phase_solver data/flywire_edges_converted.txt 500 8"
else
    echo -e "${RED}Compilation failed!${NC}"
    exit 1
fi