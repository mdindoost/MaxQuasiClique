# Makefile for MaxQuasiClique project

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -pthread
DEBUG_FLAGS = -g -O0 -DDEBUG

SRC_DIR = .
BUILD_DIR = build
TARGETS = max_quasi_clique optimized_solver

# Default target
all: directories $(TARGETS)

# Create build directory
directories:
	mkdir -p $(BUILD_DIR)

# Basic implementation
max_quasi_clique: $(SRC_DIR)/main.cpp
	$(CXX) $(CXXFLAGS) $< -o $(BUILD_DIR)/$@

# Optimized implementation
optimized_solver: $(SRC_DIR)/optimized_solver.cpp
	$(CXX) $(CXXFLAGS) $< -o $(BUILD_DIR)/$@

# Debug versions
debug: CXXFLAGS += $(DEBUG_FLAGS)
debug: all

# Run the basic implementation on sample graph
run: all
	./$(BUILD_DIR)/max_quasi_clique sample_graph.txt 10

# Run the optimized implementation on sample graph
run-opt: all
	./$(BUILD_DIR)/optimized_solver sample_graph.txt 10 4

# Generate test graphs
graphs:
	python3 generate_test_graph.py --neurons 1000 --output test_graph_1000.txt
	python3 generate_test_graph.py --neurons 2000 --output test_graph_2000.txt
	python3 generate_test_graph.py --neurons 5000 --output test_graph_5000.txt

# Run benchmarks
benchmark: all graphs
	python3 benchmark.py --executable=$(BUILD_DIR)/max_quasi_clique --optimized=$(BUILD_DIR)/optimized_solver --sizes=1000,2000,5000 --seeds=5 --threads=4

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)

.PHONY: all directories debug run run-opt graphs benchmark clean