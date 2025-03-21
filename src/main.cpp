#include "Graph.h"
#include "TwoPhaseQuasiCliqueSolver.h"
#include <iostream>
#include <chrono>
#include <csignal>
#include <string>
#include <cstdlib>  // For getenv

// Global termination flag
volatile sig_atomic_t terminationRequested = 0;

// Signal handler for graceful termination
void signalHandler(int signal) {
    std::cout << "Received termination signal " << signal << ". Finishing up..." << std::endl;
    terminationRequested = 1;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <graph_file> [num_seeds] [num_threads] [initial_solution_file]" << std::endl;
        return 1;
    }
    
    std::string filename = argv[1];
    int numSeeds = (argc > 2) ? std::stoi(argv[2]) : 20;
    int numThreads = (argc > 3) ? std::stoi(argv[3]) : 0; // 0 means use all available
    std::string initialSolutionFile = (argc > 4) ? argv[4] : "";
    double connectivityThreshold = 0.1; // Default threshold for community merging
    
    // Check for environment variable to enable core-based approach
    bool useCoreBasedApproach = false;
    if (getenv("USE_CORE_BASED") != nullptr) {
        useCoreBasedApproach = true;
    }
    
    // Register signal handlers
    signal(SIGINT, signalHandler);  // Ctrl+C
    signal(SIGTERM, signalHandler); // kill command
    
    // Print system information
    std::cout << "System information:" << std::endl;
    std::cout << "  Hardware concurrency: " << std::thread::hardware_concurrency() << " threads" << std::endl;
    std::cout << "  Using " << numThreads << " threads" << std::endl;
    std::cout << "  Community connectivity threshold: " << connectivityThreshold << std::endl;
    if (useCoreBasedApproach) {
        std::cout << "  Using core-based expansion approach" << std::endl;
    }
    
    // Load graph
    Graph graph;
    auto loadStartTime = std::chrono::high_resolution_clock::now();
    
    if (!graph.loadFromFile(filename)) {
        return 1;
    }
    
    auto loadEndTime = std::chrono::high_resolution_clock::now();
    auto loadDuration = std::chrono::duration_cast<std::chrono::milliseconds>(loadEndTime - loadStartTime).count();
    std::cout << "Graph loaded in " << loadDuration / 1000.0 << " seconds" << std::endl;
    
    // Create solver
    TwoPhaseQuasiCliqueSolver solver(graph);
    
    // Set community merger threshold
    solver.setCommunityConnectivityThreshold(connectivityThreshold);
    
    // Measure execution time
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Find large quasi-clique using the selected approach
    std::vector<int> solution;
    
    if (useCoreBasedApproach) {
        // Use the core-based approach
        solution = solver.findFromCoreNodes(numThreads);
    } else if (!initialSolutionFile.empty()) {
        // Check if we have an initial solution
        std::vector<int> initialSolution = solver.loadSolutionFromFile(initialSolutionFile);
        if (!initialSolution.empty()) {
            std::cout << "Starting from initial solution with " << initialSolution.size() << " nodes" << std::endl;
            solution = solver.expandFromExistingSolution(initialSolution, numSeeds, numThreads);
        } else {
            std::cout << "Failed to load initial solution, falling back to standard algorithm" << std::endl;
            solution = solver.findLargeQuasiClique(numSeeds, numThreads);
        }
    } else {
        // Find large quasi-clique using standard algorithm
        solution = solver.findLargeQuasiClique(numSeeds, numThreads);
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    
    // Verify and print solution
    solver.verifyAndPrintSolution(solution);
    
    std::cout << "\nAlgorithm execution time: " << duration / 1000.0 << " seconds" << std::endl;
    std::cout << "Total time (including loading): " << (duration + loadDuration) / 1000.0 << " seconds" << std::endl;
    
    // Save solution to file
    solver.saveSolution(solution);
    
    return 0;
}