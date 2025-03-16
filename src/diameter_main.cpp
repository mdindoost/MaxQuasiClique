#include "Graph.h"
#include "DiameterConstrainedQuasiCliqueSolver.h"
#include <iostream>
#include <chrono>
#include <csignal>
#include <string>
#include <filesystem>
#include <fstream>

// Global termination flag
volatile sig_atomic_t terminationRequested = 0;

// Signal handler for graceful termination
void signalHandler(int signal) {
    std::cout << "Received termination signal " << signal << ". Finishing up..." << std::endl;
    terminationRequested = 1;
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    std::string filename = "data/flywire_edges_converted.txt";
    int numThreads = 0; // 0 means use all available
    std::string initialSolutionFile = "";
    int maxDiameter = 3;
    std::string outputDir = "results/";
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--input" && i + 1 < argc) {
            filename = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            numThreads = std::stoi(argv[++i]);
        } else if (arg == "--initial-solution" && i + 1 < argc) {
            initialSolutionFile = argv[++i];
        } else if (arg == "--max-diameter" && i + 1 < argc) {
            maxDiameter = std::stoi(argv[++i]);
        } else if (arg == "--output-dir" && i + 1 < argc) {
            outputDir = argv[++i];
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --input FILE             Input graph file (default: data/flywire_edges_converted.txt)" << std::endl;
            std::cout << "  --threads N              Number of threads to use (default: all available)" << std::endl;
            std::cout << "  --initial-solution FILE  Initial solution file (optional)" << std::endl;
            std::cout << "  --max-diameter N         Maximum allowed diameter (default: 3)" << std::endl;
            std::cout << "  --output-dir DIR         Output directory (default: results/)" << std::endl;
            std::cout << "  --help                   Show this help message" << std::endl;
            return 0;
        }
    }
    
    // Register signal handlers
    signal(SIGINT, signalHandler);  // Ctrl+C
    signal(SIGTERM, signalHandler); // kill command
    
    // Print system information
    std::cout << "System information:" << std::endl;
    std::cout << "  Hardware concurrency: " << std::thread::hardware_concurrency() << " threads" << std::endl;
    std::cout << "  Using " << (numThreads > 0 ? numThreads : std::thread::hardware_concurrency()) << " threads" << std::endl;
    std::cout << "  Maximum diameter: " << maxDiameter << std::endl;
    std::cout << "  Output directory: " << outputDir << std::endl;
    
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
    DiameterConstrainedQuasiCliqueSolver solver(graph, outputDir);
    solver.setMaxDiameter(maxDiameter);
    
    // Measure execution time
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Run algorithm
    std::vector<int> solution;
    
    if (!initialSolutionFile.empty()) {
        // Load initial solution
        std::vector<int> initialSolution;
        std::ifstream file(initialSolutionFile);
        
        if (file.is_open()) {
            int nodeId;
            while (file >> nodeId) {
                initialSolution.push_back(nodeId);
            }
            file.close();
            
            std::cout << "Loaded initial solution with " << initialSolution.size() << " vertices" << std::endl;
            solution = solver.expandFromExistingSolution(initialSolution, numThreads);
        } else {
            std::cerr << "Error: Could not open initial solution file " << initialSolutionFile << std::endl;
            std::cout << "Running full algorithm instead..." << std::endl;
            solution = solver.findLargeQuasiClique(numThreads);
        }
    } else {
        // Run full algorithm
        solution = solver.findLargeQuasiClique(numThreads);
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    
    // Verify and print solution
    solver.verifyAndPrintSolution(solution);
    
    std::cout << "\nAlgorithm execution time: " << duration / 1000.0 << " seconds" << std::endl;
    std::cout << "Total time (including loading): " << (duration + loadDuration) / 1000.0 << " seconds" << std::endl;
    
    // Save solution to file
    std::string solutionFile = outputDir + "quasi_clique_diameter_" + std::to_string(maxDiameter) + ".txt";
    solver.saveSolution(solution, solutionFile);
    
    // Also save with original filename for compatibility
    solver.saveSolution(solution, "solution.txt");
    
    return 0;
}