#ifndef DIAMETER_CONSTRAINED_QUASI_CLIQUE_SOLVER_H
#define DIAMETER_CONSTRAINED_QUASI_CLIQUE_SOLVER_H

#include "Graph.h"
#include "ThreadPool.h"
#include "CommunityDetector.h"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <mutex>
#include <atomic>
#include <string>
#include <fstream>
#include <algorithm>
#include <queue>
#include <chrono>
#include <future>
#include <cmath>
#include <csignal>

// Forward declaration of global termination variable
extern volatile sig_atomic_t terminationRequested;

/**
 * Diameter-constrained quasi-clique solver that focuses on finding dense subgraphs
 * with diameter <= 3 and more than 50% of possible edges
 */
class DiameterConstrainedQuasiCliqueSolver {
private:
    const Graph& graph;
    std::unordered_map<int, double> clusteringCoefficients;
    std::vector<int> bestSolutionOverall;
    std::mutex bestSolutionMutex;
    std::mutex fileWriteMutex;
    std::atomic<int> completedStructures{0};
    int totalStructures;
    CommunityDetector communityDetector;
    
    // Caches for efficient computation
    std::unordered_map<int, std::unordered_set<int>> twoHopNeighborhoods;
    std::unordered_map<int, std::unordered_set<int>> directNeighbors;
    
    // Configuration options
    int maxDiameter = 3;
    int minCliqueSize = 5;
    bool useNodeSwapping = true;
    std::string outputDirectory = "results/";
    
    // Base file paths for saving intermediate results
    std::string maximalCliquesFile;
    std::string kCoresFile;
    std::string communitiesFile;
    std::string filteredStructuresFile;
    
    /**
     * Pre-compute clustering coefficients
     */
    void precomputeClusteringCoefficients(int numThreads);
    
    /**
     * Pre-compute two-hop neighborhoods for all vertices or a subset
     */
    void precomputeTwoHopNeighborhoods(const std::vector<int>& vertices = {});
    
    /**
     * Calculate vertex potential for expansion
     */
    double calculateVertexPotential(int v, const std::vector<int>& solution) const;
    
    /**
     * Check if a subgraph is a quasi-clique (> 50% of possible edges)
     */
    bool isQuasiClique(const std::vector<int>& nodes) const;
    
    /**
     * Count edges in a subgraph
     */
    int countEdges(const std::vector<int>& nodes) const;
    
    /**
     * Check if a solution is connected
     */
    bool isConnected(const std::vector<int>& nodes) const;
    
    /**
     * Find boundary vertices of a solution (neighbors of solution vertices)
     */
    std::unordered_set<int> findBoundaryVertices(const std::vector<int>& solution) const;
    
    /**
     * Count connections from a vertex to a solution
     */
    int countConnectionsToSolution(int candidate, const std::vector<int>& solution) const;
    
    /**
     * Calculate the exact diameter of a subgraph
     */
    int calculateDiameter(const std::vector<int>& nodes) const;
    
    /**
     * Quick check if adding a vertex could maintain diameter <= maxDiameter
     */
    bool couldMaintainDiameter(const std::vector<int>& solution, int candidate) const;
    
    /**
     * Find all maximal cliques in the graph (Bron-Kerbosch algorithm)
     */
    std::vector<std::vector<int>> findMaximalCliques();
    
    /**
     * Recursive helper for Bron-Kerbosch algorithm
     */
    void bronKerbosch(std::unordered_set<int>& R, std::unordered_set<int>& P, 
        std::unordered_set<int>& X, std::vector<std::vector<int>>& cliques, 
        int depth = 0, int maxDepth = 5);
    
    /**
     * Select pivot for Bron-Kerbosch algorithm
     */
    int selectPivot(const std::unordered_set<int>& P, const std::unordered_set<int>& X) const;
    
    /**
     * Extract k-cores for multiple k values
     */
    std::vector<std::pair<int, std::vector<int>>> extractKCores(int maxK);
    
    /**
     * Filter out structures that are contained within others
     */
    std::vector<std::vector<int>> filterContainedStructures(
        const std::vector<std::vector<int>>& structures);
    
    /**
     * Filter structures based on diameter constraint
     */
    std::vector<std::vector<int>> filterStructuresByDiameter(
        const std::vector<std::vector<int>>& structures);
    
    /**
     * Expand a solution with diameter constraint
     */
    std::vector<int> expandWithDiameterConstraint(std::vector<int> initialSolution);
    
    /**
     * Process structures in batches for better parallelization
     */
    void processBatchedStructures(const std::vector<std::vector<int>>& structures, int numThreads);
    
    /**
     * Select top structures based on quality metrics to limit memory usage
     */
    std::vector<std::vector<int>> selectTopStructures(
        const std::vector<std::vector<int>>& structures, int maxToKeep);
    
    /**
     * Calculate density of a subgraph
     */
    double calculateDensity(const std::vector<int>& structure) const;
    
    /**
     * Optimize a solution by swapping nodes based on degree and connectivity
     */
    std::vector<int> optimizeByNodeSwapping(const std::vector<int>& solution, int maxIterations = 100);
    
    /**
     * Find connected components in a subgraph
     */
    std::vector<std::vector<int>> findConnectedComponents(const std::vector<int>& nodes) const;
    
    /**
     * Repair a solution to make it a valid quasi-clique with diameter <= maxDiameter
     */
    std::vector<int> repairSolution(const std::vector<int>& solution);
    
    /**
     * Save structures to file
     */
    bool saveStructuresToFile(const std::vector<std::vector<int>>& structures, const std::string& filename);
    
    /**
     * Load structures from file
     */
    std::vector<std::vector<int>> loadStructuresFromFile(const std::string& filename);
    
    /**
     * Update the best solution found so far and save it to file
     */
    void updateBestSolution(const std::vector<int>& solution);

public:
    /**
     * Constructor
     */
    DiameterConstrainedQuasiCliqueSolver(const Graph& g, const std::string& outputDir = "results/");
    
    /**
     * Set the maximum allowed diameter
     */
    void setMaxDiameter(int diameter) {
        maxDiameter = diameter;
    }
    
    /**
     * Set the minimum clique size to consider
     */
    void setMinCliqueSize(int size) {
        minCliqueSize = size;
    }
    
    /**
     * Enable or disable node swapping optimization
     */
    void setUseNodeSwapping(bool value) {
        useNodeSwapping = value;
    }
    
    /**
     * Find a large quasi-clique with diameter constraint
     */
    std::vector<int> findLargeQuasiClique(int numThreads = 1);
    
    /**
     * Expand from an existing solution
     */
    std::vector<int> expandFromExistingSolution(const std::vector<int>& initialSolution, int numThreads = 1);
    
    /**
     * Verify and print information about the solution
     */
    void verifyAndPrintSolution(const std::vector<int>& solution);
    
    /**
     * Save solution to file
     */
    bool saveSolution(const std::vector<int>& solution, const std::string& filename = "solution.txt");
};

#endif // DIAMETER_CONSTRAINED_QUASI_CLIQUE_SOLVER_H