#ifndef TWO_PHASE_QUASI_CLIQUE_SOLVER_H
#define TWO_PHASE_QUASI_CLIQUE_SOLVER_H

#include "Graph.h"
#include "ThreadPool.h"
#include "CommunityDetector.h"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <atomic>
#include <string>
#include <csignal>  // For sig_atomic_t

// Forward declaration of global termination variable
extern volatile sig_atomic_t terminationRequested;

/**
 * Two-phase quasi-clique solver with community integration
 */
class TwoPhaseQuasiCliqueSolver {
private:
    const Graph& graph;
    std::unordered_map<int, double> clusteringCoefficients;
    std::vector<int> bestSolutionOverall;
    std::vector<std::vector<int>> candidateSolutions;
    std::mutex bestSolutionMutex;
    std::mutex candidateSolutionsMutex;
    std::atomic<bool> solutionFound{false};
    std::atomic<int> completedSeeds{0};
    int totalSeeds;
    CommunityDetector communityDetector;
    std::vector<int> initialSeedVertices;  // Added for warm-start capability
    
        /**
     * Compute k-core values for all vertices
     */
    std::vector<std::pair<int, int>> computeKCoreDecomposition();
    
    /**
     * Select seeds based on k-core decomposition
     */
    std::vector<int> selectSeedsBasedOnKCore(int numSeeds);

    /**
     * Pre-compute clustering coefficients
     */
    void precomputeClusteringCoefficients(int numThreads);
    
    /**
     * Calculate vertex potential
     */
    double calculateVertexPotential(int v) const;
    
    /**
     * Calculate vertex potential with community awareness
     */
    double calculateCommunityAwareVertexPotential(int v, int targetCommunity) const;
    
    /**
     * Check if a subgraph is a quasi-clique
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
     * Find boundary vertices of a solution
     */
    std::unordered_set<int> findBoundaryVertices(const std::vector<int>& solution) const;
    
    /**
     * Count connections from a vertex to a solution
     */
    int countConnectionsToSolution(int candidate, const std::vector<int>& solution) const;
    
    /**
     * Find a single quasi-clique starting from a seed vertex
     */
    std::vector<int> findQuasiCliqueFromSeed(int seed, int seedIdx, int targetCommunity = -1);
    
    /**
     * Calculate the Jaccard similarity between two sets
     */
    double calculateJaccardSimilarity(const std::vector<int>& a, const std::vector<int>& b) const;
    
    /**
     * Check if two solutions can be merged into a quasi-clique
     */
    bool canMerge(const std::vector<int>& solution1, const std::vector<int>& solution2) const;
    
    /**
     * Merge two solutions
     */
    std::vector<int> mergeSolutions(const std::vector<int>& solution1, const std::vector<int>& solution2) const;
    
    /**
     * Sort seed vertices based on community structure
     */
    std::vector<int> selectSeedsWithCommunityAwareness(const std::vector<int>& potentialSeeds, int numSeeds);
    
    /**
     * Phase 1: Find multiple candidate solutions
     */
    void phase1_findCandidateSolutions(int numSeeds, int numThreads);
    
    /**
     * Phase 2: Refine and merge solutions
     */
    void phase2_refineSolutions();

    // New helper methods for degree-based expansion
    std::vector<int> selectSeedsBasedOnDegree(int numSeeds);
    int countConnectionsToSet(int candidate, const std::vector<int>& nodeSet);
    std::vector<int> expandSolutionFromSeed(const std::vector<int>& startingSolution, int seedIdx);
    
    std::vector<int> attemptToMerge(const std::vector<int>& solution1, const std::vector<int>& solution2) const;

       /**
     * Select seeds using combined k-core and community awareness
     */
    std::vector<int> selectSeedsWithKCoreAndCommunityAwareness(int numSeeds);
    /**
     * Perform local search to improve a solution
     */
    std::vector<int> performLocalSearch(const std::vector<int>& initialSolution);

    double communityConnectivityThreshold = 0.1;

      /**
     * Improved expansion method with more thorough cluster exploration
     */
    std::vector<int> improvedExpansionMethod(const std::vector<int>& initialSolution);
    
    /**
     * Evaluate a candidate vertex for addition to a solution
     */
    double evaluateCandidate(int vertex, const std::vector<int>& solution);

        /**
     * Enhanced k-core based seed selection focusing on highest k-cores
     */
    std::vector<int> enhancedKCoreSeedSelection(int numSeeds);

        /**
     * Multi-round exploration that revisits promising areas of the graph
     */
    void multiRoundExploration(int numSeeds, int numRounds, int numThreads);
    
    /**
     * Get boundary vertices to explore from existing solutions
     */
    std::vector<int> getBoundaryVerticesToExplore(const std::vector<std::vector<int>>& solutions, int maxSeeds = 100);
    
    /**
     * Get perturbed seeds from existing solutions
     */
    std::vector<int> getPerturbedSeeds(const std::vector<std::vector<int>>& solutions, int numSeeds);
 /**
     * Quick check if two solutions have enough overlap to consider merging
     */
    bool worthTryingMerge(const std::vector<int>& a, const std::vector<int>& b);
    
    /**
     * Check if two solutions have significant overlap without calculating full Jaccard similarity
     */
    bool haveSignificantOverlap(const std::vector<int>& a, const std::vector<int>& b);
    
    /**
     * Cluster solutions based on overlap before trying merges
     */
    void fastClusterAndMergeSolutions();
    
    /**
     * Merge solutions within a single cluster
     */
    std::vector<int> mergeSolutionsInCluster(const std::vector<int>& clusterIndices);
 /**
     * Find connected components in a subgraph
     */
    std::vector<std::vector<int>> findConnectedComponents(const std::vector<int>& nodes);
    
    /**
     * Repair a solution to make it a valid quasi-clique
     */
    std::vector<int> repairSolution(const std::vector<int>& solution);
    
    /**
     * Find the largest valid subset of a solution
     */
    std::vector<int> findLargestValidSubset(const std::vector<int>& solution);

public:
    /**
     * Set the threshold for community merging
     */
    void setCommunityConnectivityThreshold(double threshold) {
        communityConnectivityThreshold = threshold;
    }
    /**
     * Constructor
     */
    TwoPhaseQuasiCliqueSolver(const Graph& g);
    
    /**
     * Load initial seed vertices from file
     */
    bool loadSeedVertices(const std::string& filename);
    
    /**
     * Find a large quasi-clique using two-phase approach
     */
    std::vector<int> findLargeQuasiClique(int numSeeds = 20, int numThreads = 1);
    
    /**
     * Verify and print information about the solution
     */
    void verifyAndPrintSolution(const std::vector<int>& solution);
    
    /**
     * Save solution to file
     */
    bool saveSolution(const std::vector<int>& solution, const std::string& filename = "solution.txt");
    
    // Methods for working with existing solutions
    std::vector<int> loadSolutionFromFile(const std::string& filename);
    std::vector<int> expandFromExistingSolution(const std::vector<int>& initialSolution, int numSeeds, int numThreads);
};

#endif // TWO_PHASE_QUASI_CLIQUE_SOLVER_H