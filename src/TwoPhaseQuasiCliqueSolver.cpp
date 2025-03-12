#include "TwoPhaseQuasiCliqueSolver.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <queue>
#include <csignal>

using namespace std;

TwoPhaseQuasiCliqueSolver::TwoPhaseQuasiCliqueSolver(const Graph& g) 
    : graph(g), totalSeeds(0), communityDetector(g) {}

void TwoPhaseQuasiCliqueSolver::precomputeClusteringCoefficients(int numThreads) {
    cout << "Pre-computing clustering coefficients using " << numThreads << " threads..." << endl;
    vector<int> vertices = graph.getVertices();
    size_t n = vertices.size();
    
    if (n == 0) return;
    
    // Create thread pool and partition work
    ThreadPool pool(numThreads);
    mutex coeffMutex;
    atomic<int> processedCount(0);
    
    size_t chunkSize = max(size_t(1), n / (numThreads * 10));
    
    auto startTime = chrono::high_resolution_clock::now();
    
    for (size_t start = 0; start < n; start += chunkSize) {
        size_t end = min(start + chunkSize, n);
        
        pool.enqueue([this, &vertices, start, end, &coeffMutex, &processedCount, n, startTime]() {
            unordered_map<int, double> localCoeffs;
            
            for (size_t i = start; i < end; i++) {
                int v = vertices[i];
                localCoeffs[v] = graph.getClusteringCoefficient(v);
                
                int newCount = ++processedCount;
                if (newCount % 10000 == 0 || newCount == (int)n) {
                    auto currentTime = chrono::high_resolution_clock::now();
                    auto duration = chrono::duration_cast<chrono::seconds>(currentTime - startTime).count();
                    cout << "  Processed " << newCount << "/" << n << " vertices (" 
                         << (duration > 0 ? newCount / duration : newCount) << " vertices/sec)" << endl;
                }
                
                // Check for termination request
                if (terminationRequested) {
                    break;
                }
            }
            
            // Merge results
            lock_guard<mutex> lock(coeffMutex);
            clusteringCoefficients.insert(localCoeffs.begin(), localCoeffs.end());
        });
    }
    
    // Wait for all threads to finish (ThreadPool destructor handles this)
}

// Helper method to count connections to a set of nodes
int TwoPhaseQuasiCliqueSolver::countConnectionsToSet(int candidate, const vector<int>& nodeSet) {
    int connections = 0;
    const auto& neighbors = graph.getNeighbors(candidate);
    
    unordered_set<int> nodeSetLookup(nodeSet.begin(), nodeSet.end());
    for (int neighbor : neighbors) {
        if (nodeSetLookup.find(neighbor) != nodeSetLookup.end()) {
            connections++;
        }
    }
    
    return connections;
}

// Load an existing solution from a file
vector<int> TwoPhaseQuasiCliqueSolver::loadSolutionFromFile(const string& filename) {
    vector<int> solution;
    ifstream file(filename);
    if (file.is_open()) {
        int nodeId;
        while (file >> nodeId) {
            solution.push_back(nodeId);
        }
        cout << "Loaded " << solution.size() << " nodes from " << filename << endl;
    } else {
        cerr << "Error: Could not open file " << filename << endl;
    }
    return solution;
}
// Expand from an existing initial solution
vector<int> TwoPhaseQuasiCliqueSolver::expandFromExistingSolution(
                                const vector<int>& initialSolution, int numSeeds, int numThreads) {

    // Verify the initial solution is valid
    if (!initialSolution.empty()) {
        if (!isQuasiClique(initialSolution)) {
            cout << "Warning: Initial solution is not a valid quasi-clique." << endl;
        }
        if (!isConnected(initialSolution)) {
            cout << "Warning: Initial solution is not connected." << endl;
        }
    } else {
        cout << "Warning: Initial solution is empty. Proceeding with regular algorithm." << endl;
        return findLargeQuasiClique(numSeeds, numThreads);
    }

    // Start with the initial solution as our best
    bestSolutionOverall = initialSolution;

    // Determine number of threads to use
    if (numThreads <= 0) {
        numThreads = thread::hardware_concurrency();
        if (numThreads == 0) numThreads = 1;
    }

    // Pre-compute clustering coefficients
    precomputeClusteringCoefficients(numThreads);

    if (terminationRequested) {
        cout << "Termination requested during preprocessing. Exiting." << endl;
        return bestSolutionOverall;
    }

    // Detect communities
    cout << "Step 1: Detecting communities in the graph..." << endl;
    communityDetector.detectCommunities();

    if (terminationRequested) {
        cout << "Termination requested during community detection. Exiting." << endl;
        return bestSolutionOverall;
    }

    // Phase 1: Use the initial solution and expand
    cout << "Phase 1: Expanding from initial solution of " << initialSolution.size() << " nodes" << endl;

    // First approach: Use the initial solution as a single seed
    candidateSolutions.push_back(initialSolution);

    // Second approach: Use boundary vertices of the initial solution as seeds
    unordered_set<int> boundaryCandidates = findBoundaryVertices(initialSolution);
    vector<int> boundarySeeds(boundaryCandidates.begin(), boundaryCandidates.end());

    // Sort boundary seeds by degree
    sort(boundarySeeds.begin(), boundarySeeds.end(), [this](int a, int b) {
        return graph.getDegree(a) > graph.getDegree(b);
    });

    // Limit the number of boundary seeds to use
    int boundaryCount = min((int)boundarySeeds.size(), numSeeds);
    boundarySeeds.resize(boundaryCount);

    // Also add high-degree nodes not in boundary as potential seeds
    vector<int> degreeSeeds = selectSeedsBasedOnDegree(numSeeds);
    for (int seed : degreeSeeds) {
        // Check if seed is already in boundary seeds
        if (find(boundarySeeds.begin(), boundarySeeds.end(), seed) == boundarySeeds.end()) {
            boundarySeeds.push_back(seed);
            // if (boundarySeeds.size() >= numSeeds) break;
            if (boundarySeeds.size() >= static_cast<size_t>(numSeeds)) break;
        }
    }

    // Process these seeds in parallel
    ThreadPool pool(numThreads);
    totalSeeds = boundarySeeds.size();
    completedSeeds = 0;

    for (size_t seedIdx = 0; seedIdx < boundarySeeds.size(); seedIdx++) {
        int seed = boundarySeeds[seedIdx];
        
        pool.enqueue([this, seed, seedIdx, initialSolution]() {
            if (terminationRequested) return;
            
            // Start with the initial solution plus this seed
            vector<int> solution = initialSolution;
            if (find(solution.begin(), solution.end(), seed) == solution.end()) {
                solution.push_back(seed);
            }
            
            // Continue expanding from here
            vector<int> expanded = expandSolutionFromSeed(solution, seedIdx);
            
            // Update best solution if better
            if (expanded.size() > bestSolutionOverall.size() && 
                isQuasiClique(expanded) && isConnected(expanded)) {
                lock_guard<mutex> lock(bestSolutionMutex);
                if (expanded.size() > bestSolutionOverall.size()) {
                    bestSolutionOverall = expanded;
                    solutionFound = true;
                    
                    // Save progress
                    ofstream solutionFile("solution_in_progress.txt");
                    for (int v : bestSolutionOverall) {
                        solutionFile << v << endl;
                    }
                    solutionFile.close();
                    
                    cout << "New best solution found: " << bestSolutionOverall.size() << " vertices" << endl;
                }
            }
            
            // Add to candidate solutions for phase 2
            if (expanded.size() > initialSolution.size() && 
                isQuasiClique(expanded) && isConnected(expanded)) {
                lock_guard<mutex> lock(candidateSolutionsMutex);
                candidateSolutions.push_back(expanded);
            }
            
            completedSeeds++;
        });
    }

    // Wait for all expansions to complete
    while (completedSeeds < totalSeeds && !terminationRequested) {
        this_thread::sleep_for(chrono::seconds(5));
        cout << "Progress: " << completedSeeds << "/" << totalSeeds 
            << " seeds processed, candidate solutions: " << candidateSolutions.size() << endl;
    }

    // Sort candidate solutions by size
    sort(candidateSolutions.begin(), candidateSolutions.end(), 
        [](const vector<int>& a, const vector<int>& b) { return a.size() > b.size(); });

    // Keep only top 100 solutions to limit computational complexity in phase 2
    if (candidateSolutions.size() > 300) {
        std::cout << "Currently, we have " << candidateSolutions.size() << " candidate solutions.\n";
        cout << "Limiting to top 300 candidate solutions for phase 2" << endl;
        candidateSolutions.resize(300);
    }

    // Phase 2: Refine and merge solutions
    phase2_refineSolutions();

    // Random Restart Logic 
    // Check if the best solution so far didn't improve much compared to the initial solution.
    if (bestSolutionOverall.size() <= initialSolution.size() + 5) {
        cout << "***************Solution not improving significantly. Trying random restarts..." << endl;

        // We'll create 10 perturbed solutions
        vector<vector<int>> perturbedSolutions;
        
        for (int i = 0; i < 10; i++) {
            vector<int> perturbed = initialSolution;
            
            // Remove 10-20% of nodes randomly
            if (!perturbed.empty()) {
                int removeCount = static_cast<int>( 
                        perturbed.size() * (0.1 + (rand() % 10) / 100.0) 
                );
                for (int j = 0; j < removeCount; j++) {
                    if (perturbed.empty()) break;
                    int indexToRemove = rand() % perturbed.size();
                    perturbed.erase(perturbed.begin() + indexToRemove);
                }
            }
            
            // Add some high-degree neighbors
            unordered_set<int> boundary;
            unordered_set<int> solutionSet(perturbed.begin(), perturbed.end());
            
            for (int node : perturbed) {
                for (int neighbor : graph.getNeighbors(node)) {
                    if (solutionSet.find(neighbor) == solutionSet.end()) {
                        boundary.insert(neighbor);
                    }
                }
            }
            
            vector<int> boundaryVec(boundary.begin(), boundary.end());
            sort(boundaryVec.begin(), boundaryVec.end(), [this](int a, int b) {
                return graph.getDegree(a) > graph.getDegree(b);
            });
            
            // Try adding up to 10 highest-degree neighbors
            for (int j = 0; j < min(10, (int)boundaryVec.size()); j++) {
                perturbed.push_back(boundaryVec[j]);
                if (!isQuasiClique(perturbed) || !isConnected(perturbed)) {
                    // If adding this node breaks quasi-clique or connectivity,
                    // remove it and stop adding more
                    perturbed.pop_back();
                }
            }
            
            // Only keep if it's valid
            if (isQuasiClique(perturbed) && isConnected(perturbed)) {
                perturbedSolutions.push_back(perturbed);
            }
        }
        
        // Add these perturbed solutions to candidate solutions
        for (const auto& perturbed : perturbedSolutions) {
            candidateSolutions.push_back(perturbed);
        }
        
        // Run another round of merging after random restarts
        cout << "Running additional merge phase with perturbed solutions..." << endl;
        phase2_refineSolutions();
    }

    return bestSolutionOverall;
    }

// Expand from an existing solution
vector<int> TwoPhaseQuasiCliqueSolver::expandSolutionFromSeed(
    const vector<int>& startingSolution, int seedIdx) {

cout << "  Expanding solution " << seedIdx + 1 << "/" << totalSeeds 
     << " (size: " << startingSolution.size() << ")" << endl;

vector<int> solution = startingSolution;

// Use all nodes in the solution as "recently added" for prioritization
vector<int> recentlyAdded = startingSolution;
if (recentlyAdded.size() > 10) {
    // If too many, just use the last 10 to be more efficient
    recentlyAdded = vector<int>(
        solution.end() - min(10, (int)solution.size()), 
        solution.end()
    );
}

// Expansion phase
int iteration = 0;
while (!terminationRequested) {
    iteration++;
    int bestCandidate = -1;
    double bestScore = -1;
    
    // Find boundary vertices
    unordered_set<int> boundary = findBoundaryVertices(solution);
    
    // Early stopping if no boundary vertices
    if (boundary.empty()) {
        break;
    }
    
    // Prioritize neighbors of recently added nodes
    vector<int> priorityNodes;
    for (int recent : recentlyAdded) {
        for (int neighbor : graph.getNeighbors(recent)) {
            if (boundary.count(neighbor) > 0) {
                priorityNodes.push_back(neighbor);
            }
        }
    }
    
    // Create a set for quick lookup
    unordered_set<int> processed;
    
    // First evaluate priority nodes
    for (int candidate : priorityNodes) {
        if (processed.count(candidate) > 0) continue;
        processed.insert(candidate);
        
        // Enhanced score calculation with bonus for connections to recent nodes
        int connections = countConnectionsToSolution(candidate, solution);
        int recentConnections = countConnectionsToSet(candidate, recentlyAdded);
        
        double directRatio = static_cast<double>(connections) / solution.size();
        double recentBonus = static_cast<double>(recentConnections) / recentlyAdded.size();
        
        // Get clustering coefficient
        auto it = clusteringCoefficients.find(candidate);
        double candidateClustering = (it != clusteringCoefficients.end()) ? it->second : 0.0;
        
        // Adaptive alpha with emphasis on recent connections
        double alpha = max(0.5, 0.95 - 0.005 * solution.size());
        double score = alpha * directRatio + (1 - alpha) * candidateClustering + 0.2 * recentBonus;
        
        // Check if adding maintains quasi-clique property
        vector<int> newSolution = solution;
        newSolution.push_back(candidate);
        
        if (isQuasiClique(newSolution) && isConnected(newSolution) && score > bestScore) {
            bestScore = score;
            bestCandidate = candidate;
        }
    }
    
    // Then evaluate remaining boundary vertices if needed
    if (bestCandidate == -1) {
        for (int candidate : boundary) {
            if (processed.count(candidate) > 0) continue;
            processed.insert(candidate);
            
            // Regular scoring as before
            int connections = countConnectionsToSolution(candidate, solution);
            double directRatio = static_cast<double>(connections) / solution.size();
            
            // Get clustering coefficient
            auto it = clusteringCoefficients.find(candidate);
            double candidateClustering = (it != clusteringCoefficients.end()) ? it->second : 0.0;
            
            // Adaptive alpha
            double alpha = max(0.5, 0.95 - 0.005 * solution.size());
            double score = alpha * directRatio + (1 - alpha) * candidateClustering;
            
            // Check if adding maintains quasi-clique property
            vector<int> newSolution = solution;
            newSolution.push_back(candidate);
            
            if (isQuasiClique(newSolution) && isConnected(newSolution) && score > bestScore) {
                bestScore = score;
                bestCandidate = candidate;
            }
        }
    }
    
    // If no suitable candidate found, break
    if (bestCandidate == -1) {
        cout << "    No more candidates found after " << iteration << " iterations" << endl;
        break;
    }
    
    // Add the best candidate
    solution.push_back(bestCandidate);
    
    // Update recently added nodes (keep a sliding window of the last 10 added)
    recentlyAdded.push_back(bestCandidate);
    if (recentlyAdded.size() > 10) {
        recentlyAdded.erase(recentlyAdded.begin());
    }
    
    // Progress reporting
    if (iteration % 10 == 0) {
        cout << "    Iteration " << iteration << ": solution size = " << solution.size() << endl;
    }
    
    // Check if solution is better than current best, and if so, save it immediately
    if (solution.size() > 5) {
        lock_guard<mutex> lock(bestSolutionMutex);
        if (solution.size() > bestSolutionOverall.size() && isQuasiClique(solution) && isConnected(solution)) {
            bestSolutionOverall = solution;
            solutionFound = true;
            
            // Save progress
            ofstream solutionFile("solution_in_progress.txt");
            for (int v : bestSolutionOverall) {
                solutionFile << v << endl;
            }
            solutionFile.close();
            
            cout << "  New best solution found: " << bestSolutionOverall.size() << " vertices (saved to solution_in_progress.txt)" << endl;
        }
    }
}

cout << "  Final expanded solution size: " << solution.size() << endl;
return solution;
}

// Select seeds based purely on node degree
vector<int> TwoPhaseQuasiCliqueSolver::selectSeedsBasedOnDegree(int numSeeds) {
    vector<int> vertices = graph.getVertices();
    
    // Sort vertices by degree (descending)
    sort(vertices.begin(), vertices.end(), [this](int a, int b) {
        return graph.getDegree(a) > graph.getDegree(b);
    });
    
    // Take the top numSeeds high-degree vertices
    int seedsToSelect = min(numSeeds, (int)vertices.size());
    vector<int> selectedSeeds(vertices.begin(), vertices.begin() + seedsToSelect);
    
    cout << "Selected " << selectedSeeds.size() << " seeds based on degree" << endl;
    
    return selectedSeeds;
}

double TwoPhaseQuasiCliqueSolver::calculateVertexPotential(int v) const {
    auto it = clusteringCoefficients.find(v);
    double clustering = (it != clusteringCoefficients.end()) ? it->second : 0.0;
    double degree = graph.getDegree(v);
    return 0.7 * degree + 0.3 * clustering * degree;
}

double TwoPhaseQuasiCliqueSolver::calculateCommunityAwareVertexPotential(int v, int targetCommunity) const {
    double basePotential = calculateVertexPotential(v);
    
    // If vertex is in the target community, give it a boost
    if (communityDetector.getCommunity(v) == targetCommunity) {
        return basePotential * 1.1;
    }
    
    return basePotential;
}

bool TwoPhaseQuasiCliqueSolver::isQuasiClique(const vector<int>& nodes) const {
    int n = nodes.size();
    int possibleEdges = (n * (n - 1)) / 2;
    int minEdgesNeeded = (possibleEdges / 2) + 1; // Strictly more than half
    
    int actualEdges = countEdges(nodes);
    return actualEdges >= minEdgesNeeded;
}

int TwoPhaseQuasiCliqueSolver::countEdges(const vector<int>& nodes) const {
    int count = 0;
    
    // Use a set for quick membership testing
    unordered_set<int> nodeSet(nodes.begin(), nodes.end());
    
    // Iterate through all pairs (optimized to only check each edge once)
    for (size_t i = 0; i < nodes.size(); i++) {
        const auto& neighbors = graph.getNeighbors(nodes[i]);
        
        for (int neighbor : neighbors) {
            // Only count if neighbor is in the subgraph and has a higher index (to avoid counting twice)
            if (nodeSet.find(neighbor) != nodeSet.end() && neighbor > nodes[i]) {
                count++;
            }
        }
    }
    
    return count;
}

bool TwoPhaseQuasiCliqueSolver::isConnected(const vector<int>& nodes) const {
    if (nodes.empty()) return true;
    
    // Create adjacency list for the subgraph
    unordered_map<int, vector<int>> subgraphAdj;
    for (int u : nodes) {
        subgraphAdj[u] = vector<int>();
        for (int v : nodes) {
            if (u != v && graph.hasEdge(u, v)) {
                subgraphAdj[u].push_back(v);
            }
        }
    }
    
    // Run BFS to check connectivity
    unordered_set<int> visited;
    queue<int> q;
    q.push(nodes[0]);
    visited.insert(nodes[0]);
    
    while (!q.empty()) {
        int current = q.front();
        q.pop();
        
        for (int neighbor : subgraphAdj[current]) {
            if (visited.find(neighbor) == visited.end()) {
                visited.insert(neighbor);
                q.push(neighbor);
            }
        }
    }
    
    return visited.size() == nodes.size();
}

unordered_set<int> TwoPhaseQuasiCliqueSolver::findBoundaryVertices(const vector<int>& solution) const {
    unordered_set<int> boundary;
    unordered_set<int> solutionSet(solution.begin(), solution.end());
    
    for (int v : solution) {
        const auto& neighbors = graph.getNeighbors(v);
        
        for (int neighbor : neighbors) {
            if (solutionSet.find(neighbor) == solutionSet.end()) {
                boundary.insert(neighbor);
            }
        }
    }
    
    return boundary;
}

int TwoPhaseQuasiCliqueSolver::countConnectionsToSolution(int candidate, const vector<int>& solution) const {
    int connections = 0;
    const auto& neighbors = graph.getNeighbors(candidate);
    
    // Use a set for larger solutions
    if (solution.size() > 20) {
        unordered_set<int> solutionSet(solution.begin(), solution.end());
        
        for (int neighbor : neighbors) {
            if (solutionSet.find(neighbor) != solutionSet.end()) {
                connections++;
            }
        }
    } else {
        // Use direct iteration for smaller solutions (faster than set construction)
        for (int v : solution) {
            if (find(neighbors.begin(), neighbors.end(), v) != neighbors.end()) {
                connections++;
            }
        }
    }
    
    return connections;
}
// Methods for finding and merging solutions

vector<int> TwoPhaseQuasiCliqueSolver::findQuasiCliqueFromSeed(int seed, int seedIdx, int targetCommunity) {
    cout << "Processing seed " << seedIdx + 1 << "/" << totalSeeds << endl;
    cout << "  Starting from seed: " << seed 
         << " (degree: " << graph.getDegree(seed) 
         << ", clustering: " << clusteringCoefficients[seed];
    
    if (targetCommunity >= 0) {
        cout << ", community: " << communityDetector.getCommunity(seed) << ")";
    } else {
        cout << ")";
    }
    cout << endl;
    
    // Initialize with single seed
    vector<int> solution = {seed};
    
    // Track recently added nodes for prioritization (sliding window)
    vector<int> recentlyAdded = {seed};
    
    // Expansion phase
    int iteration = 0;
    while (!terminationRequested) {
        iteration++;
        int bestCandidate = -1;
        double bestScore = -1;
        
        // Find boundary vertices
        unordered_set<int> boundary = findBoundaryVertices(solution);
        
        // Early stopping if no boundary vertices
        if (boundary.empty()) {
            break;
        }
        
        // Prioritize neighbors of recently added nodes
        vector<int> priorityNodes;
        for (int recent : recentlyAdded) {
            for (int neighbor : graph.getNeighbors(recent)) {
                if (boundary.count(neighbor) > 0) {
                    priorityNodes.push_back(neighbor);
                }
            }
        }
        
        // Create a set for quick lookup of processed nodes
        unordered_set<int> processed;
        
        // First evaluate priority nodes
        for (int candidate : priorityNodes) {
            if (processed.count(candidate) > 0) continue;
            processed.insert(candidate);
            
            // Enhanced score calculation with bonus for connections to recent nodes
            int connections = countConnectionsToSolution(candidate, solution);
            int recentConnections = countConnectionsToSet(candidate, recentlyAdded);
            
            double directRatio = static_cast<double>(connections) / solution.size();
            double recentBonus = static_cast<double>(recentConnections) / recentlyAdded.size();
            
            // Get clustering coefficient
            auto it = clusteringCoefficients.find(candidate);
            double candidateClustering = (it != clusteringCoefficients.end()) ? it->second : 0.0;
            
            // Adaptive alpha with emphasis on recent connections
            double alpha = max(0.5, 0.95 - 0.005 * solution.size());
            double score = alpha * directRatio + (1 - alpha) * candidateClustering + 0.2 * recentBonus;
            
            // Community awareness: boost score if candidate is in target community
            if (targetCommunity >= 0 && communityDetector.getCommunity(candidate) == targetCommunity) {
                score *= 1.1;
            }
            
            // Check if adding maintains quasi-clique property
            vector<int> newSolution = solution;
            newSolution.push_back(candidate);
            
            if (isQuasiClique(newSolution) && isConnected(newSolution) && score > bestScore) {
                bestScore = score;
                bestCandidate = candidate;
            }
        }
        
        // Then evaluate remaining boundary vertices if needed
        if (bestCandidate == -1) {
            for (int candidate : boundary) {
                if (processed.count(candidate) > 0) continue;
                processed.insert(candidate);
                
                // Regular scoring with community awareness
                int connections = countConnectionsToSolution(candidate, solution);
                
                double directRatio = static_cast<double>(connections) / solution.size();
                
                // Get clustering coefficient
                auto it = clusteringCoefficients.find(candidate);
                double candidateClustering = (it != clusteringCoefficients.end()) ? it->second : 0.0;
                
                // Adaptive alpha
                double alpha = max(0.5, 0.95 - 0.005 * solution.size());
                double score = alpha * directRatio + (1 - alpha) * candidateClustering;
                
                // Community awareness: boost score if candidate is in target community
                if (targetCommunity >= 0 && communityDetector.getCommunity(candidate) == targetCommunity) {
                    score *= 1.1;
                }
                
                // Check if adding maintains quasi-clique property
                vector<int> newSolution = solution;
                newSolution.push_back(candidate);
                
                if (isQuasiClique(newSolution) && isConnected(newSolution) && score > bestScore) {
                    bestScore = score;
                    bestCandidate = candidate;
                }
            }
        }
        
        // If no suitable candidate found, break
        if (bestCandidate == -1) {
            cout << "    No more candidates found after " << iteration << " iterations" << endl;
            break;
        }
        
        // Add the best candidate
        solution.push_back(bestCandidate);
        
        // Update recently added nodes (keep a sliding window of the last 10 added)
        recentlyAdded.push_back(bestCandidate);
        if (recentlyAdded.size() > 10) {
            recentlyAdded.erase(recentlyAdded.begin());
        }
        
        // Progress reporting
        if (iteration % 10 == 0) {
            cout << "    Iteration " << iteration << ": solution size = " << solution.size() << endl;
        }
        
        // Check if solution is better than current best, and if so, save it immediately
        if (solution.size() > 5) {  // Only consider solutions of reasonable size
            lock_guard<mutex> lock(bestSolutionMutex);
            if (solution.size() > bestSolutionOverall.size() && isQuasiClique(solution) && isConnected(solution)) {
                bestSolutionOverall = solution;
                solutionFound = true;
                
                // Write the current best solution to file (overwrites previous)
                ofstream solutionFile("solution_in_progress.txt");
                for (int v : bestSolutionOverall) {
                    solutionFile << v << endl;
                }
                solutionFile.close();
                
                cout << "  New best solution found: " << bestSolutionOverall.size() << " vertices (saved to solution_in_progress.txt)" << endl;
            }
        }
    }
    
    cout << "  Final solution size: " << solution.size() << endl;
    completedSeeds++;
    
    // Add to candidate solutions if it's a valid quasi-clique and of reasonable size
    if (solution.size() > 5 && isQuasiClique(solution) && isConnected(solution)) {
        lock_guard<mutex> lock(candidateSolutionsMutex);
        candidateSolutions.push_back(solution);
    }
    
    return solution;
}

double TwoPhaseQuasiCliqueSolver::calculateJaccardSimilarity(const vector<int>& a, const vector<int>& b) const {
    unordered_set<int> setA(a.begin(), a.end());
    unordered_set<int> setB(b.begin(), b.end());
    
    int intersection = 0;
    for (int item : setB) {
        if (setA.find(item) != setA.end()) {
            intersection++;
        }
    }
    
    int unionSize = setA.size() + setB.size() - intersection;
    return static_cast<double>(intersection) / unionSize;
}

bool TwoPhaseQuasiCliqueSolver::canMerge(const vector<int>& solution1, const vector<int>& solution2) const {
    // Combine the solutions
    vector<int> combined;
    combined.reserve(solution1.size() + solution2.size());
    
    // Add all vertices from solution1
    combined.insert(combined.end(), solution1.begin(), solution1.end());
    
    // Add unique vertices from solution2
    unordered_set<int> solution1Set(solution1.begin(), solution1.end());
    for (int v : solution2) {
        if (solution1Set.find(v) == solution1Set.end()) {
            combined.push_back(v);
        }
    }
    
    // Check if the combined solution is a valid quasi-clique and connected
    return isQuasiClique(combined) && isConnected(combined);
}

vector<int> TwoPhaseQuasiCliqueSolver::mergeSolutions(const vector<int>& solution1, const vector<int>& solution2) const {
    vector<int> merged;
    merged.reserve(solution1.size() + solution2.size());
    
    // Add all vertices from solution1
    merged.insert(merged.end(), solution1.begin(), solution1.end());
    
    // Add unique vertices from solution2
    unordered_set<int> solution1Set(solution1.begin(), solution1.end());
    for (int v : solution2) {
        if (solution1Set.find(v) == solution1Set.end()) {
            merged.push_back(v);
        }
    }
    
    return merged;
}

vector<int> TwoPhaseQuasiCliqueSolver::selectSeedsWithCommunityAwareness(const vector<int>& potentialSeeds, int numSeeds) {
    if (numSeeds >= (int)potentialSeeds.size()) {
        return potentialSeeds;
    }
    
    // Get community sizes and sort communities by size (descending)
    vector<int> sizes = communityDetector.getCommunitySizes();
    vector<pair<int, int>> communitySizes;
    for (int i = 0; i < (int)sizes.size(); i++) {
        communitySizes.push_back({i, sizes[i]});
    }
    
    sort(communitySizes.begin(), communitySizes.end(), 
         [](const pair<int, int>& a, const pair<int, int>& b) { 
             return a.second > b.second;
         });
    
    // Allocate seeds proportionally to community sizes
    vector<int> selectedSeeds;
    selectedSeeds.reserve(numSeeds);
    
    // First, get boundary vertices (vertices connecting different communities)
    vector<int> boundaryVertices = communityDetector.findBoundaryVertices();
    
    // Sort boundary vertices by potential
    sort(boundaryVertices.begin(), boundaryVertices.end(), [this](int a, int b) {
        return calculateVertexPotential(a) > calculateVertexPotential(b);
    });
    
    // Take top 20% of seeds from boundary vertices
    int boundaryCount = min(numSeeds / 5, (int)boundaryVertices.size());
    for (int i = 0; i < boundaryCount; i++) {
        selectedSeeds.push_back(boundaryVertices[i]);
    }
    
    // Allocate remaining seeds to communities proportionally
    int remainingSeeds = numSeeds - boundaryCount;
    vector<int> seedsPerCommunity(communityDetector.getNumCommunities(), 0);
    
    // Calculate total size of all communities
    int totalSize = 0;
    for (const auto& pair : communitySizes) {
        totalSize += pair.second;
    }
    
    // First pass: allocate integer number of seeds
    int allocatedSeeds = 0;
    for (const auto& pair : communitySizes) {
        int community = pair.first;
        int size = pair.second;
        
        int alloc = (size * remainingSeeds) / totalSize;
        seedsPerCommunity[community] = alloc;
        allocatedSeeds += alloc;
    }
    
    // Second pass: allocate any remaining seeds to largest communities
    int leftoverSeeds = remainingSeeds - allocatedSeeds;
    for (int i = 0; i < leftoverSeeds && i < (int)communitySizes.size(); i++) {
        seedsPerCommunity[communitySizes[i].first]++;
    }
    
    // Now select top vertices from each community
    for (int community = 0; community < (int)seedsPerCommunity.size(); community++) {
        int seedsToSelect = seedsPerCommunity[community];
        if (seedsToSelect == 0) continue;
        
        // Get vertices in this community
        vector<int> communityVertices;
        for (int v : potentialSeeds) {
            if (communityDetector.getCommunity(v) == community) {
                communityVertices.push_back(v);
            }
        }
        
        // Sort by potential
        sort(communityVertices.begin(), communityVertices.end(), [this](int a, int b) {
            return calculateVertexPotential(a) > calculateVertexPotential(b);
        });
        
        // Select top seeds
        for (int i = 0; i < seedsToSelect && i < (int)communityVertices.size(); i++) {
            selectedSeeds.push_back(communityVertices[i]);
        }
    }
    
    return selectedSeeds;
}


void TwoPhaseQuasiCliqueSolver::phase1_findCandidateSolutions(int numSeeds, int numThreads) {
    cout << "Phase 1: Finding candidate solutions using community-aware approach..." << endl;
    
    // Sort vertices by potential
    vector<int> vertices = graph.getVertices();
    cout << "Sorting " << vertices.size() << " vertices by potential..." << endl;
    
    sort(vertices.begin(), vertices.end(), [this](int a, int b) {
        return calculateVertexPotential(a) > calculateVertexPotential(b);
    });
    
    // Select seeds with community awareness
    vector<int> seeds = selectSeedsWithCommunityAwareness(vertices, numSeeds);
    totalSeeds = seeds.size();
    
    cout << "Selected " << seeds.size() << " seeds with community awareness" << endl;
    
    // Process seeds in parallel
    ThreadPool pool(numThreads);
    
    for (size_t seedIdx = 0; seedIdx < seeds.size(); seedIdx++) {
        int seed = seeds[seedIdx];
        int community = communityDetector.getCommunity(seed);
        
        pool.enqueue([this, seed, seedIdx, community]() {
            if (terminationRequested) return;
            
            vector<int> solution = findQuasiCliqueFromSeed(seed, seedIdx, community);
            
            // Update best solution if better
            if (solution.size() > 5 && isQuasiClique(solution) && isConnected(solution)) {
                lock_guard<mutex> lock(bestSolutionMutex);
                if (solution.size() > bestSolutionOverall.size()) {
                    bestSolutionOverall = solution;
                    solutionFound = true;
                    cout << "New best solution found: " << bestSolutionOverall.size() << " vertices" << endl;
                }
            }
        });
    }
    
    // Wait for all threads to finish
    while (completedSeeds < totalSeeds && !terminationRequested) {
        this_thread::sleep_for(chrono::seconds(5));
        cout << "Progress: " << completedSeeds << "/" << totalSeeds 
             << " seeds processed, candidate solutions: " << candidateSolutions.size() << endl;
    }
    
    cout << "Phase 1 complete. Found " << candidateSolutions.size() << " candidate solutions." << endl;
    
    // Sort candidate solutions by size (descending)
    sort(candidateSolutions.begin(), candidateSolutions.end(), 
         [](const vector<int>& a, const vector<int>& b) { return a.size() > b.size(); });
    
    // Keep only top 100 solutions to limit computational complexity in phase 2
    if (candidateSolutions.size() > 100) {
        cout << "Limiting to top 100 candidate solutions for phase 2" << endl;
        candidateSolutions.resize(100);
    }
}

void TwoPhaseQuasiCliqueSolver::phase2_refineSolutions() {
    cout << "Phase 2: Refining and merging solutions..." << endl;
    
    if (candidateSolutions.empty()) {
        cout << "No candidate solutions to refine." << endl;
        return;
    }
    
    // First, try to merge solutions
    cout << "Attempting to merge solutions..." << endl;
    
    bool improved = true;
    int mergeIterations = 0;
    
    // Modified: increased max iterations to 10 and adjusted similarity thresholds
    while (improved && !terminationRequested && mergeIterations < 10) {
        improved = false;
        mergeIterations++;
        
        cout << "Merge iteration " << mergeIterations << endl;
        
        vector<vector<int>> newSolutions;
        
        // Try all pairs of solutions
        for (size_t i = 0; i < candidateSolutions.size(); i++) {
            for (size_t j = i + 1; j < candidateSolutions.size(); j++) {
                // Calculate Jaccard similarity to quickly filter out unlikely pairs
                double similarity = calculateJaccardSimilarity(candidateSolutions[i], candidateSolutions[j]);
                
                // Modified: expanded similarity range from [0.1, 0.8] to [0.05, 0.9]
                if (similarity >= 0.05 && similarity <= 0.9) {
                    if (canMerge(candidateSolutions[i], candidateSolutions[j])) {
                        vector<int> merged = mergeSolutions(candidateSolutions[i], candidateSolutions[j]);
                        
                        if (merged.size() > max(candidateSolutions[i].size(), candidateSolutions[j].size())) {
                            newSolutions.push_back(merged);
                            improved = true;
                            cout << "  Merged solutions of sizes " << candidateSolutions[i].size() 
                                 << " and " << candidateSolutions[j].size() 
                                 << " into new solution of size " << merged.size() << endl;
                        }
                    }
                }
                
                if (terminationRequested) break;
            }
            if (terminationRequested) break;
        }
        
        // Add new solutions to candidate pool
        for (const auto& solution : newSolutions) {
            candidateSolutions.push_back(solution);
        }
        
        // Sort solutions by size (descending)
        sort(candidateSolutions.begin(), candidateSolutions.end(), 
             [](const vector<int>& a, const vector<int>& b) { return a.size() > b.size(); });
        
        cout << "  After merge iteration " << mergeIterations 
             << ": " << candidateSolutions.size() << " candidate solutions" << endl;
        
        // Update best solution
        if (!candidateSolutions.empty() && candidateSolutions[0].size() > bestSolutionOverall.size()) {
            bestSolutionOverall = candidateSolutions[0];
            
            // Save best solution
            ofstream solutionFile("solution_in_progress.txt");
            for (int v : bestSolutionOverall) {
                solutionFile << v << endl;
            }
            solutionFile.close();
            
            cout << "New best solution found: " << bestSolutionOverall.size() << " vertices" << endl;
        }
        
        // Limit candidates to keep computational complexity manageable
        if (candidateSolutions.size() > 100) {
            candidateSolutions.resize(100);
        }
    }
    
    cout << "Phase 2 complete. Best solution size: " << bestSolutionOverall.size() << endl;
}

// void TwoPhaseQuasiCliqueSolver::phase2_refineSolutions() {
//     cout << "Phase 2: Refining and merging solutions (aggressive mode)..." << endl;
    
//     if (candidateSolutions.empty()) {
//         cout << "No candidate solutions to refine." << endl;
//         return;
//     }
    
//     // First, try to merge solutions
//     cout << "Attempting to merge solutions..." << endl;
    
//     bool improved = true;
//     int mergeIterations = 0;
    
//     // Increase max iterations and use much more relaxed similarity thresholds
//     while (improved && !terminationRequested && mergeIterations < 20) { // Increased from 10 to 20
//         improved = false;
//         mergeIterations++;
        
//         cout << "Merge iteration " << mergeIterations << endl;
        
//         vector<vector<int>> newSolutions;
        
//         // Try all pairs of solutions
//         for (size_t i = 0; i < candidateSolutions.size(); i++) {
//             for (size_t j = i + 1; j < candidateSolutions.size(); j++) {
//                 // Calculate Jaccard similarity
//                 double similarity = calculateJaccardSimilarity(candidateSolutions[i], candidateSolutions[j]);
                
//                 // Extremely relaxed similarity range - try almost anything
//                 if (similarity >= 0.01 && similarity <= 0.99) { // Changed from 0.05-0.9 to 0.01-0.99
//                     // Try to merge even if we're not 100% sure it will work
//                     vector<int> merged = attemptToMerge(candidateSolutions[i], candidateSolutions[j]);
                    
//                     if (!merged.empty() && merged.size() > max(candidateSolutions[i].size(), candidateSolutions[j].size())) {
//                         newSolutions.push_back(merged);
//                         improved = true;
//                         cout << "  Merged solutions of sizes " << candidateSolutions[i].size() 
//                              << " and " << candidateSolutions[j].size() 
//                              << " into new solution of size " << merged.size() << endl;
//                     }
//                 }
                
//                 if (terminationRequested) break;
//             }
//             if (terminationRequested) break;
//         }
        
//         // Add new solutions to candidate pool
//         for (const auto& solution : newSolutions) {
//             candidateSolutions.push_back(solution);
//         }
        
//         // Sort solutions by size (descending)
//         sort(candidateSolutions.begin(), candidateSolutions.end(), 
//              [](const vector<int>& a, const vector<int>& b) { return a.size() > b.size(); });
        
//         cout << "  After merge iteration " << mergeIterations 
//              << ": " << candidateSolutions.size() << " candidate solutions" << endl;
        
//         // Update best solution
//         if (!candidateSolutions.empty() && candidateSolutions[0].size() > bestSolutionOverall.size()) {
//             bestSolutionOverall = candidateSolutions[0];
            
//             // Save best solution
//             ofstream solutionFile("solution_in_progress.txt");
//             for (int v : bestSolutionOverall) {
//                 solutionFile << v << endl;
//             }
//             solutionFile.close();
            
//             cout << "New best solution found: " << bestSolutionOverall.size() << " vertices" << endl;
//         }
        
//         // Keep more candidates
//         if (candidateSolutions.size() > 300) { // Increased from 100 to 200
//             candidateSolutions.resize(300);
//         }
//     }
    
//     cout << "Phase 2 complete. Best solution size: " << bestSolutionOverall.size() << endl;
// }

// New helper method for aggressive merging
vector<int> TwoPhaseQuasiCliqueSolver::attemptToMerge(
        const vector<int>& solution1, const vector<int>& solution2) const {
    
    // Combine the solutions
    vector<int> combined;
    combined.reserve(solution1.size() + solution2.size());
    
    // Add all vertices from solution1
    combined.insert(combined.end(), solution1.begin(), solution1.end());
    
    // Add unique vertices from solution2
    unordered_set<int> solution1Set(solution1.begin(), solution1.end());
    for (int v : solution2) {
        if (solution1Set.find(v) == solution1Set.end()) {
            combined.push_back(v);
        }
    }
    
    // If the combined solution is a valid quasi-clique and connected, return it
    if (isQuasiClique(combined) && isConnected(combined)) {
        return combined;
    }
    
    // If not valid, try to make it valid by removing low-connectivity nodes
    vector<pair<int, double>> nodeConnectivity;
    for (int node : combined) {
        int connections = 0;
        for (int other : combined) {
            if (node != other && graph.hasEdge(node, other)) {
                connections++;
            }
        }
        double connectivity = (double)connections / (combined.size() - 1);
        nodeConnectivity.push_back({node, connectivity});
    }
    
    // Sort by connectivity (ascending)
    sort(nodeConnectivity.begin(), nodeConnectivity.end(), 
         [](const pair<int, double>& a, const pair<int, double>& b) {
             return a.second < b.second;
         });
    
    // Try removing up to 30% of the lowest-connectivity nodes
    int maxToRemove = combined.size() * 0.3;
    vector<int> prunedSolution = combined;
    
    for (int i = 0; i < min(maxToRemove, (int)nodeConnectivity.size()); i++) {
        int nodeToRemove = nodeConnectivity[i].first;
        prunedSolution.erase(remove(prunedSolution.begin(), prunedSolution.end(), nodeToRemove), 
                             prunedSolution.end());
        
        if (isQuasiClique(prunedSolution) && isConnected(prunedSolution) && 
            prunedSolution.size() > max(solution1.size(), solution2.size())) {
            return prunedSolution;
        }
    }
    
    // If we couldn't make a valid solution, return empty vector
    return vector<int>();
}

vector<int> TwoPhaseQuasiCliqueSolver::findLargeQuasiClique(int numSeeds, int numThreads) {
    // Determine number of threads to use
    if (numThreads <= 0) {
        numThreads = thread::hardware_concurrency();
        if (numThreads == 0) numThreads = 1;
    }
    
    // // Register signal handlers for graceful termination
    // signal(SIGINT, signalHandler);  // Ctrl+C
    // signal(SIGTERM, signalHandler); // kill command
    
    // Pre-compute clustering coefficients
    precomputeClusteringCoefficients(numThreads);
    
    if (terminationRequested) {
        cout << "Termination requested during preprocessing. Exiting." << endl;
        return bestSolutionOverall;
    }
    
    // Detect communities
    cout << "Step 1: Detecting communities in the graph..." << endl;
    communityDetector.detectCommunities();
    
    if (terminationRequested) {
        cout << "Termination requested during community detection. Exiting." << endl;
        return bestSolutionOverall;
    }
    
    // Phase 1: Find multiple candidate solutions using community-aware approach
    cout << "Step 2: Starting Phase 1 - Finding candidate solutions..." << endl;
    phase1_findCandidateSolutions(numSeeds, numThreads);
    
    if (terminationRequested) {
        cout << "Termination requested during Phase 1. Returning best solution found so far." << endl;
        return bestSolutionOverall;
    }
    
    // Phase 2: Refine and merge solutions
    cout << "Step 3: Starting Phase 2 - Refining and merging solutions..." << endl;
    phase2_refineSolutions();
    
    if (terminationRequested) {
        cout << "Termination requested during Phase 2. Returning best solution found so far." << endl;
    }
    
    return bestSolutionOverall;
}

void TwoPhaseQuasiCliqueSolver::verifyAndPrintSolution(const vector<int>& solution) {
    int n = solution.size();
    int possibleEdges = (n * (n - 1)) / 2;
    int actualEdges = countEdges(solution);
    double density = (n > 1) ? static_cast<double>(actualEdges) / possibleEdges : 0;
    
    cout << "\n=== Solution Summary ===" << endl;
    cout << "Vertices: " << n << endl;
    cout << "Edges: " << actualEdges << "/" << possibleEdges << endl;
    cout << "Density: " << density << endl;
    cout << "Minimum required edges for quasi-clique: " << (possibleEdges / 2) + 1 << endl;
    
    bool isConnectedSolution = isConnected(solution);
    
    if (n > 0) {
        if (actualEdges > possibleEdges / 2) {
            cout << "✓ Solution is a valid quasi-clique!" << endl;
        } else {
            cout << "✗ Solution is NOT a valid quasi-clique!" << endl;
        }
        
        cout << "✓ Solution is " << (isConnectedSolution ? "connected" : "NOT connected") << endl;
        
        if (n <= 100) {
            cout << "\nVertices in solution: ";
            for (int v : solution) {
                cout << v << " ";
            }
            cout << endl;
        } else {
            cout << "\nSolution has " << n << " vertices (too many to display)" << endl;
        }
    } else {
        cout << "No solution found." << endl;
    }
}

bool TwoPhaseQuasiCliqueSolver::saveSolution(const vector<int>& solution, const string& filename) {
    try {
        ofstream solutionFile(filename);
        if (!solutionFile.is_open()) {
            cerr << "Error: Could not open file " << filename << " for writing" << endl;
            return false;
        }
        
        for (int v : solution) {
            solutionFile << v << endl;
        }
        
        cout << "Solution saved to " << filename << endl;
        return true;
    } catch (const exception& e) {
        cerr << "Error saving solution: " << e.what() << endl;
        return false;
    }
}