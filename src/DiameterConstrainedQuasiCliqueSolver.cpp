#include "DiameterConstrainedQuasiCliqueSolver.h"
#include <iostream>
#include <fstream>
#include <random>
#include <algorithm>
#include <chrono>
#include <queue>
#include <filesystem>
#include <sstream>
#include <iomanip>

using namespace std;
namespace fs = std::filesystem;

DiameterConstrainedQuasiCliqueSolver::DiameterConstrainedQuasiCliqueSolver(const Graph& g, const std::string& outputDir) 
    : graph(g), communityDetector(g), outputDirectory(outputDir) {
    
    // Create output directory if it doesn't exist
    fs::create_directories(outputDirectory);
    
    // Initialize file paths
    maximalCliquesFile = outputDirectory + "maximal_cliques.txt";
    kCoresFile = outputDirectory + "k_cores.txt";
    communitiesFile = outputDirectory + "communities.txt";
    filteredStructuresFile = outputDirectory + "filtered_structures.txt";
}

void DiameterConstrainedQuasiCliqueSolver::precomputeClusteringCoefficients(int numThreads) {
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

void DiameterConstrainedQuasiCliqueSolver::precomputeTwoHopNeighborhoods(const vector<int>& vertices) {
    cout << "Pre-computing two-hop neighborhoods..." << endl;
    
    vector<int> vertsToProcess;
    if (vertices.empty()) {
        // Use all vertices
        vertsToProcess = graph.getVertices();
    } else {
        vertsToProcess = vertices;
    }
    
    auto startTime = chrono::high_resolution_clock::now();
    int processed = 0;
    int total = vertsToProcess.size();
    
    for (int v : vertsToProcess) {
        if (terminationRequested) break;
        
        // Skip if already computed
        if (twoHopNeighborhoods.find(v) != twoHopNeighborhoods.end()) {
            processed++;
            continue;
        }
        
        // Get direct neighbors
        const auto& neighbors = graph.getNeighbors(v);
        unordered_set<int> directNeighborSet(neighbors.begin(), neighbors.end());
        directNeighbors[v] = directNeighborSet;
        
        // Initialize two-hop set with direct neighbors
        unordered_set<int> twoHopSet = directNeighborSet;
        twoHopSet.insert(v); // Include self
        
        // Add neighbors of neighbors
        for (int neighbor : neighbors) {
            if (terminationRequested) break;
            
            const auto& secondHops = graph.getNeighbors(neighbor);
            for (int secondHop : secondHops) {
                twoHopSet.insert(secondHop);
            }
        }
        
        // Store result
        twoHopNeighborhoods[v] = twoHopSet;
        
        // Progress report
        processed++;
        if (processed % 10000 == 0 || processed == total) {
            auto currentTime = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::seconds>(currentTime - startTime).count();
            cout << "  Processed " << processed << "/" << total << " vertices (" 
                 << (duration > 0 ? processed / duration : processed) << " vertices/sec)" << endl;
        }
    }
    
    cout << "Completed two-hop neighborhood computation for " << processed << " vertices" << endl;
}

double DiameterConstrainedQuasiCliqueSolver::calculateVertexPotential(int v, const vector<int>& solution) const {
    // Calculate connections to current solution
    int connections = countConnectionsToSolution(v, solution);
    double connectionRatio = static_cast<double>(connections) / solution.size();
    
    // Get clustering coefficient
    auto it = clusteringCoefficients.find(v);
    double clustering = (it != clusteringCoefficients.end()) ? it->second : 0.0;
    
    // Get degree
    int degree = graph.getDegree(v);
    double normalizedDegree = min(1.0, degree / 50.0); // Normalize degree (adjust denominator as needed)
    
    // Weights for different factors
    double alpha = 0.7;  // Connection ratio weight
    double beta = 0.2;   // Clustering weight
    double gamma = 0.1;  // Degree weight
    
    // Combine factors
    return alpha * connectionRatio + beta * clustering + gamma * normalizedDegree;
}

bool DiameterConstrainedQuasiCliqueSolver::isQuasiClique(const vector<int>& nodes) const {
    int n = nodes.size();
    int possibleEdges = (n * (n - 1)) / 2;
    int minEdgesNeeded = (possibleEdges / 2) + 1; // Strictly more than half
    
    int actualEdges = countEdges(nodes);
    return actualEdges >= minEdgesNeeded;
}

int DiameterConstrainedQuasiCliqueSolver::countEdges(const vector<int>& nodes) const {
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

bool DiameterConstrainedQuasiCliqueSolver::isConnected(const vector<int>& nodes) const {
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

unordered_set<int> DiameterConstrainedQuasiCliqueSolver::findBoundaryVertices(const vector<int>& solution) const {
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

int DiameterConstrainedQuasiCliqueSolver::countConnectionsToSolution(int candidate, const vector<int>& solution) const {
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
int DiameterConstrainedQuasiCliqueSolver::calculateDiameter(const vector<int>& nodes) const {
    if (nodes.size() <= 1) return 0;
    
    int maxDist = 0;
    unordered_set<int> nodeSet(nodes.begin(), nodes.end());
    
    // Optimize for small subgraphs by only checking a subset of nodes as sources
    vector<int> sources;
    if (nodes.size() <= 100) {
        // For small graphs, check all pairs
        sources = nodes;
    } else {
        // For larger graphs, sample nodes to estimate diameter
        // This is a heuristic that works well in practice
        random_device rd;
        mt19937 g(rd());
        
        // Sample at least sqrt(n) nodes, but cap at 100
        int sampleSize = min(100, max(20, (int)sqrt(nodes.size())));
        
        vector<int> shuffled = nodes;
        shuffle(shuffled.begin(), shuffled.end(), g);
        sources.assign(shuffled.begin(), shuffled.begin() + sampleSize);
    }
    
    // For each source, run BFS to find longest shortest path
    for (int source : sources) {
        if (terminationRequested) return maxDist;
        
        unordered_map<int, int> distances;
        queue<int> q;
        
        q.push(source);
        distances[source] = 0;
        
        while (!q.empty()) {
            int current = q.front();
            q.pop();
            
            for (int neighbor : graph.getNeighbors(current)) {
                if (nodeSet.count(neighbor) > 0 && distances.count(neighbor) == 0) {
                    distances[neighbor] = distances[current] + 1;
                    maxDist = max(maxDist, distances[neighbor]);
                    q.push(neighbor);
                }
            }
        }
    }
    
    return maxDist;
}

bool DiameterConstrainedQuasiCliqueSolver::couldMaintainDiameter(const vector<int>& solution, int candidate) const {
    // Quick check using two-hop neighborhoods
    // If all vertices in solution are within two hops of candidate, adding candidate couldn't increase diameter beyond 3
    
    // First check if we have pre-computed two-hop neighborhoods
    auto candidateTwoHop = twoHopNeighborhoods.find(candidate);
    if (candidateTwoHop != twoHopNeighborhoods.end()) {
        // Check if all solution vertices are in candidate's two-hop neighborhood
        for (int v : solution) {
            if (candidateTwoHop->second.count(v) == 0) {
                return false; // At least one vertex would be beyond 2 hops from candidate
            }
        }
        return true; // All vertices are within 2 hops of candidate
    }
    
    // If two-hop neighborhoods not pre-computed, do a direct check
    // This is less efficient but doesn't require pre-computation
    
    // Check if all vertices in solution are within 2 hops of candidate
    unordered_set<int> directNeighborsOfCandidate(graph.getNeighbors(candidate).begin(), 
                                               graph.getNeighbors(candidate).end());
    
    for (int v : solution) {
        if (v == candidate) continue; // Skip self
        
        // If v is a direct neighbor of candidate, it's within 1 hop
        if (directNeighborsOfCandidate.count(v) > 0) continue;
        
        // Check if v shares a neighbor with candidate
        bool hasSharedNeighbor = false;
        for (int u : graph.getNeighbors(v)) {
            if (directNeighborsOfCandidate.count(u) > 0) {
                hasSharedNeighbor = true;
                break;
            }
        }
        
        if (!hasSharedNeighbor) {
            return false; // v is more than 2 hops away from candidate
        }
    }
    
    return true; // All vertices are within 2 hops of candidate
}

vector<vector<int>> DiameterConstrainedQuasiCliqueSolver::findMaximalCliques() {
    cout << "Finding maximal cliques..." << endl;
    
    // Check if we already have computed maximal cliques
    if (fs::exists(maximalCliquesFile)) {
        cout << "Loading pre-computed maximal cliques from " << maximalCliquesFile << endl;
        return loadStructuresFromFile(maximalCliquesFile);
    }
    
    auto startTime = chrono::high_resolution_clock::now(); // Add this line
    
    vector<vector<int>> cliques;
    
    // For very large graphs, we need to limit the search
    if (graph.getNumVertices() > 10000) {
        cout << "  Graph is very large, using heuristic clique finding instead of exact algorithm" << endl;
        
        // Use a heuristic approach - find cliques in high k-cores
        vector<pair<int, vector<int>>> kCores = extractKCores(20);
        
        // Only consider the highest k-cores
        if (!kCores.empty()) {
            int highestK = kCores[0].first;
            for (const auto& [k, core] : kCores) {
                if (k >= highestK - 2 && core.size() <= 1000) {  // Only process manageable cores
                    cout << "  Finding cliques in k-core with k=" << k << ", size=" << core.size() << endl;
                    
                    unordered_set<int> P(core.begin(), core.end());
                    unordered_set<int> R, X;
                    int maxDepth = 3;  // Limit recursion depth
                    
                    bronKerbosch(R, P, X, cliques, 0, maxDepth);
                }
            }
        }
    } else {
        // For smaller graphs, use the full algorithm with some limits
        unordered_set<int> P(graph.getVertices().begin(), graph.getVertices().end());
        unordered_set<int> R, X;
        int maxDepth = 5;  // Limit recursion depth
        
        bronKerbosch(R, P, X, cliques, 0, maxDepth);
    }
    
    auto endTime = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::seconds>(endTime - startTime).count();
    
    cout << "Found " << cliques.size() << " maximal cliques in " << duration << " seconds" << endl;
    
    
    // Filter by minimum size
    vector<vector<int>> filteredCliques;
    for (const auto& clique : cliques) {
        if (clique.size() >= static_cast<size_t>(minCliqueSize)) {
            filteredCliques.push_back(clique);
        }
    }
    
    cout << "Filtered to " << filteredCliques.size() << " cliques of size >= " << minCliqueSize << endl;
    
    // Save to file
    saveStructuresToFile(filteredCliques, maximalCliquesFile);
    
    return filteredCliques;
}

// Also update the bronKerbosch method signature to include maxDepth:
void DiameterConstrainedQuasiCliqueSolver::bronKerbosch(
    unordered_set<int>& R, unordered_set<int>& P, unordered_set<int>& X, 
    vector<vector<int>>& cliques, int depth, int maxDepth) {
    
    if (terminationRequested) return;
    
    // Base case: if P and X are empty, R is a maximal clique
    if (P.empty() && X.empty()) {
        if (R.size() >= static_cast<size_t>(minCliqueSize)) {
            cliques.push_back(vector<int>(R.begin(), R.end()));
            
            // Progress reporting
            if (cliques.size() % 1000 == 0) {
                cout << "  Found " << cliques.size() << " maximal cliques so far" << endl;
            }
        }
        return;
    }
    
    // Limit recursion depth
    if (depth >= maxDepth) {
        return;
    }
    
    // Select pivot to reduce branches
    int pivot = selectPivot(P, X);
    
    // Process vertices in P that are not neighbors of pivot
    unordered_set<int> P_copy = P;
    for (int v : P_copy) {
        if (graph.hasEdge(v, pivot)) continue;
        
        // Move v from P to R
        P.erase(v);
        R.insert(v);
        
        // Create new sets for recursive call
        unordered_set<int> new_P, new_X;
        for (int u : P) {
            if (graph.hasEdge(u, v)) {
                new_P.insert(u);
            }
        }
        for (int u : X) {
            if (graph.hasEdge(u, v)) {
                new_X.insert(u);
            }
        }
        
        // Recursive call with incremented depth
        bronKerbosch(R, new_P, new_X, cliques, depth + 1, maxDepth);
        
        // Move v from R to X
        R.erase(v);
        X.insert(v);
    }
}

int DiameterConstrainedQuasiCliqueSolver::selectPivot(
    const unordered_set<int>& P, const unordered_set<int>& X) const {
    
    // Choose pivot that maximizes neighbors in P
    int bestPivot = -1;
    int maxNeighbors = -1;
    
    // Check potential pivots from both P and X
    for (int pivot : P) {
        int neighbors = 0;
        for (int u : P) {
            if (u != pivot && graph.hasEdge(pivot, u)) {
                neighbors++;
            }
        }
        if (neighbors > maxNeighbors) {
            maxNeighbors = neighbors;
            bestPivot = pivot;
        }
    }
    
    for (int pivot : X) {
        int neighbors = 0;
        for (int u : P) {
            if (graph.hasEdge(pivot, u)) {
                neighbors++;
            }
        }
        if (neighbors > maxNeighbors) {
            maxNeighbors = neighbors;
            bestPivot = pivot;
        }
    }
    
    // If no pivot found, return any vertex from P (or -1 if P is empty)
    if (bestPivot == -1 && !P.empty()) {
        bestPivot = *P.begin();
    }
    
    return bestPivot;
}
vector<pair<int, vector<int>>> DiameterConstrainedQuasiCliqueSolver::extractKCores(int maxK) {
    cout << "Extracting k-cores up to k=" << maxK << "..." << endl;
    
    // Check if we already have computed k-cores
    if (fs::exists(kCoresFile)) {
        cout << "Loading pre-computed k-cores from " << kCoresFile << endl;
        
        // Load into appropriate format
        vector<vector<int>> rawCores = loadStructuresFromFile(kCoresFile);
        vector<pair<int, vector<int>>> kCores;
        
        for (size_t i = 0; i < rawCores.size(); i++) {
            if (rawCores[i].empty()) continue;
            
            // First element is the k value
            int k = rawCores[i][0];
            vector<int> core(rawCores[i].begin() + 1, rawCores[i].end());
            kCores.push_back({k, core});
        }
        
        return kCores;
    }
    
    // Compute k-core decomposition
    vector<int> vertices = graph.getVertices();
    int n = vertices.size();
    
    // Map vertex IDs to indices
    unordered_map<int, int> vertexToIndex;
    for (int i = 0; i < n; i++) {
        vertexToIndex[vertices[i]] = i;
    }
    
    // Initialize with degrees
    vector<int> degrees(n);
    for (int i = 0; i < n; i++) {
        degrees[i] = graph.getDegree(vertices[i]);
    }
    
    // Find maximum degree
    int maxDegree = 0;
    for (int d : degrees) {
        maxDegree = max(maxDegree, d);
    }
    
    // Create bins for each degree
    vector<vector<int>> bins(maxDegree + 1);
    for (int i = 0; i < n; i++) {
        bins[degrees[i]].push_back(i);
    }
    
    // Array to track if a vertex has been processed
    vector<bool> processed(n, false);
    
    // Resulting core numbers
    vector<int> coreNumbers(n, 0);
    
    // Process vertices in order of increasing degree
    int currentK = 0;
    int numProcessed = 0;
    
    cout << "  Running k-core algorithm..." << endl;
    auto startTime = chrono::high_resolution_clock::now();
    
    while (numProcessed < n && !terminationRequested) {
        // Find the next non-empty bin
        while (currentK <= maxDegree && bins[currentK].empty()) {
            currentK++;
        }
        
        if (currentK > maxDegree) break;
        
        // Process all vertices in the current bin
        while (!bins[currentK].empty() && !terminationRequested) {
            int vIdx = bins[currentK].back();
            bins[currentK].pop_back();
            
            if (processed[vIdx]) continue;
            
            processed[vIdx] = true;
            coreNumbers[vIdx] = currentK;
            numProcessed++;
            
            // Progress reporting
            if (numProcessed % 10000 == 0) {
                auto currentTime = chrono::high_resolution_clock::now();
                auto elapsed = chrono::duration_cast<chrono::seconds>(
                    currentTime - startTime).count();
                
                cout << "    Processed " << numProcessed << "/" << n 
                      << " vertices (" << (elapsed > 0 ? numProcessed / elapsed : numProcessed) 
                      << " vertices/sec)" << endl;
            }
            
            // Update neighbors
            for (int neighbor : graph.getNeighbors(vertices[vIdx])) {
                auto it = vertexToIndex.find(neighbor);
                if (it == vertexToIndex.end()) continue;
                
                int nIdx = it->second;
                if (!processed[nIdx] && degrees[nIdx] > currentK) {
                    // Remove from current bin
                    auto binIt = find(bins[degrees[nIdx]].begin(), bins[degrees[nIdx]].end(), nIdx);
                    if (binIt != bins[degrees[nIdx]].end()) {
                        *binIt = bins[degrees[nIdx]].back();
                        bins[degrees[nIdx]].pop_back();
                    }
                    
                    // Decrement degree
                    degrees[nIdx]--;
                    
                    // Add to new bin
                    bins[degrees[nIdx]].push_back(nIdx);
                }
            }
        }
    }
    ///////////////////////
    cout << "  K-core decomposition complete. Maximum coreness: " 
          << *max_element(coreNumbers.begin(), coreNumbers.end()) << endl;
    
    // Create k-core sets for multiple k values (all vertices with coreness >= k)
    vector<pair<int, vector<int>>> kCores;
    unordered_set<int> alreadySavedVertices;
    
    for (int k = 3; k <= min(maxK, *max_element(coreNumbers.begin(), coreNumbers.end())); k++) {
        vector<int> kCore;
        for (int i = 0; i < n; i++) {
            if (coreNumbers[i] >= k) {
                kCore.push_back(vertices[i]);
            }
        }
        
        // Only add if not empty and contains new vertices
        if (!kCore.empty()) {
            // Check if this core has at least some new vertices
            bool hasNewVertices = false;
            for (int v : kCore) {
                if (alreadySavedVertices.count(v) == 0) {
                    hasNewVertices = true;
                    alreadySavedVertices.insert(v);
                }
            }
            
            if (hasNewVertices || k == maxK) {
                kCores.push_back({k, kCore});
                cout << "    Found k-core with k=" << k << ", size=" << kCore.size() << endl;
            }
        }
    }
    
    // Save k-cores to file in a format we can reload
    vector<vector<int>> rawCores;
    for (const auto& [k, core] : kCores) {
        vector<int> rawCore;
        rawCore.push_back(k); // First element is k value
        rawCore.insert(rawCore.end(), core.begin(), core.end());
        rawCores.push_back(rawCore);
    }
    
    saveStructuresToFile(rawCores, kCoresFile);
    
    return kCores;
}

vector<vector<int>> DiameterConstrainedQuasiCliqueSolver::filterContainedStructures(
    const vector<vector<int>>& structures) {
    
    cout << "Filtering " << structures.size() << " structures based on containment..." << endl;
    
    vector<vector<int>> filteredStructures;
    vector<bool> isContained(structures.size(), false);
    
    // Check each pair for containment (O(n²) complexity)
    auto startTime = chrono::high_resolution_clock::now();
    int comparisons = 0;
    
    for (size_t i = 0; i < structures.size(); i++) {
        if (terminationRequested) break;
        if (isContained[i]) continue;
        
        unordered_set<int> structI(structures[i].begin(), structures[i].end());
        
        for (size_t j = i + 1; j < structures.size(); j++) {
            if (isContained[j]) continue;
            
            comparisons++;
            
            // Only check if sizes are compatible for containment
            if (structures[i].size() >= structures[j].size()) {
                // Check if i contains j
                bool iContainsJ = true;
                for (int v : structures[j]) {
                    if (structI.count(v) == 0) {
                        iContainsJ = false;
                        break;
                    }
                }
                
                if (iContainsJ) {
                    // Structure i contains structure j, mark j as contained
                    isContained[j] = true;
                }
            } else {
                // Check if j contains i
                unordered_set<int> structJ(structures[j].begin(), structures[j].end());
                bool jContainsI = true;
                for (int v : structures[i]) {
                    if (structJ.count(v) == 0) {
                        jContainsI = false;
                        break;
                    }
                }
                
                if (jContainsI) {
                    // Structure j contains structure i, mark i as contained
                    isContained[i] = true;
                    break; // No need to check further for i
                }
            }
            
            // Progress reporting
            if (comparisons % 1000000 == 0) {
                auto currentTime = chrono::high_resolution_clock::now();
                auto elapsed = chrono::duration_cast<chrono::seconds>(currentTime - startTime).count();
                cout << "  Processed " << comparisons << " comparisons in " 
                      << elapsed << " seconds (" 
                      << (elapsed > 0 ? comparisons / elapsed : comparisons) << " comparisons/sec)" << endl;
            }
        }
    }
    
    // Collect non-contained structures
    for (size_t i = 0; i < structures.size(); i++) {
        if (!isContained[i]) {
            filteredStructures.push_back(structures[i]);
        }
    }
    
    auto endTime = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::seconds>(endTime - startTime).count();
    
    cout << "Filtering complete in " << duration << " seconds. Removed " 
          << (structures.size() - filteredStructures.size()) << " contained structures." << endl;
    cout << "Remaining structures: " << filteredStructures.size() << endl;
    
    return filteredStructures;
}

vector<vector<int>> DiameterConstrainedQuasiCliqueSolver::filterStructuresByDiameter(
    const vector<vector<int>>& structures) {
    
    cout << "Filtering structures by diameter (<= " << maxDiameter << ")..." << endl;
    vector<vector<int>> validStructures;
    
    // Process in parallel for efficiency
    int total = structures.size();
    atomic<int> processed(0);
    
    // For large inputs, limit the number to check (focusing on better quality ones)
    const int MAX_TO_CHECK = 5000;
    vector<vector<int>> structuresToCheck = structures;
    
    if (structuresToCheck.size() > MAX_TO_CHECK) {
        // Sort by a quality metric first
        vector<pair<vector<int>, double>> scoredStructures;
        for (const auto& structure : structures) {
            double score = calculateDensity(structure) * structure.size();
            scoredStructures.push_back({structure, score});
        }
        
        sort(scoredStructures.begin(), scoredStructures.end(),
            [](const auto& a, const auto& b) {
                return a.second > b.second;
            });
        
        // Keep top structures
        structuresToCheck.clear();
        for (int i = 0; i < MAX_TO_CHECK; i++) {
            structuresToCheck.push_back(scoredStructures[i].first);
        }
        
        cout << "  Limited to checking " << MAX_TO_CHECK << " highest quality structures" << endl;
    }
    
    // Process all structures
    mutex validStructuresMutex;
    vector<future<void>> futures;
    
    for (const auto& structure : structuresToCheck) {
        futures.push_back(async(launch::async, [this, &structure, &validStructuresMutex, &validStructures, &processed, total]() {
            if (terminationRequested) return;
            
            int diameter = calculateDiameter(structure);
            
            if (diameter <= maxDiameter) {
                lock_guard<mutex> lock(validStructuresMutex);
                validStructures.push_back(structure);
            }
            
            int count = ++processed;
            if (count % 100 == 0 || count == total) {
                cout << "  Processed " << count << "/" << total << " structures" << endl;
            }
        }));
    }
    
    // Wait for all tasks to complete
    for (auto& f : futures) {
        f.wait();
    }
    
    cout << "Diameter filtering complete. Found " << validStructures.size() 
         << " structures with diameter <= " << maxDiameter << endl;
    
    return validStructures;
}
vector<int> DiameterConstrainedQuasiCliqueSolver::expandWithDiameterConstraint(vector<int> initialSolution) {
    cout << "Expanding solution with diameter constraint (max diameter = " << maxDiameter << ")..." << endl;
    cout << "Initial solution size: " << initialSolution.size() << endl;
    
    unordered_set<int> solutionSet(initialSolution.begin(), initialSolution.end());
    bool improved = true;
    int iterations = 0;
    
    while (improved && !terminationRequested) {
        improved = false;
        iterations++;
        
        // Find boundary vertices (neighbors of solution vertices)
        unordered_set<int> boundary = findBoundaryVertices(initialSolution);
        
        // Convert to vector and sort by potential value
        vector<pair<int, double>> sortedCandidates;
        for (int candidate : boundary) {
            // First, quick check for diameter constraint
            if (!couldMaintainDiameter(initialSolution, candidate)) {
                continue;
            }
            
            // Calculate potential score
            double score = calculateVertexPotential(candidate, initialSolution);
            sortedCandidates.push_back({candidate, score});
        }
        
        // Sort by score (descending)
        sort(sortedCandidates.begin(), sortedCandidates.end(),
            [](const pair<int, double>& a, const pair<int, double>& b) {
                return a.second > b.second;
            });
        
        // Try adding top candidates
        for (const auto& [candidate, score] : sortedCandidates) {
            // Test solution with candidate added
            vector<int> testSolution = initialSolution;
            testSolution.push_back(candidate);
            
            // Verify it's still a quasi-clique with acceptable diameter
            if (isQuasiClique(testSolution)) {
                // Only calculate exact diameter if necessary (it's expensive)
                int diameter = calculateDiameter(testSolution);
                if (diameter <= maxDiameter) {
                    initialSolution = testSolution;
                    solutionSet.insert(candidate);
                    improved = true;
                    
                    // Progress report
                    if (iterations % 10 == 0) {
                        cout << "  Iteration " << iterations << ": added vertex " << candidate 
                              << ", new size = " << initialSolution.size() << endl;
                    }
                    
                    break;
                }
            }
        }
        
        // Check if we should update best solution
        if (improved && initialSolution.size() > bestSolutionOverall.size()) {
            updateBestSolution(initialSolution);
        }
    }
    
    cout << "Expansion complete after " << iterations << " iterations." << endl;
    cout << "Final solution size: " << initialSolution.size() << endl;
    
    return initialSolution;
}

void DiameterConstrainedQuasiCliqueSolver::processBatchedStructures(
    const vector<vector<int>>& structures, int numThreads) {
    
    cout << "Processing " << structures.size() << " structures in batches using " 
          << numThreads << " threads..." << endl;
    
    // Create thread pool
    ThreadPool pool(numThreads);
    
    // Track the total number of structures to process
    totalStructures = structures.size();
    completedStructures = 0;
    
    // Process structures in batches
    const int BATCH_SIZE = 50;
    
    for (size_t start = 0; start < structures.size(); start += BATCH_SIZE) {
        size_t end = min(start + BATCH_SIZE, structures.size());
        
        // Process batch
        for (size_t i = start; i < end; i++) {
            pool.enqueue([this, i, &structures]() {
                if (terminationRequested) return;
                
                // Expand the structure
                vector<int> expanded = expandWithDiameterConstraint(structures[i]);
                
                // Update best solution if better
                if (expanded.size() > bestSolutionOverall.size()) {
                    updateBestSolution(expanded);
                }
                
                // Update progress counter
                completedStructures++;
                
                // Report progress
                if (completedStructures % 10 == 0 || completedStructures == totalStructures) {
                    cout << "Processed " << completedStructures << "/" << totalStructures << " structures" << endl;
                }
            });
        }
        
        // Wait for the batch to complete before starting the next one
        // This prevents overloading the queue with too many tasks
        while (pool.pendingTasks() > 0 && !terminationRequested) {
            this_thread::sleep_for(chrono::milliseconds(100));
        }
        
        if (terminationRequested) {
            cout << "Termination requested. Stopping structure processing." << endl;
            break;
        }
    }
    
    // Wait for all remaining tasks to complete
    while (pool.pendingTasks() > 0 && !terminationRequested) {
        this_thread::sleep_for(chrono::seconds(1));
        cout << "Waiting for " << pool.pendingTasks() << " remaining tasks to complete..." << endl;
    }
    
    cout << "Structure processing complete. Best solution size: " << bestSolutionOverall.size() << endl;
}

vector<vector<int>> DiameterConstrainedQuasiCliqueSolver::selectTopStructures(
    const vector<vector<int>>& structures, int maxToKeep) {
    
        if (structures.size() <= static_cast<size_t>(maxToKeep)) {
        return structures;
    }
    
    cout << "Selecting top " << maxToKeep << " structures from " << structures.size() << "..." << endl;
    
    // Score structures based on quality metrics
    vector<pair<vector<int>, double>> scoredStructures;
    
    for (const auto& structure : structures) {
        double density = calculateDensity(structure);
        double sizeBonus = log(structure.size()) / log(10); // Logarithmic size bonus
        double score = density * 0.7 + sizeBonus * 0.3;
        
        scoredStructures.push_back({structure, score});
    }
    
    // Sort by score (descending)
    sort(scoredStructures.begin(), scoredStructures.end(),
        [](const auto& a, const auto& b) {
            return a.second > b.second;
        });
    
    // Keep only top structures
    vector<vector<int>> selectedStructures;
    for (int i = 0; i < min(maxToKeep, (int)scoredStructures.size()); i++) {
        selectedStructures.push_back(scoredStructures[i].first);
    }
    
    cout << "Selected " << selectedStructures.size() << " top-quality structures" << endl;
    
    return selectedStructures;
}

double DiameterConstrainedQuasiCliqueSolver::calculateDensity(const vector<int>& structure) const {
    int n = structure.size();
    
    if (n <= 1) return 1.0; // Single vertex or empty set has density 1 by definition
    
    int possibleEdges = (n * (n - 1)) / 2;
    int actualEdges = countEdges(structure);
    
    return static_cast<double>(actualEdges) / possibleEdges;
}
vector<int> DiameterConstrainedQuasiCliqueSolver::optimizeByNodeSwapping(
    const vector<int>& solution, int maxIterations) {
    
    if (solution.empty()) return solution;
    
    cout << "Optimizing solution of size " << solution.size() << " using node swapping..." << endl;
    
    vector<int> bestSolution = solution;
    unordered_set<int> solutionSet(solution.begin(), solution.end());
    
    // Find minimum and maximum degree within the solution
    int minDegree = numeric_limits<int>::max();
    int maxDegree = 0;
    
    for (int node : solution) {
        int degree = graph.getDegree(node);
        minDegree = min(minDegree, degree);
        maxDegree = max(maxDegree, degree);
    }
    
    cout << "  Degree range in solution: " << minDegree << " to " << maxDegree << endl;
    
    // Calculate internal connections for each node in the solution
    vector<pair<int, int>> internalConnections;
    for (int node : solution) {
        int connections = countConnectionsToSolution(node, solution) - 1; // -1 to exclude self-connection count
        internalConnections.push_back({node, connections});
    }
    
    // Sort by internal connections (ascending)
    sort(internalConnections.begin(), internalConnections.end(),
         [](const pair<int, int>& a, const pair<int, int>& b) {
             return a.second < b.second;
         });
    
    cout << "  Lowest internal connections: " << internalConnections[0].second 
          << ", highest: " << internalConnections.back().second << endl;
    
    // Get all nodes in the graph with degree in the specified range
    vector<int> allNodes = graph.getVertices();
    vector<int> candidateNodes;
    
    for (int node : allNodes) {
        if (solutionSet.count(node) == 0) { // Node not in solution
            int degree = graph.getDegree(node);
            if (degree >= minDegree && degree <= maxDegree) {
                candidateNodes.push_back(node);
            }
        }
    }
    
    cout << "  Found " << candidateNodes.size() << " candidate nodes with degree in range" << endl;
    
    // Sort candidates by their connections to the solution (descending)
    sort(candidateNodes.begin(), candidateNodes.end(),
         [this, &solution](int a, int b) {
             return countConnectionsToSolution(a, solution) > countConnectionsToSolution(b, solution);
         });
    
    // Only keep top candidates to reduce computation
    const int MAX_CANDIDATES = 2500;
    if (candidateNodes.size() > MAX_CANDIDATES) {
        candidateNodes.resize(MAX_CANDIDATES);
    }
    
    // Try swapping nodes
    int iterations = 0;
    int improvements = 0;
    bool madeAnySwaps = false;
    
    while (iterations < maxIterations && !terminationRequested) {
        bool improved = false;
        iterations++;
        
        // Take a copy of the current solution to work with
        vector<int> currentSolution = bestSolution;
        unordered_set<int> currentSet(currentSolution.begin(), currentSolution.end());
        
        // Try swapping each candidate with nodes that have fewer internal connections
        for (int candidate : candidateNodes) {
            if (currentSet.count(candidate) > 0) continue; // Skip if already in solution
            
            int candidateConnections = countConnectionsToSolution(candidate, currentSolution);
            
            // Only consider candidates with good connectivity
            if (candidateConnections < internalConnections[0].second * 0.9) continue;
            
            // Try swapping with each low-connectivity node
            for (size_t i = 0; i < min(size_t(25), internalConnections.size()); i++) {
                int weakNode = internalConnections[i].first;
                int weakNodeConnections = internalConnections[i].second;
                
                // Only swap if candidate has more connections
                if (candidateConnections <= weakNodeConnections) continue;
                
                // Create a test solution with the swap
                vector<int> testSolution;
                for (int node : currentSolution) {
                    if (node != weakNode) {
                        testSolution.push_back(node);
                    }
                }
                testSolution.push_back(candidate);
                
                // Check if valid quasi-clique with acceptable diameter
                if (isQuasiClique(testSolution) && isConnected(testSolution)) {
                    // Calculate diameter (expensive, so only do if necessary)
                    int diameter = calculateDiameter(testSolution);
                    if (diameter <= maxDiameter) {
                        // Accept the swap
                        currentSolution = testSolution;
                        currentSet.erase(weakNode);
                        currentSet.insert(candidate);
                        
                        // Update internal connections for the modified solution
                        internalConnections.clear();
                        for (int node : currentSolution) {
                            int connections = countConnectionsToSolution(node, currentSolution) - 1;
                            internalConnections.push_back({node, connections});
                        }
                        
                        // Sort by internal connections (ascending)
                        sort(internalConnections.begin(), internalConnections.end(),
                             [](const pair<int, int>& a, const pair<int, int>& b) {
                                 return a.second < b.second;
                             });
                        
                        improved = true;
                        improvements++;
                        madeAnySwaps = true;
                        
                        cout << "  Iteration " << iterations << ": Swapped node " << weakNode 
                             << " (connections: " << weakNodeConnections << ") with node " << candidate
                             << " (connections: " << candidateConnections << ")" << endl;
                        
                        break; // Move to next candidate
                    }
                }
            }
            
            if (improved) break; // Try next iteration with the improved solution
        }
        
        // If we improved, update best solution
        if (improved) {
            bestSolution = currentSolution;
            
            // Update candidate nodes list based on the new solution
            solutionSet.clear();
            solutionSet.insert(bestSolution.begin(), bestSolution.end());
            
            // Re-filter candidates
            candidateNodes.clear();
            for (int node : allNodes) {
                if (solutionSet.count(node) == 0) { // Node not in solution
                    int degree = graph.getDegree(node);
                    if (degree >= minDegree && degree <= maxDegree) {
                        candidateNodes.push_back(node);
                    }
                }
            }
            
            // Sort candidates by their connections to the solution (descending)
            sort(candidateNodes.begin(), candidateNodes.end(),
                 [this, &bestSolution](int a, int b) {
                     return countConnectionsToSolution(a, bestSolution) > 
                            countConnectionsToSolution(b, bestSolution);
                 });
            
            // Only keep top candidates
            if (candidateNodes.size() > MAX_CANDIDATES) {
                candidateNodes.resize(MAX_CANDIDATES);
            }
            
            // Update best overall solution if better
            if (bestSolution.size() > bestSolutionOverall.size()) {
                updateBestSolution(bestSolution);
            }
        } else {
            // No improvement in this iteration
            cout << "  Iteration " << iterations << ": No improvement found" << endl;
            
            // If we've had some improvements, try a few more iterations before giving up
            if (improvements > 0 && iterations >= 10) {
                break;
            }
        }
    }
    
    cout << "Node swapping optimization completed after " << iterations << " iterations" << endl;
    cout << "Made " << improvements << " successful swaps" << endl;
    cout << "Final solution size: " << bestSolution.size() << endl;
    
    return bestSolution;
}
vector<vector<int>> DiameterConstrainedQuasiCliqueSolver::findConnectedComponents(
    const vector<int>& nodes) const {
    
    if (nodes.empty()) return {};
    
    // Create a lookup for fast membership testing
    unordered_set<int> nodeSet(nodes.begin(), nodes.end());
    
    // Create adjacency list for the subgraph
    unordered_map<int, vector<int>> subgraphAdj;
    for (int u : nodes) {
        subgraphAdj[u] = vector<int>();
        for (int v : graph.getNeighbors(u)) {
            if (u != v && nodeSet.count(v) > 0) {
                subgraphAdj[u].push_back(v);
            }
        }
    }
    
    // Keep track of visited nodes
    unordered_set<int> visited;
    
    // Store all connected components
    vector<vector<int>> components;
    
    // Perform BFS for each unvisited node
    for (int startNode : nodes) {
        if (visited.count(startNode) > 0) continue;
        
        // This will store the current component
        vector<int> component;
        
        // BFS
        queue<int> q;
        q.push(startNode);
        visited.insert(startNode);
        
        while (!q.empty()) {
            int current = q.front();
            q.pop();
            
            component.push_back(current);
            
            for (int neighbor : subgraphAdj[current]) {
                if (visited.count(neighbor) == 0) {
                    visited.insert(neighbor);
                    q.push(neighbor);
                }
            }
        }
        
        components.push_back(component);
    }
    
    return components;
}

vector<int> DiameterConstrainedQuasiCliqueSolver::repairSolution(const vector<int>& solution) {
    // First, check what's wrong
    bool isQuasi = isQuasiClique(solution);
    bool isConn = isConnected(solution);
    
    cout << "Repairing solution: isQuasiClique=" << isQuasi << ", isConnected=" << isConn << endl;
    
    if (isQuasi && isConn) {
        // Check diameter
        int diameter = calculateDiameter(solution);
        cout << "  Solution diameter: " << diameter << endl;
        
        if (diameter <= maxDiameter) {
            return solution; // Nothing to repair
        }
    }
    
    vector<int> result = solution;
    
    // First, handle connectivity issues
    if (!isConn) {
        // Find the largest connected component
        vector<vector<int>> components = findConnectedComponents(result);
        if (!components.empty()) {
            // Sort by size (descending)
            sort(components.begin(), components.end(), 
                [](const vector<int>& a, const vector<int>& b) {
                    return a.size() > b.size();
                });
            
            cout << "  Found " << components.size() << " components, largest has " 
                  << components[0].size() << " nodes" << endl;
            
            result = components[0]; // Use largest component
        }
    }
    
    // Now handle quasi-clique property if needed
    if (!isQuasiClique(result)) {
        // Calculate connectivity of each vertex
        vector<pair<int, double>> vertexConnectivity;
        for (int v : result) {
            int connections = 0;
            for (int u : result) {
                if (u != v && graph.hasEdge(u, v)) {
                    connections++;
                }
            }
            double connectivity = (double)connections / (result.size() - 1);
            vertexConnectivity.push_back({v, connectivity});
        }
        
        // Sort by connectivity (lowest first)
        sort(vertexConnectivity.begin(), vertexConnectivity.end(),
            [](const pair<int, double>& a, const pair<int, double>& b) {
                return a.second < b.second;
            });
        
        cout << "  Removing low-connectivity nodes to reach quasi-clique property..." << endl;
        
        // Iteratively remove least connected vertices until we have a valid quasi-clique
        vector<int> workingSolution = result;
        int removed = 0;
        
        for (size_t i = 0; i < vertexConnectivity.size(); i++) {
            int vertexToRemove = vertexConnectivity[i].first;
            
            // Skip if we'd remove too many vertices
            if (removed > vertexConnectivity.size() * 0.4) {
                cout << "  Giving up after removing 40% of vertices" << endl;
                break;
            }
            
            // Remove the vertex
            workingSolution.erase(remove(workingSolution.begin(), workingSolution.end(), vertexToRemove), 
                           workingSolution.end());
            removed++;
            
            // Check if we have a valid solution now
            if (isQuasiClique(workingSolution) && isConnected(workingSolution)) {
                result = workingSolution;
                cout << "  Found valid solution after removing " << removed << " vertices" << endl;
                break;
            }
        }
    }
    
    // Finally, fix diameter issues if needed
    int diameter = calculateDiameter(result);
    if (diameter > maxDiameter) {
        cout << "  Solution diameter is " << diameter << ", reducing..." << endl;
        
        // We'll use a greedy approach to reduce diameter
        // Remove vertices that are furthest from the "center" of the graph
        
        // First, find the center (vertex with minimum eccentricity)
        int center = -1;
        int minEccentricity = numeric_limits<int>::max();
        
        for (int v : result) {
            unordered_map<int, int> distances;
            queue<int> q;
            
            q.push(v);
            distances[v] = 0;
            
            while (!q.empty()) {
                int current = q.front();
                q.pop();
                
                for (int neighbor : graph.getNeighbors(current)) {
                    if (find(result.begin(), result.end(), neighbor) != result.end() && 
                        distances.count(neighbor) == 0) {
                        distances[neighbor] = distances[current] + 1;
                        q.push(neighbor);
                    }
                }
            }
            
            // Find maximum distance (eccentricity)
            int eccentricity = 0;
            for (int u : result) {
                if (u != v) {
                    if (distances.count(u) == 0) {
                        // Disconnected, treat as infinite
                        eccentricity = numeric_limits<int>::max();
                        break;
                    }
                    eccentricity = max(eccentricity, distances[u]);
                }
            }
            
            if (eccentricity < minEccentricity) {
                minEccentricity = eccentricity;
                center = v;
            }
        }
        
        if (center != -1) {
            cout << "  Found center vertex " << center << " with eccentricity " << minEccentricity << endl;
            
            // Calculate distances from center
            unordered_map<int, int> distancesFromCenter;
            queue<int> q;
            
            q.push(center);
            distancesFromCenter[center] = 0;
            
            while (!q.empty()) {
                int current = q.front();
                q.pop();
                
                for (int neighbor : graph.getNeighbors(current)) {
                    if (find(result.begin(), result.end(), neighbor) != result.end() && 
                        distancesFromCenter.count(neighbor) == 0) {
                        distancesFromCenter[neighbor] = distancesFromCenter[current] + 1;
                        q.push(neighbor);
                    }
                }
            }
            
            // Sort vertices by distance from center (descending)
            vector<pair<int, int>> verticesByDistance;
            for (int v : result) {
                int distance = (distancesFromCenter.count(v) > 0) ? distancesFromCenter[v] : numeric_limits<int>::max();
                verticesByDistance.push_back({v, distance});
            }
            
            sort(verticesByDistance.begin(), verticesByDistance.end(),
                [](const pair<int, int>& a, const pair<int, int>& b) {
                    return a.second > b.second;
                });
            
            // Iteratively remove furthest vertices until diameter is acceptable
            vector<int> diameterReducedSolution = result;
            int removed = 0;
            
            for (const auto& [vertex, distance] : verticesByDistance) {
                // Skip center vertex
                if (vertex == center) continue;
                
                // Skip if we'd remove too many vertices
                if (removed > result.size() * 0.3) {
                    cout << "  Giving up after removing 30% of vertices" << endl;
                    break;
                }
                
                // Remove the vertex
                diameterReducedSolution.erase(
                    remove(diameterReducedSolution.begin(), diameterReducedSolution.end(), vertex),
                    diameterReducedSolution.end());
                removed++;
                
                // Check diameter
                int newDiameter = calculateDiameter(diameterReducedSolution);
                
                if (newDiameter <= maxDiameter && isQuasiClique(diameterReducedSolution) && 
                    isConnected(diameterReducedSolution)) {
                    result = diameterReducedSolution;
                    cout << "  Reduced diameter to " << newDiameter << " after removing " 
                         << removed << " vertices" << endl;
                    break;
                }
            }
        }
    }
    
    cout << "Repair complete: original size=" << solution.size() 
          << ", repaired size=" << result.size() << endl;
    
    return result;
}
bool DiameterConstrainedQuasiCliqueSolver::saveStructuresToFile(
    const vector<vector<int>>& structures, const string& filename) {
    
    cout << "Saving " << structures.size() << " structures to " << filename << endl;
    
    try {
        // Create directory if it doesn't exist
        fs::path filepath(filename);
        fs::create_directories(filepath.parent_path());
        
        ofstream file(filename);
        if (!file.is_open()) {
            cerr << "Error: Could not open file " << filename << " for writing" << endl;
            return false;
        }
        
        for (const auto& structure : structures) {
            for (size_t i = 0; i < structure.size(); i++) {
                file << structure[i];
                if (i < structure.size() - 1) {
                    file << " ";
                }
            }
            file << endl;
        }
        
        cout << "Successfully saved structures to " << filename << endl;
        return true;
    } catch (const exception& e) {
        cerr << "Error saving structures: " << e.what() << endl;
        return false;
    }
}

vector<vector<int>> DiameterConstrainedQuasiCliqueSolver::loadStructuresFromFile(const string& filename) {
    vector<vector<int>> structures;
    
    try {
        ifstream file(filename);
        if (!file.is_open()) {
            cerr << "Error: Could not open file " << filename << " for reading" << endl;
            return structures;
        }
        
        string line;
        while (getline(file, line)) {
            istringstream iss(line);
            vector<int> structure;
            int value;
            
            while (iss >> value) {
                structure.push_back(value);
            }
            
            if (!structure.empty()) {
                structures.push_back(structure);
            }
        }
        
        cout << "Loaded " << structures.size() << " structures from " << filename << endl;
    } catch (const exception& e) {
        cerr << "Error loading structures: " << e.what() << endl;
    }
    
    return structures;
}

void DiameterConstrainedQuasiCliqueSolver::updateBestSolution(const vector<int>& solution) {
    lock_guard<mutex> lock(bestSolutionMutex);
    
    if (solution.size() > bestSolutionOverall.size() && 
        isQuasiClique(solution) && isConnected(solution)) {
        
        // Verify diameter
        int diameter = calculateDiameter(solution);
        if (diameter <= maxDiameter) {
            bestSolutionOverall = solution;
            
            // Save progress
            lock_guard<mutex> fileLock(fileWriteMutex);
            ofstream solutionFile("solution_in_progress.txt");
            for (int v : bestSolutionOverall) {
                solutionFile << v << endl;
            }
            solutionFile.close();
            
            cout << "New best solution found: " << bestSolutionOverall.size() 
                  << " vertices (diameter: " << diameter << ")" << endl;
        }
    }
}
/////////////////////////////////
vector<int> DiameterConstrainedQuasiCliqueSolver::findLargeQuasiClique(int numThreads) {
    // Determine number of threads to use
    if (numThreads <= 0) {
        numThreads = thread::hardware_concurrency();
        if (numThreads == 0) numThreads = 1;
    }
    
    cout << "Starting diameter-constrained quasi-clique finder with " << numThreads << " threads" << endl;
    cout << "Maximum allowed diameter: " << maxDiameter << endl;
    
    // Create output directory
    fs::create_directories(outputDirectory);
    
    // Step 1: Precomputation phase
    auto startTime = chrono::high_resolution_clock::now();
    
    // Precompute clustering coefficients
    precomputeClusteringCoefficients(numThreads);
    
    if (terminationRequested) {
        cout << "Termination requested during preprocessing. Exiting." << endl;
        return bestSolutionOverall;
    }
    
    // Step 2: Community detection
    cout << "Step 1: Detecting communities in the graph..." << endl;
    communityDetector.detectCommunities();
    vector<vector<int>> communities = communityDetector.getAllCommunities();
    
    // Save communities to file
    saveStructuresToFile(communities, communitiesFile);
    
    if (terminationRequested) {
        cout << "Termination requested during community detection. Exiting." << endl;
        return bestSolutionOverall;
    }
    
    // Step 3: Find maximal cliques
    cout << "Step 2: Finding maximal cliques..." << endl;
    vector<vector<int>> maximalCliques = findMaximalCliques();
    
    if (terminationRequested) {
        cout << "Termination requested during maximal clique finding. Exiting." << endl;
        return bestSolutionOverall;
    }
    
    // Step 4: Extract k-cores
    cout << "Step 3: Extracting k-cores..." << endl;
    vector<pair<int, vector<int>>> kCores = extractKCores(20); // Up to k=20
    
    // Convert to uniform format
    vector<vector<int>> kCoreStructures;
    for (const auto& [k, core] : kCores) {
        kCoreStructures.push_back(core);
    }
    
    if (terminationRequested) {
        cout << "Termination requested during k-core extraction. Exiting." << endl;
        return bestSolutionOverall;
    }
    
    // Step 5: Combine all structures
    cout << "Step 4: Combining all identified structures..." << endl;
    vector<vector<int>> allStructures;
    allStructures.insert(allStructures.end(), maximalCliques.begin(), maximalCliques.end());
    allStructures.insert(allStructures.end(), kCoreStructures.begin(), kCoreStructures.end());
    allStructures.insert(allStructures.end(), communities.begin(), communities.end());
    
    cout << "Total structures identified: " << allStructures.size() << endl;
    cout << "  - Maximal cliques: " << maximalCliques.size() << endl;
    cout << "  - K-cores: " << kCores.size() << endl;
    cout << "  - Communities: " << communities.size() << endl;
    
    // Step 6: Filter by containment
    cout << "Step 5: Filtering structures by containment..." << endl;
    vector<vector<int>> nonContainedStructures = filterContainedStructures(allStructures);
    
    if (terminationRequested) {
        cout << "Termination requested during containment filtering. Exiting." << endl;
        return bestSolutionOverall;
    }
    
    // Step 7: Filter by diameter
    cout << "Step 6: Filtering structures by diameter constraint (<= " << maxDiameter << ")..." << endl;
    vector<vector<int>> diameterConstrainedStructures = filterStructuresByDiameter(nonContainedStructures);
    
    // Save filtered structures to file
    saveStructuresToFile(diameterConstrainedStructures, filteredStructuresFile);
    
    if (terminationRequested) {
        cout << "Termination requested during diameter filtering. Exiting." << endl;
        return bestSolutionOverall;
    }
    
    // Step 8: Select top structures to process (to limit computational resources)
    const int MAX_STRUCTURES_TO_PROCESS = 1000;
    vector<vector<int>> structuresToProcess = 
        selectTopStructures(diameterConstrainedStructures, MAX_STRUCTURES_TO_PROCESS);
    
    // Step 9: Process filtered structures (expansion phase)
    cout << "Step 7: Expanding filtered structures with diameter constraint..." << endl;
    processBatchedStructures(structuresToProcess, numThreads);
    
    if (terminationRequested) {
        cout << "Termination requested during expansion phase. Returning best solution found so far." << endl;
        return bestSolutionOverall;
    }
    
    // Step 10: Apply node swapping optimization to best solution
    if (useNodeSwapping && !bestSolutionOverall.empty()) {
        cout << "Step 8: Applying node swapping optimization to best solution..." << endl;
        vector<int> optimizedSolution = optimizeByNodeSwapping(bestSolutionOverall);
        
        if (optimizedSolution.size() > bestSolutionOverall.size()) {
            cout << "Node swapping improved solution from " << bestSolutionOverall.size() 
                 << " to " << optimizedSolution.size() << " vertices" << endl;
            bestSolutionOverall = optimizedSolution;
            
            // Save the improved solution
            saveSolution(bestSolutionOverall);
        } else {
            cout << "Node swapping did not improve the solution" << endl;
        }
    }
    
    auto endTime = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::seconds>(endTime - startTime).count();
    
    cout << "Algorithm completed in " << duration << " seconds" << endl;
    cout << "Best solution found: " << bestSolutionOverall.size() << " vertices" << endl;
    
    return bestSolutionOverall;
}
vector<int> DiameterConstrainedQuasiCliqueSolver::expandFromExistingSolution(
    const vector<int>& initialSolution, int numThreads) {
    
    if (initialSolution.empty()) {
        cout << "Empty initial solution provided. Running full algorithm instead." << endl;
        return findLargeQuasiClique(numThreads);
    }
    
    // Verify the initial solution
    bool isValid = isQuasiClique(initialSolution) && isConnected(initialSolution);
    int diameter = calculateDiameter(initialSolution);
    
    cout << "Expanding from existing solution of size " << initialSolution.size() << endl;
    cout << "Initial solution is " << (isValid ? "valid" : "invalid") 
          << " quasi-clique with diameter " << diameter << endl;
    
    // If invalid or diameter too large, try to repair
    vector<int> workingSolution = initialSolution;
    
    if (!isValid || diameter > maxDiameter) {
        cout << "Repairing initial solution to meet constraints..." << endl;
        workingSolution = repairSolution(initialSolution);
        
        if (workingSolution.size() < initialSolution.size() * 0.7) {
            cout << "Warning: Repair resulted in significant loss of vertices. "
                  << "Consider using a different initial solution." << endl;
        }
    }
    
    // Set as initial best solution
    if (!workingSolution.empty() && isQuasiClique(workingSolution) && 
        isConnected(workingSolution) && calculateDiameter(workingSolution) <= maxDiameter) {
        bestSolutionOverall = workingSolution;
    }
    
    // Precompute clustering coefficients
    precomputeClusteringCoefficients(numThreads);
    
    if (terminationRequested) {
        cout << "Termination requested during preprocessing. Exiting." << endl;
        return bestSolutionOverall;
    }
    
    // Precompute two-hop neighborhoods for vertices in and around the solution
    // This helps with diameter checking during expansion
    vector<int> verticesToPrecompute = workingSolution;
    
    // Add boundary vertices
    unordered_set<int> boundary = findBoundaryVertices(workingSolution);
    verticesToPrecompute.insert(verticesToPrecompute.end(), boundary.begin(), boundary.end());
    
    cout << "Precomputing two-hop neighborhoods for " << verticesToPrecompute.size() << " vertices..." << endl;
    precomputeTwoHopNeighborhoods(verticesToPrecompute);
    
    // Expand the solution
    cout << "Expanding solution with diameter constraint..." << endl;
    vector<int> expandedSolution = expandWithDiameterConstraint(workingSolution);
    
    // Apply node swapping optimization
    if (useNodeSwapping) {
        cout << "Applying node swapping optimization..." << endl;
        vector<int> optimizedSolution = optimizeByNodeSwapping(expandedSolution);
        
        if (optimizedSolution.size() > expandedSolution.size()) {
            cout << "Node swapping improved solution from " << expandedSolution.size() 
                 << " to " << optimizedSolution.size() << " vertices" << endl;
            expandedSolution = optimizedSolution;
        } else {
            cout << "Node swapping did not improve the solution" << endl;
        }
    }
    
    // Update best solution if better
    if (expandedSolution.size() > bestSolutionOverall.size()) {
        updateBestSolution(expandedSolution);
    }
    
    return bestSolutionOverall;
}
void DiameterConstrainedQuasiCliqueSolver::verifyAndPrintSolution(const vector<int>& solution) {
    int n = solution.size();
    int possibleEdges = (n * (n - 1)) / 2;
    int actualEdges = countEdges(solution);
    double density = (n > 1) ? static_cast<double>(actualEdges) / possibleEdges : 0;
    int diameter = (n > 1) ? calculateDiameter(solution) : 0;
    
    cout << "\n=== Solution Summary ===" << endl;
    cout << "Vertices: " << n << endl;
    cout << "Edges: " << actualEdges << "/" << possibleEdges << endl;
    cout << "Density: " << density << endl;
    cout << "Diameter: " << diameter << endl;
    cout << "Minimum required edges for quasi-clique: " << (possibleEdges / 2) + 1 << endl;
    
    bool isValidQuasiClique = (actualEdges > possibleEdges / 2);
    bool isConnectedSolution = isConnected(solution);
    bool hasDiameterConstraint = (diameter <= maxDiameter);
    
    if (n > 0) {
        if (isValidQuasiClique) {
            cout << "✓ Solution is a valid quasi-clique!" << endl;
        } else {
            cout << "✗ Solution is NOT a valid quasi-clique! (Missing " 
                 << ((possibleEdges / 2) + 1) - actualEdges << " edges)" << endl;
        }
        
        cout << (isConnectedSolution ? "✓ Solution is connected" : "✗ Solution is NOT connected") << endl;
        
        cout << (hasDiameterConstraint ? 
            "✓ Solution satisfies diameter constraint (≤ " + to_string(maxDiameter) + ")" : 
            "✗ Solution violates diameter constraint (> " + to_string(maxDiameter) + ")") << endl;
        
        // Provide more detailed diagnostics if invalid
        if (!isValidQuasiClique || !isConnectedSolution || !hasDiameterConstraint) {
            cout << "\n=== Detailed Diagnostics ===" << endl;
            
            if (!isValidQuasiClique) {
                // Calculate and report lowest connected vertices
                vector<pair<int, double>> vertexConnectivity;
                for (int v : solution) {
                    int connections = 0;
                    for (int u : solution) {
                        if (u != v && graph.hasEdge(u, v)) {
                            connections++;
                        }
                    }
                    double connectivity = (double)connections / (solution.size() - 1);
                    vertexConnectivity.push_back({v, connectivity});
                }
                
                // Sort by connectivity (lowest first)
                sort(vertexConnectivity.begin(), vertexConnectivity.end(),
                    [](const pair<int, double>& a, const pair<int, double>& b) {
                        return a.second < b.second;
                    });
                
                cout << "  5 least connected vertices:" << endl;
                for (int i = 0; i < min(5, (int)vertexConnectivity.size()); i++) {
                    cout << "    Vertex " << vertexConnectivity[i].first 
                         << ": " << (vertexConnectivity[i].second * 100) << "% connected" << endl;
                }
            }
            
            if (!isConnectedSolution) {
                // Report connected components
                vector<vector<int>> components = findConnectedComponents(solution);
                cout << "  Found " << components.size() << " disconnected components" << endl;
                
                // Sort by size (descending)
                sort(components.begin(), components.end(), 
                    [](const vector<int>& a, const vector<int>& b) {
                        return a.size() > b.size();
                    });
                
                for (size_t i = 0; i < min(size_t(3), components.size()); i++) {
                    cout << "    Component " << (i+1) << ": " << components[i].size() << " vertices" << endl;
                }
            }
            
            if (!hasDiameterConstraint) {
                cout << "  Diameter " << diameter << " exceeds maximum allowed " << maxDiameter << endl;
                
                // TODO: Could add more diameter diagnostics here if needed
            }
        }
        
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

bool DiameterConstrainedQuasiCliqueSolver::saveSolution(
    const vector<int>& solution, const string& filename) {
    
    try {
        // Create directory if it doesn't exist
        fs::path filepath(filename);
        fs::create_directories(filepath.parent_path());
        
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