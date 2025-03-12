// Optimized implementation of the Density Gradient with Clustering Coefficient Algorithm
// This version includes multi-threading and memory optimizations for large graphs
// Fixed version with improved synchronization for solution saving

#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <queue>
#include <condition_variable>
#include <functional>
#include <cmath>
#include <csignal>

// Global flag for signal handling
volatile sig_atomic_t terminationRequested = 0;

// Signal handler for graceful termination
void signalHandler(int signal) {
    std::cout << "Received termination signal " << signal << ". Finishing up..." << std::endl;
    terminationRequested = 1;
}

using namespace std;

// Optimized graph representation using compressed adjacency lists
class OptimizedGraph {
private:
    // Vector of adjacency lists
    vector<vector<int>> adjacencyLists;
    int numEdges = 0;

    // Optional lookup table for non-contiguous vertex IDs
    unordered_map<int, int> vertexToIndex;
    vector<int> indexToVertex;
    bool usingLookupTable = false;

public:
    // Constructor
    OptimizedGraph() {}
    
    // Initialize the graph with a known number of vertices
    void initialize(int numVertices) {
        adjacencyLists.resize(numVertices);
    }

    // Initialize with vertex lookup (for non-contiguous IDs)
    void initializeWithLookup(const vector<int>& vertices) {
        usingLookupTable = true;
        indexToVertex.reserve(vertices.size());
        
        for (size_t i = 0; i < vertices.size(); i++) {
            vertexToIndex[vertices[i]] = i;
            indexToVertex.push_back(vertices[i]);
        }
        
        adjacencyLists.resize(vertices.size());
    }

    // Map external vertex ID to internal index
    int mapVertex(int vertex) const {
        if (!usingLookupTable) return vertex;
        auto it = vertexToIndex.find(vertex);
        return (it != vertexToIndex.end()) ? it->second : -1;
    }

    // Map internal index to external vertex ID
    int mapIndex(int index) const {
        if (!usingLookupTable) return index;
        return (index >= 0 && index < (int)indexToVertex.size()) ? indexToVertex[index] : -1;
    }

    // Add edge to the graph
    void addEdge(int u, int v) {
        if (u == v) return; // No self-loops
        
        int uIdx = mapVertex(u);
        int vIdx = mapVertex(v);
        
        if (uIdx < 0 || vIdx < 0 || uIdx >= (int)adjacencyLists.size() || vIdx >= (int)adjacencyLists.size()) {
            return;
        }
        
        // Check if edge already exists
        if (find(adjacencyLists[uIdx].begin(), adjacencyLists[uIdx].end(), vIdx) == adjacencyLists[uIdx].end()) {
            adjacencyLists[uIdx].push_back(vIdx);
            adjacencyLists[vIdx].push_back(uIdx);
            numEdges++;
        }
    }

    // Check if edge exists
    bool hasEdge(int u, int v) const {
        int uIdx = mapVertex(u);
        int vIdx = mapVertex(v);
        
        if (uIdx < 0 || vIdx < 0 || uIdx >= (int)adjacencyLists.size() || vIdx >= (int)adjacencyLists.size()) {
            return false;
        }
        
        const auto& neighbors = adjacencyLists[uIdx];
        return find(neighbors.begin(), neighbors.end(), vIdx) != neighbors.end();
    }

    // Get neighbors of a vertex
    const vector<int>& getNeighbors(int v) const {
        static const vector<int> emptyVector;
        int vIdx = mapVertex(v);
        
        if (vIdx < 0 || vIdx >= (int)adjacencyLists.size()) {
            return emptyVector;
        }
        
        return adjacencyLists[vIdx];
    }

    // Get degree of a vertex
    int getDegree(int v) const {
        int vIdx = mapVertex(v);
        
        if (vIdx < 0 || vIdx >= (int)adjacencyLists.size()) {
            return 0;
        }
        
        return adjacencyLists[vIdx].size();
    }

    // Get number of vertices
    int getNumVertices() const {
        return adjacencyLists.size();
    }

    // Get number of edges
    int getNumEdges() const {
        return numEdges;
    }

    // Get all vertices
    vector<int> getVertices() const {
        if (usingLookupTable) {
            return indexToVertex;
        } else {
            vector<int> vertices(adjacencyLists.size());
            for (size_t i = 0; i < adjacencyLists.size(); i++) {
                vertices[i] = i;
            }
            return vertices;
        }
    }

    // Calculate clustering coefficient efficiently
    double getClusteringCoefficient(int v) const {
        int vIdx = mapVertex(v);
        
        if (vIdx < 0 || vIdx >= (int)adjacencyLists.size()) {
            return 0.0;
        }
        
        const auto& neighbors = adjacencyLists[vIdx];
        int k = neighbors.size();
        
        if (k < 2) return 0.0;
        
        // Create a set for quick lookup
        unordered_set<int> neighborSet(neighbors.begin(), neighbors.end());
        
        // Count triangles
        int triangles = 0;
        for (size_t i = 0; i < neighbors.size(); i++) {
            for (size_t j = i + 1; j < neighbors.size(); j++) {
                int u = neighbors[i];
                int w = neighbors[j];
                
                // Check if edge exists between neighbors
                if (find(adjacencyLists[u].begin(), adjacencyLists[u].end(), w) != adjacencyLists[u].end()) {
                    triangles++;
                }
            }
        }
        
        double possibleConnections = (k * (k - 1)) / 2.0;
        return triangles / possibleConnections;
    }

    // Load graph from edge list file
    bool loadFromFile(const string& filename) {
        ifstream file(filename);
        if (!file.is_open()) {
            cerr << "Error: Could not open file " << filename << endl;
            return false;
        }

        // First pass: identify all unique vertices
        unordered_set<int> uniqueVertices;
        string line;
        
        cout << "Scanning for vertices..." << endl;
        while (getline(file, line)) {
            if (line.empty() || line[0] == '#') continue; // Skip empty lines and comments
            
            istringstream iss(line);
            int u, v;
            if (!(iss >> u >> v)) {
                cout << "Debug: Failed to parse line: " << line << endl;
                continue; // Skip invalid lines
            }
            
            uniqueVertices.insert(u);
            uniqueVertices.insert(v);
        }
        
        // Initialize graph
        vector<int> vertices(uniqueVertices.begin(), uniqueVertices.end());
        sort(vertices.begin(), vertices.end());
        
        // Decide whether to use lookup table
        bool useContiguousIds = true;
        for (size_t i = 0; i < vertices.size(); i++) {
            if (vertices[i] != (int)i) {
                useContiguousIds = false;
                break;
            }
        }
        
        if (useContiguousIds) {
            cout << "Using contiguous vertex IDs..." << endl;
            initialize(vertices.size());
        } else {
            cout << "Using vertex ID lookup table..." << endl;
            initializeWithLookup(vertices);
        }
        
        // Second pass: add all edges
        file.clear();
        file.seekg(0);
        
        cout << "Loading edges..." << endl;
        int edgeCount = 0;
        int lineCount = 0;
        auto startTime = chrono::high_resolution_clock::now();
        
        while (getline(file, line)) {
            lineCount++;
            
            if (lineCount % 1000000 == 0) {
                auto currentTime = chrono::high_resolution_clock::now();
                auto duration = chrono::duration_cast<chrono::seconds>(currentTime - startTime).count();
                cout << "  Processed " << lineCount << " lines, " << edgeCount << " edges added (" 
                     << (duration > 0 ? lineCount / duration : lineCount) << " lines/sec)" << endl;
            }
            
            if (line.empty() || line[0] == '#') continue;
            
            istringstream iss(line);
            int u, v;
            if (!(iss >> u >> v)) {
                continue;
            }
            
            addEdge(u, v);
            edgeCount++;
        }
        
        cout << "Loaded graph with " << getNumVertices() << " vertices and " << getNumEdges() << " edges." << endl;
        return true;
    }
};

// Thread pool for parallel computation
class ThreadPool {
private:
    vector<thread> workers;
    queue<function<void()>> tasks;
    
    mutex queue_mutex;
    condition_variable condition;
    bool stop;

public:
    ThreadPool(size_t numThreads) : stop(false) {
        for (size_t i = 0; i < numThreads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    function<void()> task;
                    
                    {
                        unique_lock<mutex> lock(queue_mutex);
                        condition.wait(lock, [this] { return stop || !tasks.empty(); });
                        
                        if (stop && tasks.empty()) {
                            return;
                        }
                        
                        task = move(tasks.front());
                        tasks.pop();
                    }
                    
                    task();
                }
            });
        }
    }
    
    template<class F>
    void enqueue(F&& f) {
        {
            unique_lock<mutex> lock(queue_mutex);
            if (stop) throw runtime_error("enqueue on stopped ThreadPool");
            tasks.emplace(forward<F>(f));
        }
        condition.notify_one();
    }

    // Get the number of pending tasks
    size_t pendingTasks() {
        unique_lock<mutex> lock(queue_mutex);
        return tasks.size();
    }
    
    ~ThreadPool() {
        {
            unique_lock<mutex> lock(queue_mutex);
            stop = true;
        }
        
        condition.notify_all();
        
        for (thread& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }
};

// Optimized solver class
class OptimizedDensityGradientSolver {
private:
    const OptimizedGraph& graph;
    unordered_map<int, double> clusteringCoefficients;
    vector<int> bestSolutionOverall;
    mutex bestSolutionMutex;
    atomic<bool> solutionFound{false};
    atomic<int> completedSeeds{0};
    int totalSeeds;

    // Pre-compute clustering coefficients in parallel
    void precomputeClusteringCoefficients(int numThreads) {
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

    // Calculate vertex potential
    double calculateVertexPotential(int v) const {
        auto it = clusteringCoefficients.find(v);
        double clustering = (it != clusteringCoefficients.end()) ? it->second : 0.0;
        double degree = graph.getDegree(v);
        return 0.7 * degree + 0.3 * clustering * degree;
    }

    // Efficient check if a subgraph is a quasi-clique
    bool isQuasiClique(const vector<int>& nodes) const {
        int n = nodes.size();
        int possibleEdges = (n * (n - 1)) / 2;
        int minEdgesNeeded = (possibleEdges / 2) + 1; // Strictly more than half
        
        int actualEdges = countEdges(nodes);
        return actualEdges >= minEdgesNeeded;
    }

    // Count edges in a subgraph efficiently
    int countEdges(const vector<int>& nodes) const {
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

    // Find boundary vertices
    unordered_set<int> findBoundaryVertices(const vector<int>& solution) const {
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

    // Calculate connections from a candidate to the current solution
    int countConnectionsToSolution(int candidate, const vector<int>& solution) const {
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

    // Find a single quasi-clique starting from a seed vertex
    vector<int> findQuasiCliqueFromSeed(int seed, int seedIdx) {
        cout << "Processing seed " << seedIdx + 1 << "/" << totalSeeds << endl;
        cout << "  Starting from seed: " << seed 
             << " (degree: " << graph.getDegree(seed) 
             << ", clustering: " << clusteringCoefficients[seed] << ")" << endl;
        
        vector<int> solution = {seed};
        
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
            
            // Evaluate each boundary vertex
            for (int candidate : boundary) {
                // Count connections to current solution
                int connections = countConnectionsToSolution(candidate, solution);
                
                // Enhanced score with clustering coefficient
                double directRatio = static_cast<double>(connections) / solution.size();
                double candidateClustering = clusteringCoefficients[candidate];
                
                // Alpha decreases as the solution grows (adaptive weighting)
                double alpha = max(0.5, 0.9 - 0.01 * solution.size());
                double score = alpha * directRatio + (1 - alpha) * candidateClustering;
                
                // Check if adding maintains quasi-clique property
                vector<int> newSolution = solution;
                newSolution.push_back(candidate);
                
                if (isQuasiClique(newSolution) && score > bestScore) {
                    bestScore = score;
                    bestCandidate = candidate;
                }
                
                // Check for termination request
                if (terminationRequested) {
                    break;
                }
            }
            
            // If no suitable candidate found, break
            if (bestCandidate == -1) {
                cout << "    No more candidates found after " << iteration << " iterations" << endl;
                break;
            }
            
            // Add the best candidate
            solution.push_back(bestCandidate);
            
            // Progress reporting
            if (iteration % 10 == 0) {
                cout << "    Iteration " << iteration << ": solution size = " << solution.size() << endl;
            }
            
            // Check if solution is better than current best, and if so, save it immediately
            // This ensures we have a solution even if the program is terminated
            if (solution.size() > 5) {  // Only consider solutions of reasonable size
                lock_guard<mutex> lock(bestSolutionMutex);
                if (solution.size() > bestSolutionOverall.size() && isQuasiClique(solution)) {
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
        
        return solution;
    }

public:
    OptimizedDensityGradientSolver(const OptimizedGraph& g) : graph(g) {}

    // Find a large quasi-clique using multiple threads
    vector<int> findLargeQuasiClique(int numSeeds = 20, int numThreads = 1) {
        // Determine number of threads to use
        if (numThreads <= 0) {
            numThreads = thread::hardware_concurrency();
            if (numThreads == 0) numThreads = 1;
        }
        
        totalSeeds = numSeeds;
        
        cout << "Using " << numThreads << " threads" << endl;
        
        // Register signal handlers for graceful termination
        signal(SIGINT, signalHandler);  // Ctrl+C
        signal(SIGTERM, signalHandler); // kill command
        
        // Pre-compute clustering coefficients
        precomputeClusteringCoefficients(numThreads);
        
        if (terminationRequested) {
            cout << "Termination requested during preprocessing. Exiting." << endl;
            return bestSolutionOverall;
        }
        
        // Sort vertices by potential
        vector<int> vertices = graph.getVertices();
        cout << "Sorting " << vertices.size() << " vertices by potential..." << endl;
        
        sort(vertices.begin(), vertices.end(), [this](int a, int b) {
            return calculateVertexPotential(a) > calculateVertexPotential(b);
        });

        if (vertices.size() > 5) {
            cout << "Top 5 vertices by potential:" << endl;
            for (int i = 0; i < min(5, (int)vertices.size()); i++) {
                int v = vertices[i];
                cout << "  " << v << ": degree=" << graph.getDegree(v) 
                     << ", clustering=" << clusteringCoefficients[v] 
                     << ", potential=" << calculateVertexPotential(v) << endl;
            }
        }
        
        // Process seeds in parallel
        int seedsToTry = min(numSeeds, (int)vertices.size());
        ThreadPool pool(numThreads);
        
        for (int seedIdx = 0; seedIdx < seedsToTry; seedIdx++) {
            int seed = vertices[seedIdx];
            
            pool.enqueue([this, seed, seedIdx, seedsToTry]() {
                if (terminationRequested) return;
                
                vector<int> solution = findQuasiCliqueFromSeed(seed, seedIdx);
                
                // Update best solution if better
                lock_guard<mutex> lock(bestSolutionMutex);
                if (solution.size() > bestSolutionOverall.size() && isQuasiClique(solution)) {
                    bestSolutionOverall = solution;
                    solutionFound = true;
                    cout << "New best solution found: " << bestSolutionOverall.size() << " vertices" << endl;
                }
            });
        }
        
        // Wait for all threads to finish
        // Instead of implicit wait in ThreadPool destructor, monitor and report progress
        while (completedSeeds < seedsToTry && !terminationRequested) {
            this_thread::sleep_for(chrono::seconds(5));
            cout << "Progress: " << completedSeeds << "/" << seedsToTry << " seeds processed, "
                 << "pending tasks: " << pool.pendingTasks() << endl;
        }
        
        if (terminationRequested) {
            cout << "Termination requested. Returning best solution found so far." << endl;
        }
        
        return bestSolutionOverall;
    }

    // Verify and print information about the solution
    void verifyAndPrintSolution(const vector<int>& solution) {
        int n = solution.size();
        int possibleEdges = (n * (n - 1)) / 2;
        int actualEdges = countEdges(solution);
        double density = (n > 1) ? static_cast<double>(actualEdges) / possibleEdges : 0;
        
        cout << "\n=== Solution Summary ===" << endl;
        cout << "Vertices: " << n << endl;
        cout << "Edges: " << actualEdges << "/" << possibleEdges << endl;
        cout << "Density: " << density << endl;
        cout << "Minimum required edges for quasi-clique: " << (possibleEdges / 2) + 1 << endl;
        
        if (n > 0) {
            if (actualEdges > possibleEdges / 2) {
                cout << "✓ Solution is a valid quasi-clique!" << endl;
            } else {
                cout << "✗ Solution is NOT a valid quasi-clique!" << endl;
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

    // Save solution to file
    bool saveSolution(const vector<int>& solution, const string& filename = "solution.txt") {
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

    // Getter for solutionFound flag
    bool hasSolution() const {
        return solutionFound;
    }
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <graph_file> [num_seeds] [num_threads]" << endl;
        return 1;
    }
    
    string filename = argv[1];
    int numSeeds = (argc > 2) ? stoi(argv[2]) : 20;
    int numThreads = (argc > 3) ? stoi(argv[3]) : 0; // 0 means use all available
    
    // Print system information
    cout << "System information:" << endl;
    cout << "  Hardware concurrency: " << thread::hardware_concurrency() << " threads" << endl;
    
    // Load graph
    OptimizedGraph graph;
    auto loadStartTime = chrono::high_resolution_clock::now();
    
    if (!graph.loadFromFile(filename)) {
        return 1;
    }
    
    auto loadEndTime = chrono::high_resolution_clock::now();
    auto loadDuration = chrono::duration_cast<chrono::milliseconds>(loadEndTime - loadStartTime).count();
    cout << "Graph loaded in " << loadDuration / 1000.0 << " seconds" << endl;
    
    // Create solver
    OptimizedDensityGradientSolver solver(graph);
    
    // Measure execution time
    auto startTime = chrono::high_resolution_clock::now();
    
    // Find large quasi-clique
    vector<int> solution = solver.findLargeQuasiClique(numSeeds, numThreads);
    
    auto endTime = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
    
    // Verify and print solution
    solver.verifyAndPrintSolution(solution);
    
    cout << "\nAlgorithm execution time: " << duration / 1000.0 << " seconds" << endl;
    cout << "Total time (including loading): " << (duration + loadDuration) / 1000.0 << " seconds" << endl;
    
    // Check if we have a solution (either from normal processing or from checkpoint)
    if (solution.empty() && !solver.hasSolution()) {
        // If no solution found, check if there's a solution_in_progress.txt
        ifstream tempSolutionFile("solution_in_progress.txt");
        if (tempSolutionFile.good()) {
            cout << "No final solution, but found in-progress solution. Using that instead." << endl;
            tempSolutionFile.close();
            
            // Load the in-progress solution
            vector<int> tempSolution;
            tempSolutionFile.open("solution_in_progress.txt");
            int vertex;
            while (tempSolutionFile >> vertex) {
                tempSolution.push_back(vertex);
            }
            tempSolutionFile.close();
            
            if (!tempSolution.empty()) {
                cout << "Loaded " << tempSolution.size() << " vertices from in-progress solution." << endl;
                solver.verifyAndPrintSolution(tempSolution);
                solver.saveSolution(tempSolution);
                return 0;
            }
        }
        
        cout << "No solution found. Exiting." << endl;
        // Still save an empty file to indicate the process completed
        solver.saveSolution(solution);
        return 0;
    }
    
    // Save solution to file
    solver.saveSolution(solution);
    
    return 0;
}