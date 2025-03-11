// MaxQuasiClique - Density Gradient with Clustering Coefficient Algorithm
// This program finds large quasi-cliques in undirected graphs using a density gradient approach

#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cmath>
#include <chrono>

using namespace std;

// Graph representation using adjacency list
class Graph {
private:
    unordered_map<int, unordered_set<int>> adjacencyList;
    int numVertices = 0;
    int numEdges = 0;

public:
    // Add vertex to the graph
    void addVertex(int v) {
        if (adjacencyList.find(v) == adjacencyList.end()) {
            adjacencyList[v] = unordered_set<int>();
            numVertices++;
        }
    }

    // Add edge to the graph
    void addEdge(int u, int v) {
        if (u == v) return; // No self-loops
        
        // Ensure vertices exist
        addVertex(u);
        addVertex(v);
        
        // Add edge if it doesn't exist
        if (adjacencyList[u].find(v) == adjacencyList[u].end()) {
            adjacencyList[u].insert(v);
            adjacencyList[v].insert(u);
            numEdges++;
        }
    }

    // Check if edge exists
    bool hasEdge(int u, int v) const {
        auto it = adjacencyList.find(u);
        if (it == adjacencyList.end()) return false;
        return it->second.find(v) != it->second.end();
    }

    // Get neighbors of a vertex
    const unordered_set<int>& getNeighbors(int v) const {
        static const unordered_set<int> emptySet;
        auto it = adjacencyList.find(v);
        if (it == adjacencyList.end()) return emptySet;
        return it->second;
    }

    // Get degree of a vertex
    int getDegree(int v) const {
        auto it = adjacencyList.find(v);
        if (it == adjacencyList.end()) return 0;
        return it->second.size();
    }

    // Get all vertices in the graph
    vector<int> getVertices() const {
        vector<int> vertices;
        vertices.reserve(adjacencyList.size());
        for (const auto& pair : adjacencyList) {
            vertices.push_back(pair.first);
        }
        return vertices;
    }

    // Get total number of vertices
    int getNumVertices() const {
        return numVertices;
    }

    // Get total number of edges
    int getNumEdges() const {
        return numEdges;
    }

    // Calculate clustering coefficient for a vertex
    double getClusteringCoefficient(int v) const {
        const auto& neighbors = getNeighbors(v);
        int k = neighbors.size();
        
        if (k < 2) return 0.0; // Not enough neighbors to form triangles
        
        // Count connections between neighbors
        int connections = 0;
        for (int u : neighbors) {
            for (int w : neighbors) {
                if (u < w && hasEdge(u, w)) { // Count each pair once
                    connections++;
                }
            }
        }
        
        double possibleConnections = (k * (k - 1)) / 2.0;
        return connections / possibleConnections;
    }

    // Load graph from edge list file
    bool loadFromFile(const string& filename) {
        ifstream file(filename);
        if (!file.is_open()) {
            cerr << "Error: Could not open file " << filename << endl;
            return false;
        }

        string line;
        while (getline(file, line)) {
            istringstream iss(line);
            int u, v;
            if (!(iss >> u >> v)) continue; // Skip invalid lines
            addEdge(u, v);
        }

        cout << "Loaded graph with " << numVertices << " vertices and " << numEdges << " edges." << endl;
        return true;
    }
};

// Class implementing the Density Gradient with Clustering Coefficient algorithm
class DensityGradientSolver {
private:
    const Graph& graph;
    unordered_map<int, double> clusteringCoefficients;

    // Pre-compute clustering coefficients for all vertices
    void precomputeClusteringCoefficients() {
        cout << "Pre-computing clustering coefficients..." << endl;
        for (int v : graph.getVertices()) {
            clusteringCoefficients[v] = graph.getClusteringCoefficient(v);
        }
        cout << "Done pre-computing clustering coefficients." << endl;
    }

    // Calculate vertex potential (combination of degree and clustering)
    double calculateVertexPotential(int v) const {
        double degree = graph.getDegree(v);
        double clustering = clusteringCoefficients.at(v);
        return 0.7 * degree + 0.3 * clustering * degree;
    }

    // Check if a subgraph satisfies the quasi-clique property
    bool isQuasiClique(const vector<int>& nodes) const {
        int n = nodes.size();
        int possibleEdges = (n * (n - 1)) / 2;
        int minEdgesNeeded = (possibleEdges / 2) + 1; // Strictly more than half
        
        int actualEdges = 0;
        for (size_t i = 0; i < nodes.size(); i++) {
            for (size_t j = i + 1; j < nodes.size(); j++) {
                if (graph.hasEdge(nodes[i], nodes[j])) {
                    actualEdges++;
                }
            }
        }
        
        return actualEdges >= minEdgesNeeded;
    }

    // Count edges in a subgraph
    int countEdges(const vector<int>& nodes) const {
        int count = 0;
        for (size_t i = 0; i < nodes.size(); i++) {
            for (size_t j = i + 1; j < nodes.size(); j++) {
                if (graph.hasEdge(nodes[i], nodes[j])) {
                    count++;
                }
            }
        }
        return count;
    }

public:
    DensityGradientSolver(const Graph& g) : graph(g) {
        precomputeClusteringCoefficients();
    }

    // Main algorithm to find large quasi-cliques
    vector<int> findLargeQuasiClique(int numSeeds = 20) {
        // Sort vertices by potential
        vector<int> vertices = graph.getVertices();
        sort(vertices.begin(), vertices.end(), [this](int a, int b) {
            return calculateVertexPotential(a) > calculateVertexPotential(b);
        });

        vector<int> bestSolution;
        
        // Try multiple seed vertices
        int seedsToTry = min(numSeeds, static_cast<int>(vertices.size()));
        for (int seedIdx = 0; seedIdx < seedsToTry; seedIdx++) {
            int seed = vertices[seedIdx];
            
            cout << "Trying seed " << seedIdx + 1 << "/" << seedsToTry 
                 << ": vertex " << seed 
                 << " (degree: " << graph.getDegree(seed) 
                 << ", clustering: " << clusteringCoefficients[seed] << ")" << endl;
            
            vector<int> currentSolution = {seed};
            
            // Expansion phase
            int iteration = 0;
            while (true) {
                iteration++;
                int bestCandidate = -1;
                double bestScore = -1;
                
                // Find boundary vertices
                unordered_set<int> boundary;
                for (int v : currentSolution) {
                    const auto& neighbors = graph.getNeighbors(v);
                    boundary.insert(neighbors.begin(), neighbors.end());
                }
                
                // Remove vertices already in the solution
                for (int v : currentSolution) {
                    boundary.erase(v);
                }
                
                // Evaluate each boundary vertex
                for (int candidate : boundary) {
                    // Count connections to current solution
                    int connections = 0;
                    for (int v : currentSolution) {
                        if (graph.hasEdge(candidate, v)) {
                            connections++;
                        }
                    }
                    
                    // Enhanced score with clustering coefficient
                    double directRatio = static_cast<double>(connections) / currentSolution.size();
                    double candidateClustering = clusteringCoefficients[candidate];
                    
                    // Alpha decreases as the solution grows (adaptive weighting)
                    double alpha = max(0.5, 0.9 - 0.01 * currentSolution.size());
                    double score = alpha * directRatio + (1 - alpha) * candidateClustering;
                    
                    // Check if adding maintains quasi-clique property
                    vector<int> newSolution = currentSolution;
                    newSolution.push_back(candidate);
                    
                    if (isQuasiClique(newSolution) && score > bestScore) {
                        bestScore = score;
                        bestCandidate = candidate;
                    }
                }
                
                // If no suitable candidate found, break
                if (bestCandidate == -1) {
                    cout << "  No more candidates found after " << iteration << " iterations" << endl;
                    cout << "  Final solution size: " << currentSolution.size() << endl;
                    break;
                }
                
                // Add the best candidate
                currentSolution.push_back(bestCandidate);
                
                // Progress reporting
                if (iteration % 10 == 0) {
                    cout << "  Iteration " << iteration << ": solution size = " << currentSolution.size() << endl;
                }
            }
            
            // Update best solution
            if (currentSolution.size() > bestSolution.size()) {
                bestSolution = currentSolution;
                cout << "New best solution found: " << bestSolution.size() << " vertices" << endl;
            }
        }
        
        return bestSolution;
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
            
            cout << "\nVertices in solution: ";
            for (int v : solution) {
                cout << v << " ";
            }
            cout << endl;
        } else {
            cout << "No solution found." << endl;
        }
    }
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <graph_file> [num_seeds]" << endl;
        return 1;
    }
    
    string filename = argv[1];
    int numSeeds = (argc > 2) ? stoi(argv[2]) : 20;
    
    // Load graph
    Graph graph;
    if (!graph.loadFromFile(filename)) {
        return 1;
    }
    
    // Create solver
    DensityGradientSolver solver(graph);
    
    // Measure execution time
    auto startTime = chrono::high_resolution_clock::now();
    
    // Find large quasi-clique
    vector<int> solution = solver.findLargeQuasiClique(numSeeds);
    
    auto endTime = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
    
    // Verify and print solution
    solver.verifyAndPrintSolution(solution);
    
    cout << "\nExecution time: " << duration / 1000.0 << " seconds" << endl;
    
    // Save solution to file
    ofstream solutionFile("solution.txt");
    for (int v : solution) {
        solutionFile << v << endl;
    }
    cout << "Solution saved to solution.txt" << endl;
    
    return 0;
}