#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>

/**
 * Optimized graph representation using compressed adjacency lists
 */
class Graph {
private:
    // Vector of adjacency lists
    std::vector<std::vector<int>> adjacencyLists;
    int numEdges = 0;

    // Optional lookup table for non-contiguous vertex IDs
    std::unordered_map<int, int> vertexToIndex;
    std::vector<int> indexToVertex;
    bool usingLookupTable = false;

public:
    // Constructor
    Graph();
    
    // Initialize the graph with a known number of vertices
    void initialize(int numVertices);

    // Initialize with vertex lookup (for non-contiguous IDs)
    void initializeWithLookup(const std::vector<int>& vertices);

    // Map external vertex ID to internal index
    int mapVertex(int vertex) const;

    // Map internal index to external vertex ID
    int mapIndex(int index) const;

    // Add edge to the graph
    void addEdge(int u, int v);

    // Check if edge exists
    bool hasEdge(int u, int v) const;

    // Get neighbors of a vertex
    const std::vector<int>& getNeighbors(int v) const;

    // Get degree of a vertex
    int getDegree(int v) const;

    // Get number of vertices
    int getNumVertices() const;

    // Get number of edges
    int getNumEdges() const;

    // Get all vertices
    std::vector<int> getVertices() const;

    // Calculate clustering coefficient efficiently
    double getClusteringCoefficient(int v) const;

    // Load graph from edge list file
    bool loadFromFile(const std::string& filename);
};

#endif // GRAPH_H