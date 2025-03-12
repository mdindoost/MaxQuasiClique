#include "Graph.h"

using namespace std;

Graph::Graph() {}

void Graph::initialize(int numVertices) {
    adjacencyLists.resize(numVertices);
}

void Graph::initializeWithLookup(const vector<int>& vertices) {
    usingLookupTable = true;
    indexToVertex.reserve(vertices.size());
    
    for (size_t i = 0; i < vertices.size(); i++) {
        vertexToIndex[vertices[i]] = i;
        indexToVertex.push_back(vertices[i]);
    }
    
    adjacencyLists.resize(vertices.size());
}

int Graph::mapVertex(int vertex) const {
    if (!usingLookupTable) return vertex;
    auto it = vertexToIndex.find(vertex);
    return (it != vertexToIndex.end()) ? it->second : -1;
}

int Graph::mapIndex(int index) const {
    if (!usingLookupTable) return index;
    return (index >= 0 && index < (int)indexToVertex.size()) ? indexToVertex[index] : -1;
}

void Graph::addEdge(int u, int v) {
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

bool Graph::hasEdge(int u, int v) const {
    int uIdx = mapVertex(u);
    int vIdx = mapVertex(v);
    
    if (uIdx < 0 || vIdx < 0 || uIdx >= (int)adjacencyLists.size() || vIdx >= (int)adjacencyLists.size()) {
        return false;
    }
    
    const auto& neighbors = adjacencyLists[uIdx];
    return find(neighbors.begin(), neighbors.end(), vIdx) != neighbors.end();
}

const vector<int>& Graph::getNeighbors(int v) const {
    static const vector<int> emptyVector;
    int vIdx = mapVertex(v);
    
    if (vIdx < 0 || vIdx >= (int)adjacencyLists.size()) {
        return emptyVector;
    }
    
    return adjacencyLists[vIdx];
}

int Graph::getDegree(int v) const {
    int vIdx = mapVertex(v);
    
    if (vIdx < 0 || vIdx >= (int)adjacencyLists.size()) {
        return 0;
    }
    
    return adjacencyLists[vIdx].size();
}

int Graph::getNumVertices() const {
    return adjacencyLists.size();
}

int Graph::getNumEdges() const {
    return numEdges;
}

vector<int> Graph::getVertices() const {
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

double Graph::getClusteringCoefficient(int v) const {
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

bool Graph::loadFromFile(const string& filename) {
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