#include "CommunityDetector.h"
#include <unordered_map>
#include <chrono>
#include <iostream>
#include <csignal>

using namespace std;

CommunityDetector::CommunityDetector(const Graph& g) : graph(g), numCommunities(0) {}

void CommunityDetector::detectCommunitiesLouvain(int numPasses) {
    int numNodes = graph.getNumVertices();
    nodeCommunities.resize(numNodes);
    
    // Initialize: each node in its own community
    for (int i = 0; i < numNodes; i++) {
        nodeCommunities[i] = i;
    }
    
    numCommunities = numNodes;
    
    // Run multiple passes of the algorithm
    for (int pass = 0; pass < numPasses && !terminationRequested; pass++) {
        cout << "Community detection: pass " << pass + 1 << "/" << numPasses << endl;
        
        bool improvement = false;
        
        // Local optimization phase
        for (int node = 0; node < numNodes && !terminationRequested; node++) {
            int currentCommunity = nodeCommunities[node];
            
            // Calculate gains for moving to neighbor communities
            unordered_map<int, double> communityGains;
            
            // Process neighbors
            const auto& neighbors = graph.getNeighbors(node);
            for (int neighbor : neighbors) {
                int neighborCommunity = nodeCommunities[neighbor];
                communityGains[neighborCommunity] += 1.0;  // Simple gain metric
            }
            
            // Find best community
            int bestCommunity = currentCommunity;
            double bestGain = 0.0;
            
            for (const auto& pair : communityGains) {
                int community = pair.first;
                double gain = pair.second;
                
                if (gain > bestGain) {
                    bestGain = gain;
                    bestCommunity = community;
                }
            }
            
            // Move to best community if there's improvement
            if (bestCommunity != currentCommunity) {
                nodeCommunities[node] = bestCommunity;
                improvement = true;
            }
            
            // Periodically report progress
            if (node % 10000 == 0) {
                cout << "  Processed " << node << "/" << numNodes << " nodes" << endl;
            }
        }
        
        // Consolidate communities
        vector<int> communityMap(numNodes, -1);
        int newNumCommunities = 0;
        
        for (int i = 0; i < numNodes; i++) {
            int community = nodeCommunities[i];
            if (communityMap[community] == -1) {
                communityMap[community] = newNumCommunities++;
            }
            nodeCommunities[i] = communityMap[nodeCommunities[i]];
        }
        
        numCommunities = newNumCommunities;
        cout << "  Found " << numCommunities << " communities after pass " << pass + 1 << endl;
        
        // If no improvement, stop
        if (!improvement) {
            cout << "  No improvement, stopping community detection" << endl;
            break;
        }
    }
}

void CommunityDetector::detectCommunities() {
    cout << "Detecting communities using Louvain method..." << endl;
    auto startTime = chrono::high_resolution_clock::now();
    
    detectCommunitiesLouvain();
    
    auto endTime = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::seconds>(endTime - startTime).count();
    
    cout << "Community detection completed in " << duration << " seconds" << endl;
    cout << "Detected " << numCommunities << " communities" << endl;
}

int CommunityDetector::getCommunity(int node) const {
    if (node >= 0 && node < (int)nodeCommunities.size()) {
        return nodeCommunities[node];
    }
    return -1;
}

vector<vector<int>> CommunityDetector::getAllCommunities() const {
    vector<vector<int>> communities(numCommunities);
    
    for (int i = 0; i < (int)nodeCommunities.size(); i++) {
        int community = nodeCommunities[i];
        communities[community].push_back(i);
    }
    
    return communities;
}

vector<int> CommunityDetector::getCommunitySizes() const {
    vector<int> sizes(numCommunities, 0);
    
    for (int community : nodeCommunities) {
        sizes[community]++;
    }
    
    return sizes;
}

int CommunityDetector::getNumCommunities() const {
    return numCommunities;
}

vector<int> CommunityDetector::findBoundaryVertices() const {
    vector<int> boundaryVertices;
    
    for (int i = 0; i < (int)nodeCommunities.size(); i++) {
        int community = nodeCommunities[i];
        const auto& neighbors = graph.getNeighbors(i);
        
        bool isBoundary = false;
        for (int neighbor : neighbors) {
            if (nodeCommunities[neighbor] != community) {
                isBoundary = true;
                break;
            }
        }
        
        if (isBoundary) {
            boundaryVertices.push_back(i);
        }
    }
    
    return boundaryVertices;
}