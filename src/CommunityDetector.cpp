#include "CommunityDetector.h"
#include <unordered_map>
#include <chrono>
#include <iostream>
#include <csignal>

using namespace std;

CommunityDetector::CommunityDetector(const Graph& g) : graph(g), numCommunities(0) {}

double CommunityDetector::calculateInterCommunityConnectivity(int community1, int community2) const {
    // Get vertices in each community
    std::vector<int> vertices1;
    std::vector<int> vertices2;
    
    for (int i = 0; i < (int)nodeCommunities.size(); i++) {
        if (nodeCommunities[i] == community1) {
            vertices1.push_back(i);
        } else if (nodeCommunities[i] == community2) {
            vertices2.push_back(i);
        }
    }
    
    if (vertices1.empty() || vertices2.empty()) {
        return 0.0;
    }
    
    // Count inter-community edges
    int interEdges = 0;
    for (int v1 : vertices1) {
        const auto& neighbors = graph.getNeighbors(v1);
        for (int neighbor : neighbors) {
            if (std::find(vertices2.begin(), vertices2.end(), neighbor) != vertices2.end()) {
                interEdges++;
            }
        }
    }
    
    // Calculate maximum possible edges between communities
    int maxPossibleEdges = vertices1.size() * vertices2.size();
    
    // Return density of connections
    return (double)interEdges / maxPossibleEdges;
}

void CommunityDetector::mergeCommunities(double connectivityThreshold) {
    std::cout << "Merging communities with connectivity threshold: " << connectivityThreshold << std::endl;
    
    // Get all communities
    std::vector<std::vector<int>> communities(numCommunities);
    for (int i = 0; i < (int)nodeCommunities.size(); i++) {
        int community = nodeCommunities[i];
        if (community >= 0 && community < numCommunities) {
            communities[community].push_back(i);
        }
    }
    
    // Build a meta-graph of community connections
    std::vector<std::vector<double>> interCommunityConnectivity(numCommunities, 
                                                            std::vector<double>(numCommunities, 0.0));
    
    std::cout << "  Calculating inter-community connectivity..." << std::endl;
    // Calculate connectivity between each pair of communities
    for (int i = 0; i < numCommunities; i++) {
        for (int j = i+1; j < numCommunities; j++) {
            double connectivity = calculateInterCommunityConnectivity(i, j);
            interCommunityConnectivity[i][j] = connectivity;
            interCommunityConnectivity[j][i] = connectivity;
            
            if (connectivity > connectivityThreshold) {
                std::cout << "  Community " << i << " and " << j << " have high connectivity: " << connectivity << std::endl;
            }
        }
        
        // Check for termination request
        if (terminationRequested) {
            std::cout << "  Termination requested during community connectivity calculation" << std::endl;
            return;
        }
    }
    
    // Merge communities with connectivity above threshold
    std::vector<bool> merged(numCommunities, false);
    std::vector<std::vector<int>> mergedCommunities;
    
    std::cout << "  Merging highly connected communities..." << std::endl;
    for (int i = 0; i < numCommunities; i++) {
        if (merged[i]) continue;
        
        std::vector<int> newCommunity = communities[i];
        merged[i] = true;
        
        // Find and merge connected communities
        for (int j = 0; j < numCommunities; j++) {
            if (i == j || merged[j]) continue;
            
            if (interCommunityConnectivity[i][j] > connectivityThreshold) {
                // Merge community j into new community
                std::cout << "    Merging community " << j << " into community " << i << std::endl;
                newCommunity.insert(newCommunity.end(), communities[j].begin(), communities[j].end());
                merged[j] = true;
            }
        }
        
        mergedCommunities.push_back(newCommunity);
        
        // Check for termination request
        if (terminationRequested) {
            std::cout << "  Termination requested during community merging" << std::endl;
            return;
        }
    }
    
    std::cout << "  Reduced from " << numCommunities << " to " << mergedCommunities.size() << " communities" << std::endl;
    
    // Update communities
    updateCommunities(mergedCommunities);
}

void CommunityDetector::updateCommunities(const std::vector<std::vector<int>>& newCommunities) {
    // Update community assignments
    numCommunities = newCommunities.size();
    
    // Reset all communities to -1 (unassigned)
    std::fill(nodeCommunities.begin(), nodeCommunities.end(), -1);
    
    // Assign new communities
    for (int commIdx = 0; commIdx < numCommunities; commIdx++) {
        for (int node : newCommunities[commIdx]) {
            if (node >= 0 && node < (int)nodeCommunities.size()) {
                nodeCommunities[node] = commIdx;
            }
        }
    }
    
    std::cout << "Communities updated: now have " << numCommunities << " communities" << std::endl;
}

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