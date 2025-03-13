#ifndef COMMUNITY_DETECTOR_H
#define COMMUNITY_DETECTOR_H

#include "Graph.h"
#include <vector>
#include <csignal>  // Add this line for sig_atomic_t

// Forward declaration of global termination variable
extern volatile sig_atomic_t terminationRequested;

/**
 * Community detection using Louvain method
 */
class CommunityDetector {
private:
    const Graph& graph;
    int numCommunities;
    std::vector<int> nodeCommunities;
    
    /**
     * Detect communities using Louvain method
     */
    void detectCommunitiesLouvain(int numPasses = 3);

public:
    /**
     * Constructor
     */
    CommunityDetector(const Graph& g);
    
    /**
     * Detect communities in the graph
     */
    void detectCommunities();
    
    /**
     * Get the community of a node
     */
    int getCommunity(int node) const;
    
    /**
     * Get all communities
     */
    std::vector<std::vector<int>> getAllCommunities() const;
    
    /**
     * Get community sizes
     */
    std::vector<int> getCommunitySizes() const;
    
    /**
     * Get number of communities
     */
    int getNumCommunities() const;
    
    /**
     * Find boundary vertices (vertices with neighbors in different communities)
     */
    std::vector<int> findBoundaryVertices() const;
        /**
     * Calculate inter-community connectivity
     */
    double calculateInterCommunityConnectivity(int community1, int community2) const;
    
    /**
     * Merge communities based on connectivity threshold
     */
    void mergeCommunities(double connectivityThreshold);
    
    /**
     * Update communities after merging
     */
    void updateCommunities(const std::vector<std::vector<int>>& newCommunities);
};

#endif // COMMUNITY_DETECTOR_H