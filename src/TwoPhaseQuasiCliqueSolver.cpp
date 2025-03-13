#include "TwoPhaseQuasiCliqueSolver.h"
#include <iostream>
#include <fstream>
#include <random>
#include <algorithm>
#include <chrono>
#include <queue>
#include <csignal>

using namespace std;

TwoPhaseQuasiCliqueSolver::TwoPhaseQuasiCliqueSolver(const Graph& g) 
    : graph(g), totalSeeds(0), communityDetector(g) {}

    std::vector<int> TwoPhaseQuasiCliqueSolver::getBoundaryVerticesToExplore(
        const std::vector<std::vector<int>>& solutions, int maxSeeds) {
        
        std::cout << "Finding boundary vertices to explore from " << solutions.size() << " solutions..." << std::endl;
        
        // Collect all solution vertices for fast lookup
        std::unordered_set<int> solutionVertices;
        for (const auto& solution : solutions) {
            solutionVertices.insert(solution.begin(), solution.end());
        }
        
        // Find boundary vertices with their connections to solutions
        std::unordered_map<int, int> boundaryConnections;
        
        for (const auto& solution : solutions) {
            std::unordered_set<int> boundary = findBoundaryVertices(solution);
            
            for (int v : boundary) {
                if (solutionVertices.count(v) == 0) { // Not already in a solution
                    boundaryConnections[v] += countConnectionsToSolution(v, solution);
                }
            }
        }
        
        // Convert to vector for sorting
        std::vector<std::pair<int, int>> rankedBoundary;
        for (const auto& pair : boundaryConnections) {
            rankedBoundary.push_back(pair);
        }
        
        // Sort by connection count (descending)
        std::sort(rankedBoundary.begin(), rankedBoundary.end(),
             [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                 return a.second > b.second;
             });
        
        // Take top vertices
        std::vector<int> result;
        for (int i = 0; i < std::min(maxSeeds, (int)rankedBoundary.size()); i++) {
            result.push_back(rankedBoundary[i].first);
        }
        
        std::cout << "Selected " << result.size() << " boundary vertices to explore" << std::endl;
        return result;
    }
    
    std::vector<int> TwoPhaseQuasiCliqueSolver::getPerturbedSeeds(
        const std::vector<std::vector<int>>& solutions, int numSeeds) {
        
        std::cout << "Generating perturbed seeds from " << solutions.size() << " solutions..." << std::endl;
        
        std::vector<int> result;
        if (solutions.empty()) return result;
        
        // Only use top solutions
        const int MAX_SOLUTIONS_TO_PERTURB = 5;
        int solutionsToUse = std::min(MAX_SOLUTIONS_TO_PERTURB, (int)solutions.size());
        
        for (int solIdx = 0; solIdx < solutionsToUse; solIdx++) {
            const auto& solution = solutions[solIdx];
            
            // Skip very small solutions
            if (solution.size() < 5) continue;
            
            // Create several perturbed versions
            for (int perturbIdx = 0; perturbIdx < 3; perturbIdx++) {
                // Select a subset of vertices (70-90% of original)
                int subsetSize = solution.size() * (0.7 + (rand() % 20) / 100.0);
                std::vector<int> subset;
                
                // Shuffle and select
                std::vector<int> shuffled = solution;
                std::random_device rd;
                std::mt19937 g(rd());
                std::shuffle(shuffled.begin(), shuffled.end(), g);
                subset.assign(shuffled.begin(), shuffled.begin() + subsetSize);
                
                // Calculate high centrality vertices in this subset
                std::vector<std::pair<int, double>> vertexCentrality;
                for (int v : subset) {
                    int connections = 0;
                    for (int u : subset) {
                        if (v != u && graph.hasEdge(v, u)) {
                            connections++;
                        }
                    }
                    double centrality = (double)connections / subset.size();
                    vertexCentrality.push_back({v, centrality});
                }
                
                // Sort by centrality
                std::sort(vertexCentrality.begin(), vertexCentrality.end(),
                     [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                         return a.second > b.second;
                     });
                
                // Take top 3 vertices as seeds from this perturbed solution
                int toTake = std::min(3, (int)vertexCentrality.size());
                for (int i = 0; i < toTake; i++) {
                    result.push_back(vertexCentrality[i].first);
                }
            }
        }
        
        // Add some high k-core vertices that are not in any solution
        std::vector<std::pair<int, int>> vertexWithCore = computeKCoreDecomposition();
        if (!vertexWithCore.empty()) {
            // Sort by k-core (descending)
            std::sort(vertexWithCore.begin(), vertexWithCore.end(),
                 [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                     return a.second > b.second;
                 });
            
            // Collect all vertices in solutions
            std::unordered_set<int> solutionVertices;
            for (const auto& solution : solutions) {
                solutionVertices.insert(solution.begin(), solution.end());
            }
            
            // Add high k-core vertices not in solutions
            int remaining = numSeeds - result.size();
            for (const auto& pair : vertexWithCore) {
                int v = pair.first;
                if (solutionVertices.count(v) == 0) {
                    result.push_back(v);
                    if (--remaining <= 0) break;
                }
            }
        }
        
        // If we still need more seeds, add random high-degree vertices
        if (result.size() < static_cast<size_t>(numSeeds)) {
            std::vector<int> vertices = graph.getVertices();
            std::sort(vertices.begin(), vertices.end(), [this](int a, int b) {
                return graph.getDegree(a) > graph.getDegree(b);
            });
            
            std::unordered_set<int> selectedSet(result.begin(), result.end());
            
            for (int v : vertices) {
                if (selectedSet.count(v) == 0) {
                    result.push_back(v);
                    if (result.size() < static_cast<size_t>(numSeeds)) break;
                }
            }
        }
        
        std::cout << "Generated " << result.size() << " perturbed seeds" << std::endl;
        return result;
    }
    
    void TwoPhaseQuasiCliqueSolver::multiRoundExploration(int numSeeds, int numRounds, int numThreads) {
        std::cout << "Starting multi-round exploration with " << numRounds << " rounds..." << std::endl;
        
        // Store top solutions from each round
        std::vector<std::vector<int>> persistentSolutions;
        
        for (int round = 0; round < numRounds && !terminationRequested; round++) {
            std::cout << "==== Starting exploration round " << round + 1 << "/" << numRounds << " ====" << std::endl;
            
            // Select seeds with different strategies in different rounds
            std::vector<int> seeds;
            if (round == 0) {
                // Round 1: Use enhanced k-core selection
                seeds = enhancedKCoreSeedSelection(numSeeds);
            } else if (round == 1) {
                // Round 2: Use boundary vertices of best solutions so far
                seeds = getBoundaryVerticesToExplore(persistentSolutions, numSeeds);
            } else {
                // Other rounds: Use perturbed versions of best solutions
                seeds = getPerturbedSeeds(persistentSolutions, numSeeds);
            }
            
            if (seeds.empty() || terminationRequested) {
                std::cout << "No seeds selected or termination requested. Skipping round." << std::endl;
                continue;
            }
            
            totalSeeds = seeds.size();
            completedSeeds = 0;
            
            // Process seeds in parallel
            ThreadPool pool(numThreads);
            std::vector<std::vector<int>> roundSolutions;
            std::mutex roundSolutionsMutex;
            
            std::cout << "Processing " << seeds.size() << " seeds with " << numThreads << " threads..." << std::endl;
            
            for (size_t seedIdx = 0; seedIdx < seeds.size(); seedIdx++) {
                int seed = seeds[seedIdx];
                int community = communityDetector.getCommunity(seed);
                
                pool.enqueue([this, seed, seedIdx, community, &roundSolutionsMutex, &roundSolutions]() {
                    if (terminationRequested) return;
                    
                    std::vector<int> solution = {seed};
                    std::vector<int> expanded = improvedExpansionMethod(solution);
                    
                    if (expanded.size() > 5 && isQuasiClique(expanded) && isConnected(expanded)) {
                        // Add to round solutions
                        {
                            std::lock_guard<std::mutex> lock(roundSolutionsMutex);
                            roundSolutions.push_back(expanded);
                        }
                        
                        // Update best solution if better
                        {
                            std::lock_guard<std::mutex> lock(bestSolutionMutex);
                            if (expanded.size() > bestSolutionOverall.size()) {
                                bestSolutionOverall = expanded;
                                solutionFound = true;
                                
                                // Save progress
                                std::ofstream solutionFile("solution_in_progress.txt");
                                for (int v : bestSolutionOverall) {
                                    solutionFile << v << std::endl;
                                }
                                solutionFile.close();
                                
                                std::cout << "New best solution found: " << bestSolutionOverall.size() 
                                          << " vertices" << std::endl;
                            }
                        }
                    }
                    
                    completedSeeds++;
                });
            }
            
            // Wait for all threads to finish
            while (completedSeeds < totalSeeds && !terminationRequested) {
                std::this_thread::sleep_for(std::chrono::seconds(5));
                std::cout << "Progress: " << completedSeeds << "/" << totalSeeds 
                        << " seeds processed, solutions found in this round: " 
                        << roundSolutions.size() << std::endl;
            }
            
            // Sort round solutions by size (descending)
            std::sort(roundSolutions.begin(), roundSolutions.end(), 
                [](const std::vector<int>& a, const std::vector<int>& b) { 
                    return a.size() > b.size(); 
                });
            
            // Keep top solutions from this round
            const int TOP_SOLUTIONS_TO_KEEP = 20;
            if (roundSolutions.size() > TOP_SOLUTIONS_TO_KEEP) {
                roundSolutions.resize(TOP_SOLUTIONS_TO_KEEP);
            }
            
            std::cout << "Round " << round + 1 << " complete. Found " << roundSolutions.size() 
                      << " solutions. Best size: " << (roundSolutions.empty() ? 0 : roundSolutions[0].size()) 
                      << std::endl;
            
            // Add to persistent solutions
            persistentSolutions.insert(persistentSolutions.end(), 
                                    roundSolutions.begin(), 
                                    roundSolutions.end());
            
            // Sort and trim persistent solutions
            std::sort(persistentSolutions.begin(), persistentSolutions.end(),
                [](const std::vector<int>& a, const std::vector<int>& b) {
                    return a.size() > b.size();
                });
            
            const int MAX_PERSISTENT_SOLUTIONS = 50;
            if (persistentSolutions.size() > MAX_PERSISTENT_SOLUTIONS) {
                persistentSolutions.resize(MAX_PERSISTENT_SOLUTIONS);
            }
            
            std::cout << "Accumulated " << persistentSolutions.size() 
                     << " solutions across all rounds" << std::endl;
            
            // Try to merge solutions within this round
            if (!roundSolutions.empty() && !terminationRequested) {
                std::cout << "Attempting to merge solutions from this round..." << std::endl;
                candidateSolutions = roundSolutions;
                phase2_refineSolutions();
            }
        }
        
        // Set candidate solutions for final phase 2
        candidateSolutions = persistentSolutions;
        
        std::cout << "Multi-round exploration complete." << std::endl;
        std::cout << "Best solution size: " << bestSolutionOverall.size() << std::endl;
        std::cout << "Candidate solutions for final phase: " << candidateSolutions.size() << std::endl;
    }

    std::vector<int> TwoPhaseQuasiCliqueSolver::enhancedKCoreSeedSelection(int numSeeds) {
        std::cout << "Selecting seeds using enhanced k-core approach..." << std::endl;
        
        // First, compute k-core decomposition
        std::vector<std::pair<int, int>> vertexWithCore = computeKCoreDecomposition();
        
        if (terminationRequested) {
            std::cout << "Termination requested during k-core computation" << std::endl;
            return std::vector<int>();
        }
        
        // Sort by k-core value (descending)
        std::sort(vertexWithCore.begin(), vertexWithCore.end(),
             [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                 return a.second > b.second;
             });
        
        // Get highest k-core value
        int maxKCore = vertexWithCore.empty() ? 0 : vertexWithCore[0].second;
        
        std::cout << "  Maximum k-core value: " << maxKCore << std::endl;
        
        // Focus on vertices in the top k-cores
        std::vector<int> highKCoreVertices;
        for (const auto& pair : vertexWithCore) {
            // Select vertices with k-core at least 80% of max
            if (pair.second >= maxKCore * 0.8) {
                highKCoreVertices.push_back(pair.first);
            }
        }
        
        std::cout << "  Found " << highKCoreVertices.size() << " vertices in top k-cores" << std::endl;
        
        // If we have very few high k-core vertices, lower the threshold
        if (highKCoreVertices.size() < static_cast<size_t>(numSeeds)) {
            highKCoreVertices.clear();
            for (const auto& pair : vertexWithCore) {
                // Lower threshold to 60% of max
                if (pair.second >= maxKCore * 0.6) {
                    highKCoreVertices.push_back(pair.first);
                }
            }
            std::cout << "  Lowered threshold: now have " << highKCoreVertices.size() << " vertices" << std::endl;
        }
        
        // Now distribute these among communities
        std::unordered_map<int, std::vector<int>> communityToVertices;
        for (int v : highKCoreVertices) {
            int community = communityDetector.getCommunity(v);
            communityToVertices[community].push_back(v);
        }
        
        std::cout << "  These vertices span " << communityToVertices.size() << " communities" << std::endl;
        
        // Select seeds from each community proportionally
        std::vector<int> selectedSeeds;
        
        // Sort communities by size (number of high k-core vertices)
        std::vector<std::pair<int, int>> communitySizes;
        for (const auto& pair : communityToVertices) {
            communitySizes.push_back({pair.first, (int)pair.second.size()});
        }
        
        std::sort(communitySizes.begin(), communitySizes.end(),
             [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                 return a.second > b.second;
             });
        
        // Process each community
        for (const auto& pair : communitySizes) {
            int communityId = pair.first;
            std::vector<int>& vertices = communityToVertices[communityId];
            
            // Skip empty communities
            if (vertices.empty()) continue;
            
            // Sort vertices in this community by quality score
            std::sort(vertices.begin(), vertices.end(), [this](int a, int b) {
                double aClust = clusteringCoefficients.count(a) ? clusteringCoefficients[a] : 0;
                double bClust = clusteringCoefficients.count(b) ? clusteringCoefficients[b] : 0;
                
                double aScore = aClust * graph.getDegree(a);
                double bScore = bClust * graph.getDegree(b);
                return aScore > bScore;
            });
            
            // Take top vertices from this community
            int communitySize = vertices.size();
            
            // Allocate seeds proportionally to community size, minimum 1
            int totalHighKCore = highKCoreVertices.size();
            int seedsToTake = std::max(1, std::min(communitySize, 
                                  (int)(numSeeds * communitySize / totalHighKCore)));
            
            std::cout << "    Taking " << seedsToTake << " seeds from community " << communityId 
                      << " (size: " << communitySize << ")" << std::endl;
            
            for (int i = 0; i < seedsToTake && i < communitySize; i++) {
                selectedSeeds.push_back(vertices[i]);
                if (selectedSeeds.size() >= static_cast<size_t>(numSeeds)) break;
            }
            
            if (selectedSeeds.size() >= static_cast<size_t>(numSeeds)) break;
        }
        
        // If we still have slots, add more from largest communities
        if (selectedSeeds.size() < static_cast<size_t>(numSeeds)) {
            std::cout << "  Adding more seeds to reach target count..." << std::endl;
            
            // Re-sort communities with updated vertex lists (some may have been selected already)
            communitySizes.clear();
            for (auto& pair : communityToVertices) {
                // Remove already selected vertices
                auto& vertices = pair.second;
                vertices.erase(
                    std::remove_if(vertices.begin(), vertices.end(),
                        [&selectedSeeds](int v) {
                            return std::find(selectedSeeds.begin(), selectedSeeds.end(), v) != selectedSeeds.end();
                        }),
                    vertices.end());
                
                communitySizes.push_back({pair.first, (int)vertices.size()});
            }
            
            std::sort(communitySizes.begin(), communitySizes.end(),
                 [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                     return a.second > b.second;
                 });
            
            for (const auto& pair : communitySizes) {
                int communityId = pair.first;
                const std::vector<int>& vertices = communityToVertices[communityId];
                
                for (int v : vertices) {
                    selectedSeeds.push_back(v);
                    if (selectedSeeds.size() >= static_cast<size_t>(numSeeds)) break;
                }
                
                if (selectedSeeds.size() >= static_cast<size_t>(numSeeds)) break;
            }
        }
        
        std::cout << "Selected " << selectedSeeds.size() << " seeds using enhanced k-core selection" << std::endl;
        
        return selectedSeeds;
    }

    double TwoPhaseQuasiCliqueSolver::evaluateCandidate(int vertex, const std::vector<int>& solution) {
        // Calculate direct connections
        int connections = countConnectionsToSolution(vertex, solution);
        double connectionRatio = static_cast<double>(connections) / solution.size();
        
        // Get clustering coefficient
        auto it = clusteringCoefficients.find(vertex);
        double clustering = (it != clusteringCoefficients.end()) ? it->second : 0.0;
        
        // Get degree of vertex
        int degree = graph.getDegree(vertex);
        
        // Calculate score based on connection ratio, clustering, and degree
        double alpha = 0.7;  // Weight for connection ratio
        double beta = 0.2;   // Weight for clustering
        double gamma = 0.1;  // Weight for normalized degree
        
        // Normalize degree (assuming average degree is around 20, adjust as needed)
        double normalizedDegree = min(1.0, degree / 50.0);
        
        // Combine factors
        double score = alpha * connectionRatio + beta * clustering + gamma * normalizedDegree;
        
        return score;
    }
    
    std::vector<int> TwoPhaseQuasiCliqueSolver::improvedExpansionMethod(const std::vector<int>& initialSolution) {
        std::cout << "  Starting improved expansion from solution of size " << initialSolution.size() << std::endl;
        
        std::vector<int> solution = initialSolution;
        
        // Maximum number of attempts without improvement before giving up
        const int MAX_STAGNANT_ITERATIONS = 20;
        int stagnantIterations = 0;
        int totalIterations = 0;
        
        while (stagnantIterations < MAX_STAGNANT_ITERATIONS && !terminationRequested) {
            totalIterations++;
            int sizeBeforeIteration = solution.size();
            
            // Find all promising boundary vertices
            std::unordered_set<int> boundary = findBoundaryVertices(solution);
            std::vector<std::pair<int, double>> rankedCandidates;
            
            // Score all boundary vertices
            for (int v : boundary) {
                double score = evaluateCandidate(v, solution);
                rankedCandidates.push_back({v, score});
            }
            
            // Sort by score (descending)
            std::sort(rankedCandidates.begin(), rankedCandidates.end(),
                 [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                     return a.second > b.second;
                 });
            
            // Try multiple candidates before giving up on this cluster
            bool improved = false;
            for (int i = 0; i < std::min(10, (int)rankedCandidates.size()); i++) {
                int candidate = rankedCandidates[i].first;
                std::vector<int> newSolution = solution;
                newSolution.push_back(candidate);
                
                if (isQuasiClique(newSolution) && isConnected(newSolution)) {
                    solution = newSolution;
                    improved = true;
                    break;
                }
            }
            
            // Check if we made any progress
            if (!improved || solution.size() == static_cast<size_t>(sizeBeforeIteration)) {
                stagnantIterations++;
            } else {
                stagnantIterations = 0; // Reset counter when we improve
            }
            
            // Progress reporting
            if (totalIterations % 10 == 0 || stagnantIterations >= MAX_STAGNANT_ITERATIONS) {
                std::cout << "    Iteration " << totalIterations 
                          << ": solution size = " << solution.size()
                          << ", stagnant iterations = " << stagnantIterations << std::endl;
            }
            
            // Check if solution is better than current best, and if so, save it immediately
            if (solution.size() > 5) {
                std::lock_guard<std::mutex> lock(bestSolutionMutex);
                if (solution.size() > bestSolutionOverall.size() && isQuasiClique(solution) && isConnected(solution)) {
                    bestSolutionOverall = solution;
                    solutionFound = true;
                    
                    // Save progress
                    std::ofstream solutionFile("solution_in_progress.txt");
                    for (int v : bestSolutionOverall) {
                        solutionFile << v << std::endl;
                    }
                    solutionFile.close();
                    
                    std::cout << "  New best solution found: " << bestSolutionOverall.size() 
                              << " vertices (saved to solution_in_progress.txt)" << std::endl;
                }
            }
        }
        
        std::cout << "  Improved expansion completed after " << totalIterations 
                  << " iterations, final size: " << solution.size() << std::endl;
        
        return solution;
    }
    std::vector<int> TwoPhaseQuasiCliqueSolver::selectSeedsWithKCoreAndCommunityAwareness(int numSeeds) {
        std::cout << "Selecting seeds with k-core and community awareness..." << std::endl;
        
        // Compute k-core decomposition
        std::vector<std::pair<int, int>> vertexWithCore = computeKCoreDecomposition();
        
        if (terminationRequested) {
            return std::vector<int>();
        }
        
        // Get community sizes and sort communities by size (descending)
        std::vector<int> sizes = communityDetector.getCommunitySizes();
        std::vector<std::pair<int, int>> communitySizes;
        for (int i = 0; i < (int)sizes.size(); i++) {
            communitySizes.push_back({i, sizes[i]});
        }
        
        std::sort(communitySizes.begin(), communitySizes.end(), 
                 [](const std::pair<int, int>& a, const std::pair<int, int>& b) { 
                     return a.second > b.second;
                 });
        
        std::cout << "  Community sizes: ";
        for (int i = 0; i < std::min(5, (int)communitySizes.size()); i++) {
            std::cout << communitySizes[i].second << " ";
        }
        if (communitySizes.size() > 5) std::cout << "...";
        std::cout << std::endl;
        
        // First, get boundary vertices (vertices connecting different communities)
        std::vector<int> boundaryVertices = communityDetector.findBoundaryVertices();
        
        // Enhance boundary vertices with k-core values
        std::vector<std::tuple<int, int, int>> boundaryWithCore; // (vertex, coreness, community)
        for (int v : boundaryVertices) {
            // Find this vertex in vertexWithCore
            for (const auto& pair : vertexWithCore) {
                if (pair.first == v) {
                    boundaryWithCore.push_back(std::make_tuple(
                        v, 
                        pair.second, 
                        communityDetector.getCommunity(v)
                    ));
                    break;
                }
            }
        }
        
        // Sort boundary vertices by coreness (descending)
        std::sort(boundaryWithCore.begin(), boundaryWithCore.end(), 
                 [](const std::tuple<int, int, int>& a, const std::tuple<int, int, int>& b) {
                     return std::get<1>(a) > std::get<1>(b);
                 });
        
        // Take top 20% of seeds from boundary vertices
        int boundaryCount = std::min(numSeeds / 5, (int)boundaryWithCore.size());
        std::vector<int> selectedSeeds;
        selectedSeeds.reserve(numSeeds);
        
        std::cout << "  Selecting " << boundaryCount << " boundary vertices as seeds" << std::endl;
        for (int i = 0; i < boundaryCount; i++) {
            selectedSeeds.push_back(std::get<0>(boundaryWithCore[i]));
        }
        
        // Allocate remaining seeds to communities proportionally
        int remainingSeeds = numSeeds - boundaryCount;
        std::vector<int> seedsPerCommunity(communityDetector.getNumCommunities(), 0);
        
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
        
        // For each community, select vertices with highest k-core values
        for (int communityIdx = 0; communityIdx < communityDetector.getNumCommunities(); communityIdx++) {
            int seedsToSelect = seedsPerCommunity[communityIdx];
            if (seedsToSelect == 0) continue;
            
            std::cout << "  Selecting " << seedsToSelect << " seeds from community " << communityIdx << std::endl;
            
            // Get vertices in this community with their k-core values
            std::vector<std::pair<int, int>> communityVerticesWithCore;
            
            for (const auto& pair : vertexWithCore) {
                int vertex = pair.first;
                int coreness = pair.second;
                
                if (communityDetector.getCommunity(vertex) == communityIdx) {
                    communityVerticesWithCore.push_back({vertex, coreness});
                }
            }
            
            // Sort by coreness (descending)
            std::sort(communityVerticesWithCore.begin(), communityVerticesWithCore.end(), 
                     [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                         return a.second > b.second;
                     });
            
            // Select top vertices
            int actualSeedsToSelect = std::min(seedsToSelect, (int)communityVerticesWithCore.size());
            for (int i = 0; i < actualSeedsToSelect; i++) {
                selectedSeeds.push_back(communityVerticesWithCore[i].first);
            }
        }
        
        std::cout << "Selected " << selectedSeeds.size() << " seeds using k-core and community awareness" << std::endl;
        
        return selectedSeeds;
    }


    std::vector<std::pair<int, int>> TwoPhaseQuasiCliqueSolver::computeKCoreDecomposition() {
        std::cout << "Computing k-core decomposition..." << std::endl;
        
        std::vector<int> vertices = graph.getVertices();
        int n = vertices.size();
        
        // Map vertex IDs to indices
        std::unordered_map<int, int> vertexToIndex;
        for (int i = 0; i < n; i++) {
            vertexToIndex[vertices[i]] = i;
        }
        
        // Initialize with degrees
        std::vector<int> degrees(n);
        for (int i = 0; i < n; i++) {
            degrees[i] = graph.getDegree(vertices[i]);
        }
        
        // Find maximum degree
        int maxDegree = 0;
        for (int d : degrees) {
            maxDegree = std::max(maxDegree, d);
        }
        
        // Create bins for each degree
        std::vector<std::vector<int>> bins(maxDegree + 1);
        for (int i = 0; i < n; i++) {
            bins[degrees[i]].push_back(i);
        }
        
        // Array to track if a vertex has been processed
        std::vector<bool> processed(n, false);
        
        // Resulting core numbers
        std::vector<int> coreNumbers(n, 0);
        
        // Process vertices in order of increasing degree
        int currentK = 0;
        int numProcessed = 0;
        
        std::cout << "  Running k-core algorithm..." << std::endl;
        auto startTime = std::chrono::high_resolution_clock::now();
        
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
                    auto currentTime = std::chrono::high_resolution_clock::now();
                    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                        currentTime - startTime).count();
                    
                    std::cout << "    Processed " << numProcessed << "/" << n 
                              << " vertices (" << (elapsed > 0 ? numProcessed / elapsed : numProcessed) 
                              << " vertices/sec)" << std::endl;
                }
                
                // Update neighbors
                for (int neighbor : graph.getNeighbors(vertices[vIdx])) {
                    auto it = vertexToIndex.find(neighbor);
                    if (it == vertexToIndex.end()) continue;
                    
                    int nIdx = it->second;
                    if (!processed[nIdx] && degrees[nIdx] > currentK) {
                        // Remove from current bin
                        auto binIt = std::find(bins[degrees[nIdx]].begin(), bins[degrees[nIdx]].end(), nIdx);
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
        
        // Create pairs of (vertex, coreness)
        std::vector<std::pair<int, int>> vertexWithCore;
        for (int i = 0; i < n; i++) {
            vertexWithCore.push_back({vertices[i], coreNumbers[i]});
        }
        
        std::cout << "  K-core decomposition complete. Maximum coreness: " 
                  << *std::max_element(coreNumbers.begin(), coreNumbers.end()) << std::endl;
        
        return vertexWithCore;
    }
    
    std::vector<int> TwoPhaseQuasiCliqueSolver::selectSeedsBasedOnKCore(int numSeeds) {
        std::cout << "Selecting seeds based on k-core values..." << std::endl;
        
        // Compute k-core decomposition
        std::vector<std::pair<int, int>> vertexWithCore = computeKCoreDecomposition();
        
        if (terminationRequested) {
            std::cout << "Termination requested during k-core computation" << std::endl;
            return std::vector<int>();
        }
        
        // Sort by coreness (descending)
        std::sort(vertexWithCore.begin(), vertexWithCore.end(), 
                 [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                     return a.second > b.second || (a.second == b.second && a.first < b.first);
                 });
        
        // Select top seeds
        int seedsToSelect = std::min(numSeeds, (int)vertexWithCore.size());
        std::vector<int> selectedSeeds;
        selectedSeeds.reserve(seedsToSelect);
        
        for (int i = 0; i < seedsToSelect; i++) {
            selectedSeeds.push_back(vertexWithCore[i].first);
        }
        
        std::cout << "Selected " << selectedSeeds.size() << " seeds based on k-core values" << std::endl;
        std::cout << "  Highest coreness: " << vertexWithCore[0].second << std::endl;
        
        return selectedSeeds;
    }
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

std::vector<int> TwoPhaseQuasiCliqueSolver::performLocalSearch(const std::vector<int>& initialSolution) {
    std::cout << "  Performing local search on solution of size " << initialSolution.size() << std::endl;
    
    // Start with the initial solution
    std::vector<int> solution = initialSolution;
    
    // Convert to set for efficient lookup
    std::unordered_set<int> solutionSet(solution.begin(), solution.end());
    
    // Variables to track improvements
    bool improved = true;
    int iterations = 0;
    const int MAX_ITERATIONS = 20;
    
    while (improved && iterations < MAX_ITERATIONS && !terminationRequested) {
        improved = false;
        iterations++;
        
        // Step 1: Try to add high-value boundary vertices
        std::unordered_set<int> boundary = findBoundaryVertices(solution);
        
        // For each boundary vertex, calculate its value to the solution
        std::vector<std::pair<int, double>> vertexValues;
        
        for (int v : boundary) {
            // Calculate connections to solution
            int connections = countConnectionsToSolution(v, solution);
            double connectionRatio = static_cast<double>(connections) / solution.size();
            
            // Get clustering coefficient
            auto it = clusteringCoefficients.find(v);
            double clustering = (it != clusteringCoefficients.end()) ? it->second : 0.0;
            
            // Calculate a combined score
            double score = 0.7 * connectionRatio + 0.3 * clustering;
            
            vertexValues.push_back({v, score});
        }
        
        // Sort by value (descending)
        std::sort(vertexValues.begin(), vertexValues.end(), 
                 [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                     return a.second > b.second;
                 });
        
        // Try to add top boundary vertices
        int numAdded = 0;
        for (int i = 0; i < std::min(20, (int)vertexValues.size()); i++) {
            int v = vertexValues[i].first;
            
            // Check if adding would maintain quasi-clique property
            std::vector<int> newSolution = solution;
            newSolution.push_back(v);
            
            if (isQuasiClique(newSolution) && isConnected(newSolution)) {
                solution = newSolution;
                solutionSet.insert(v);
                improved = true;
                numAdded++;
            }
        }
        
        if (numAdded > 0) {
            std::cout << "    Added " << numAdded << " vertices in iteration " << iterations << std::endl;
            continue;  // Skip to next iteration
        }
        
        // Step 2: If no vertices can be added, try to swap low-value vertices for better ones
        
        // Calculate vertex values within the solution
        std::vector<std::pair<int, double>> internalVertexValues;
        
        for (int v : solution) {
            // Calculate connections to other solution nodes
            int connections = 0;
            for (int other : solution) {
                if (v != other && graph.hasEdge(v, other)) {
                    connections++;
                }
            }
            
            double connectionRatio = static_cast<double>(connections) / (solution.size() - 1);
            internalVertexValues.push_back({v, connectionRatio});
        }
        
        // Sort by value (ascending - worst first)
        std::sort(internalVertexValues.begin(), internalVertexValues.end(), 
                 [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                     return a.second < b.second;
                 });
        
        // Try to swap low-value vertices for high-value boundary vertices
        for (int i = 0; i < std::min(5, (int)internalVertexValues.size()); i++) {
            int vToRemove = internalVertexValues[i].first;
            
            // Try each high-value boundary vertex
            for (int j = 0; j < std::min(10, (int)vertexValues.size()); j++) {
                int vToAdd = vertexValues[j].first;
                
                // Create test solution with swap
                std::vector<int> testSolution;
                for (int v : solution) {
                    if (v != vToRemove) {
                        testSolution.push_back(v);
                    }
                }
                testSolution.push_back(vToAdd);
                
                // Check if valid
                if (isQuasiClique(testSolution) && isConnected(testSolution)) {
                    solution = testSolution;
                    solutionSet.erase(vToRemove);
                    solutionSet.insert(vToAdd);
                    improved = true;
                    
                    std::cout << "    Swapped vertex " << vToRemove << " for " << vToAdd 
                              << " in iteration " << iterations << std::endl;
                    
                    break;  // Move to next internal vertex
                }
            }
            
            if (improved) break;  // If we made an improvement, start next iteration
        }
    }
    
    std::cout << "  Local search completed after " << iterations << " iterations, final size: " 
              << solution.size() << std::endl;
    
    return solution;
}
// void TwoPhaseQuasiCliqueSolver::phase1_findCandidateSolutions(int numSeeds, int numThreads) {
//     cout << "Phase 1: Finding candidate solutions using community-aware approach..." << endl;
    
//     // Sort vertices by potential
//     vector<int> vertices = graph.getVertices();
//     cout << "Sorting " << vertices.size() << " vertices by potential..." << endl;
    
//     sort(vertices.begin(), vertices.end(), [this](int a, int b) {
//         return calculateVertexPotential(a) > calculateVertexPotential(b);
//     });
    
//     // Select seeds with community awareness
//     vector<int> seeds = selectSeedsWithCommunityAwareness(vertices, numSeeds);
//     totalSeeds = seeds.size();
    
//     cout << "Selected " << seeds.size() << " seeds with community awareness" << endl;
    
//     // Process seeds in parallel
//     ThreadPool pool(numThreads);
    
//     for (size_t seedIdx = 0; seedIdx < seeds.size(); seedIdx++) {
//         int seed = seeds[seedIdx];
//         int community = communityDetector.getCommunity(seed);
        
//         pool.enqueue([this, seed, seedIdx, community]() {
//             if (terminationRequested) return;
            
//             vector<int> solution = findQuasiCliqueFromSeed(seed, seedIdx, community);
            
//             // Update best solution if better
//             if (solution.size() > 5 && isQuasiClique(solution) && isConnected(solution)) {
//                 lock_guard<mutex> lock(bestSolutionMutex);
//                 if (solution.size() > bestSolutionOverall.size()) {
//                     bestSolutionOverall = solution;
//                     solutionFound = true;
//                     cout << "New best solution found: " << bestSolutionOverall.size() << " vertices" << endl;
//                 }
//             }
//         });
//     }
    
//     // Wait for all threads to finish
//     while (completedSeeds < totalSeeds && !terminationRequested) {
//         this_thread::sleep_for(chrono::seconds(5));
//         cout << "Progress: " << completedSeeds << "/" << totalSeeds 
//              << " seeds processed, candidate solutions: " << candidateSolutions.size() << endl;
//     }
    
//     cout << "Phase 1 complete. Found " << candidateSolutions.size() << " candidate solutions." << endl;
    
//     // Sort candidate solutions by size (descending)
//     sort(candidateSolutions.begin(), candidateSolutions.end(), 
//          [](const vector<int>& a, const vector<int>& b) { return a.size() > b.size(); });
    
//     // Keep only top 100 solutions to limit computational complexity in phase 2
//     if (candidateSolutions.size() > 100) {
//         cout << "Limiting to top 100 candidate solutions for phase 2" << endl;
//         candidateSolutions.resize(100);
//     }
// }
void TwoPhaseQuasiCliqueSolver::phase1_findCandidateSolutions(int numSeeds, int numThreads) {
    cout << "Phase 1: Finding candidate solutions using enhanced approach..." << endl;
    
    // Select seeds using the enhanced method
    vector<int> seeds = selectSeedsWithKCoreAndCommunityAwareness(numSeeds);
    totalSeeds = seeds.size();
    
    cout << "Selected " << seeds.size() << " seeds with k-core and community awareness" << endl;
    
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
                    
                    // Save progress
                    ofstream solutionFile("solution_in_progress.txt");
                    for (int v : bestSolutionOverall) {
                        solutionFile << v << endl;
                    }
                    solutionFile.close();
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
    
    // Keep only top solutions to limit computational complexity in phase 2
    if (candidateSolutions.size() > 300) {
        cout << "Limiting to top 300 candidate solutions for phase 2" << endl;
        candidateSolutions.resize(300);
    }
}
// void TwoPhaseQuasiCliqueSolver::phase2_refineSolutions() {
//     cout << "Phase 2: Refining and merging solutions..." << endl;
    
//     if (candidateSolutions.empty()) {
//         cout << "No candidate solutions to refine." << endl;
//         return;
//     }
    
//     // First, try to merge solutions
//     cout << "Attempting to merge solutions..." << endl;
    
//     bool improved = true;
//     int mergeIterations = 0;
    
//     // Modified: increased max iterations to 10 and adjusted similarity thresholds
//     while (improved && !terminationRequested && mergeIterations < 10) {
//         improved = false;
//         mergeIterations++;
        
//         cout << "Merge iteration " << mergeIterations << endl;
        
//         vector<vector<int>> newSolutions;
        
//         // Try all pairs of solutions
//         for (size_t i = 0; i < candidateSolutions.size(); i++) {
//             for (size_t j = i + 1; j < candidateSolutions.size(); j++) {
//                 // Calculate Jaccard similarity to quickly filter out unlikely pairs
//                 double similarity = calculateJaccardSimilarity(candidateSolutions[i], candidateSolutions[j]);
                
//                 // Modified: expanded similarity range from [0.1, 0.8] to [0.05, 0.9]
//                 if (similarity >= 0.05 && similarity <= 0.9) {
//                     if (canMerge(candidateSolutions[i], candidateSolutions[j])) {
//                         vector<int> merged = mergeSolutions(candidateSolutions[i], candidateSolutions[j]);
                        
//                         if (merged.size() > max(candidateSolutions[i].size(), candidateSolutions[j].size())) {
//                             newSolutions.push_back(merged);
//                             improved = true;
//                             cout << "  Merged solutions of sizes " << candidateSolutions[i].size() 
//                                  << " and " << candidateSolutions[j].size() 
//                                  << " into new solution of size " << merged.size() << endl;
//                         }
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
        
//         // Limit candidates to keep computational complexity manageable
//         if (candidateSolutions.size() > 100) {
//             candidateSolutions.resize(100);
//         }
//     }
    
//     cout << "Phase 2 complete. Best solution size: " << bestSolutionOverall.size() << endl;
// }
void TwoPhaseQuasiCliqueSolver::phase2_refineSolutions() {
    cout << "Phase 2: Refining and merging solutions (enhanced version)..." << endl;
    
    if (candidateSolutions.empty()) {
        cout << "No candidate solutions to refine." << endl;
        return;
    }
    
    // First, try to merge solutions
    cout << "Attempting to merge " << candidateSolutions.size() << " solutions..." << endl;
        
    bool improved = true;
    int mergeIterations = 0;
    
    // Increased max iterations and relaxed similarity thresholds
    while (improved && !terminationRequested && mergeIterations < 20) {
        improved = false;
        mergeIterations++;
        
        cout << "Merge iteration " << mergeIterations << endl;
        
        vector<vector<int>> newSolutions;
        
        // Try all pairs of solutions
        for (size_t i = 0; i < candidateSolutions.size(); i++) {
            for (size_t j = i + 1; j < candidateSolutions.size(); j++) {
                // Calculate Jaccard similarity
                double similarity = calculateJaccardSimilarity(candidateSolutions[i], candidateSolutions[j]);
                
                // Extremely relaxed similarity range to try more combinations
                if (similarity >= 0.01 && similarity <= 0.99) {
                    // Try the aggressive merge approach
                    vector<int> merged = attemptToMerge(candidateSolutions[i], candidateSolutions[j]);
                    
                    if (!merged.empty() && merged.size() > max(candidateSolutions[i].size(), candidateSolutions[j].size())) {
                        newSolutions.push_back(merged);
                        improved = true;
                        cout << "  Merged solutions of sizes " << candidateSolutions[i].size() 
                             << " and " << candidateSolutions[j].size() 
                             << " into new solution of size " << merged.size() << endl;
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
        
        // Keep more candidates for more opportunities to merge
        if (candidateSolutions.size() > 500) {
            candidateSolutions.resize(500);
        }
    }
    
    // After merging, try to improve individual solutions using local search
    if (!candidateSolutions.empty() && !terminationRequested) {
        cout << "Performing local search on " << min(10, (int)candidateSolutions.size()) 
             << " best solutions..." << endl;
        
        vector<vector<int>> improvedSolutions;
        
        // Only try local search on top solutions
        for (int i = 0; i < min(10, (int)candidateSolutions.size()); i++) {
            vector<int> improved = performLocalSearch(candidateSolutions[i]);
            
            if (improved.size() > candidateSolutions[i].size()) {
                cout << "  Improved solution from " << candidateSolutions[i].size() 
                     << " to " << improved.size() << " vertices" << endl;
                
                improvedSolutions.push_back(improved);
                
                // Update best solution if needed
                if (improved.size() > bestSolutionOverall.size()) {
                    bestSolutionOverall = improved;
                    
                    // Save best solution
                    ofstream solutionFile("solution_in_progress.txt");
                    for (int v : bestSolutionOverall) {
                        solutionFile << v << endl;
                    }
                    solutionFile.close();
                    
                    cout << "New best solution found: " << bestSolutionOverall.size() << " vertices" << endl;
                }
            }
            
            if (terminationRequested) break;
        }
        
        // Add improved solutions to candidate pool
        for (const auto& solution : improvedSolutions) {
            candidateSolutions.push_back(solution);
        }
        
        // Sort solutions by size (descending)
        sort(candidateSolutions.begin(), candidateSolutions.end(), 
             [](const vector<int>& a, const vector<int>& b) { return a.size() > b.size(); });
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


std::vector<int> TwoPhaseQuasiCliqueSolver::attemptToMerge(
    const std::vector<int>& solution1, const std::vector<int>& solution2) const {

// Combine the solutions
std::vector<int> combined;
combined.reserve(solution1.size() + solution2.size());

// Add all vertices from solution1
combined.insert(combined.end(), solution1.begin(), solution1.end());

// Add unique vertices from solution2
std::unordered_set<int> solution1Set(solution1.begin(), solution1.end());
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
std::vector<std::pair<int, double>> nodeConnectivity;
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
std::sort(nodeConnectivity.begin(), nodeConnectivity.end(), 
     [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
         return a.second < b.second;
     });

// Try removing up to 40% of the lowest-connectivity nodes
int maxToRemove = combined.size() * 0.4;
std::vector<int> prunedSolution = combined;

for (int i = 0; i < std::min(maxToRemove, (int)nodeConnectivity.size()); i++) {
    int nodeToRemove = nodeConnectivity[i].first;
    prunedSolution.erase(std::remove(prunedSolution.begin(), prunedSolution.end(), nodeToRemove), 
                        prunedSolution.end());
    
    if (isQuasiClique(prunedSolution) && isConnected(prunedSolution) && 
        prunedSolution.size() > std::max(solution1.size(), solution2.size())) {
        return prunedSolution;
    }
}

// If we couldn't make a valid solution by removing low-connectivity nodes,
// try a different approach: iteratively build a new solution
std::vector<int> iterativeSolution;

// Start with highest degree nodes from each solution
std::vector<std::pair<int, int>> solution1Degrees;
std::vector<std::pair<int, int>> solution2Degrees;

for (int v : solution1) {
    solution1Degrees.push_back({v, graph.getDegree(v)});
}

for (int v : solution2) {
    solution2Degrees.push_back({v, graph.getDegree(v)});
}

std::sort(solution1Degrees.begin(), solution1Degrees.end(), 
         [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
             return a.second > b.second;
         });

std::sort(solution2Degrees.begin(), solution2Degrees.end(), 
         [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
             return a.second > b.second;
         });

// Start with top nodes from both solutions
int startSize = std::min(10, (int)std::min(solution1Degrees.size(), solution2Degrees.size()));

for (int i = 0; i < startSize; i++) {
    iterativeSolution.push_back(solution1Degrees[i].first);
    
    // Add the node from solution2 if it's not already in the solution
    int candidate = solution2Degrees[i].first;
    if (std::find(iterativeSolution.begin(), iterativeSolution.end(), candidate) == iterativeSolution.end()) {
        iterativeSolution.push_back(candidate);
    }
}

// Check if we have a valid starting point
if (!isQuasiClique(iterativeSolution) || !isConnected(iterativeSolution)) {
    // If not valid, start with just the highest degree node
    iterativeSolution.clear();
    iterativeSolution.push_back(solution1Degrees[0].first);
}

// Try to add more nodes
std::unordered_set<int> candidateSet;
for (int v : solution1) candidateSet.insert(v);
for (int v : solution2) candidateSet.insert(v);

for (int v : iterativeSolution) {
    candidateSet.erase(v);
}

std::vector<int> candidates(candidateSet.begin(), candidateSet.end());

// Sort candidates by potential value to the solution
std::sort(candidates.begin(), candidates.end(), [this, &iterativeSolution](int a, int b) {
    int aConnections = countConnectionsToSolution(a, iterativeSolution);
    int bConnections = countConnectionsToSolution(b, iterativeSolution);
    return aConnections > bConnections;
});

// Try adding each candidate
for (int candidate : candidates) {
    std::vector<int> testSolution = iterativeSolution;
    testSolution.push_back(candidate);
    
    if (isQuasiClique(testSolution) && isConnected(testSolution)) {
        iterativeSolution = testSolution;
    }
}

// Return the best solution found
if (iterativeSolution.size() > std::max(solution1.size(), solution2.size())) {
    return iterativeSolution;
}

// If all attempts failed, return an empty vector
return std::vector<int>();
}
// New helper method for aggressive merging
// vector<int> TwoPhaseQuasiCliqueSolver::attemptToMerge(
//         const vector<int>& solution1, const vector<int>& solution2) const {
    
//     // Combine the solutions
//     vector<int> combined;
//     combined.reserve(solution1.size() + solution2.size());
    
//     // Add all vertices from solution1
//     combined.insert(combined.end(), solution1.begin(), solution1.end());
    
//     // Add unique vertices from solution2
//     unordered_set<int> solution1Set(solution1.begin(), solution1.end());
//     for (int v : solution2) {
//         if (solution1Set.find(v) == solution1Set.end()) {
//             combined.push_back(v);
//         }
//     }
    
//     // If the combined solution is a valid quasi-clique and connected, return it
//     if (isQuasiClique(combined) && isConnected(combined)) {
//         return combined;
//     }
    
//     // If not valid, try to make it valid by removing low-connectivity nodes
//     vector<pair<int, double>> nodeConnectivity;
//     for (int node : combined) {
//         int connections = 0;
//         for (int other : combined) {
//             if (node != other && graph.hasEdge(node, other)) {
//                 connections++;
//             }
//         }
//         double connectivity = (double)connections / (combined.size() - 1);
//         nodeConnectivity.push_back({node, connectivity});
//     }
    
//     // Sort by connectivity (ascending)
//     sort(nodeConnectivity.begin(), nodeConnectivity.end(), 
//          [](const pair<int, double>& a, const pair<int, double>& b) {
//              return a.second < b.second;
//          });
    
//     // Try removing up to 30% of the lowest-connectivity nodes
//     int maxToRemove = combined.size() * 0.3;
//     vector<int> prunedSolution = combined;
    
//     for (int i = 0; i < min(maxToRemove, (int)nodeConnectivity.size()); i++) {
//         int nodeToRemove = nodeConnectivity[i].first;
//         prunedSolution.erase(remove(prunedSolution.begin(), prunedSolution.end(), nodeToRemove), 
//                              prunedSolution.end());
        
//         if (isQuasiClique(prunedSolution) && isConnected(prunedSolution) && 
//             prunedSolution.size() > max(solution1.size(), solution2.size())) {
//             return prunedSolution;
//         }
//     }
    
//     // If we couldn't make a valid solution, return empty vector
//     return vector<int>();
// }

// vector<int> TwoPhaseQuasiCliqueSolver::findLargeQuasiClique(int numSeeds, int numThreads) {
//     // Determine number of threads to use
//     if (numThreads <= 0) {
//         numThreads = thread::hardware_concurrency();
//         if (numThreads == 0) numThreads = 1;
//     }
    
//     // // Register signal handlers for graceful termination
//     // signal(SIGINT, signalHandler);  // Ctrl+C
//     // signal(SIGTERM, signalHandler); // kill command
    
//     // Pre-compute clustering coefficients
//     precomputeClusteringCoefficients(numThreads);
    
//     if (terminationRequested) {
//         cout << "Termination requested during preprocessing. Exiting." << endl;
//         return bestSolutionOverall;
//     }
    
//     // Detect communities
//     cout << "Step 1: Detecting communities in the graph..." << endl;
//     communityDetector.detectCommunities();
    
//     if (terminationRequested) {
//         cout << "Termination requested during community detection. Exiting." << endl;
//         return bestSolutionOverall;
//     }
    
//     // Phase 1: Find multiple candidate solutions using community-aware approach
//     cout << "Step 2: Starting Phase 1 - Finding candidate solutions..." << endl;
//     phase1_findCandidateSolutions(numSeeds, numThreads);
    
//     if (terminationRequested) {
//         cout << "Termination requested during Phase 1. Returning best solution found so far." << endl;
//         return bestSolutionOverall;
//     }
    
//     // Phase 2: Refine and merge solutions
//     cout << "Step 3: Starting Phase 2 - Refining and merging solutions..." << endl;
//     phase2_refineSolutions();
    
//     if (terminationRequested) {
//         cout << "Termination requested during Phase 2. Returning best solution found so far." << endl;
//     }
    
//     return bestSolutionOverall;
// }
std::vector<int> TwoPhaseQuasiCliqueSolver::findLargeQuasiClique(int numSeeds, int numThreads) {
    // Determine number of threads to use
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
        if (numThreads == 0) numThreads = 1;
    }
    
    // Pre-compute clustering coefficients
    precomputeClusteringCoefficients(numThreads);
    
    if (terminationRequested) {
        std::cout << "Termination requested during preprocessing. Exiting." << std::endl;
        return bestSolutionOverall;
    }
    
    // Step 1: Detect communities
    std::cout << "Step 1: Detecting communities in the graph..." << std::endl;
    communityDetector.detectCommunities();
    
    if (terminationRequested) {
        std::cout << "Termination requested during community detection. Exiting." << std::endl;
        return bestSolutionOverall;
    }
    
    // Step 2: Merge communities with high connectivity
    std::cout << "Step 2: Merging highly connected communities..." << std::endl;
    // Use the lower connectivity threshold as you observed works better
    communityDetector.mergeCommunities(communityConnectivityThreshold);
    
    if (terminationRequested) {
        std::cout << "Termination requested during community merging. Exiting." << std::endl;
        return bestSolutionOverall;
    }
    
    // Step 3: Perform multi-round exploration with improved methods
    std::cout << "Step 3: Starting multi-round exploration..." << std::endl;
    const int NUM_ROUNDS = 3;  // Number of exploration rounds
    multiRoundExploration(numSeeds, NUM_ROUNDS, numThreads);
    
    if (terminationRequested) {
        std::cout << "Termination requested during exploration. Returning best solution found so far." << std::endl;
        return bestSolutionOverall;
    }
    
    // Step 4: Final phase 2 - Refine and merge all accumulated solutions
    std::cout << "Step 4: Final refinement and merging of solutions..." << std::endl;
    if (!candidateSolutions.empty()) {
        phase2_refineSolutions();
    }
    
    if (terminationRequested) {
        std::cout << "Termination requested during final refinement. Returning best solution found so far." << std::endl;
    }
    
    return bestSolutionOverall;
}

// Update the expandFromExistingSolution method to use improved expansion
std::vector<int> TwoPhaseQuasiCliqueSolver::expandFromExistingSolution(
                                const std::vector<int>& initialSolution, int numSeeds, int numThreads) {

    // Verify the initial solution is valid
    if (!initialSolution.empty()) {
        if (!isQuasiClique(initialSolution)) {
            std::cout << "Warning: Initial solution is not a valid quasi-clique." << std::endl;
        }
        if (!isConnected(initialSolution)) {
            std::cout << "Warning: Initial solution is not connected." << std::endl;
        }
    } else {
        std::cout << "Warning: Initial solution is empty. Proceeding with regular algorithm." << std::endl;
        return findLargeQuasiClique(numSeeds, numThreads);
    }

    // Start with the initial solution as our best
    bestSolutionOverall = initialSolution;

    // Determine number of threads to use
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
        if (numThreads == 0) numThreads = 1;
    }

    // Pre-compute clustering coefficients
    precomputeClusteringCoefficients(numThreads);

    if (terminationRequested) {
        std::cout << "Termination requested during preprocessing. Exiting." << std::endl;
        return bestSolutionOverall;
    }

    // Detect communities
    std::cout << "Step 1: Detecting communities in the graph..." << std::endl;
    communityDetector.detectCommunities();

    if (terminationRequested) {
        std::cout << "Termination requested during community detection. Exiting." << std::endl;
        return bestSolutionOverall;
    }

    // Merge communities with high connectivity
    std::cout << "Step 2: Merging highly connected communities..." << std::endl;
    communityDetector.mergeCommunities(communityConnectivityThreshold);

    if (terminationRequested) {
        std::cout << "Termination requested during community merging. Exiting." << std::endl;
        return bestSolutionOverall;
    }
    
    // Phase 1: Use the initial solution and expand using improved methods
    std::cout << "Phase 1: Expanding from initial solution of " << initialSolution.size() << " nodes" << std::endl;

    // First approach: Use the improved expansion directly on the initial solution
    std::cout << "Performing improved expansion on initial solution..." << std::endl;
    std::vector<int> improvedSolution = improvedExpansionMethod(initialSolution);
    
    // Update best solution if better
    if (improvedSolution.size() > bestSolutionOverall.size() && 
        isQuasiClique(improvedSolution) && isConnected(improvedSolution)) {
        bestSolutionOverall = improvedSolution;
        
        // Save progress
        std::ofstream solutionFile("solution_in_progress.txt");
        for (int v : bestSolutionOverall) {
            solutionFile << v << std::endl;
        }
        solutionFile.close();
        
        std::cout << "New best solution found: " << bestSolutionOverall.size() << " vertices" << std::endl;
    }
    
    // Add to candidate solutions for phase 2
    candidateSolutions.push_back(improvedSolution);
    
    // Second approach: Use boundary vertices of the initial solution as seeds
    std::unordered_set<int> boundaryCandidates = findBoundaryVertices(initialSolution);
    std::vector<int> boundarySeeds(boundaryCandidates.begin(), boundaryCandidates.end());

    // Sort boundary seeds by potential (using k-core and clustering)
    std::sort(boundarySeeds.begin(), boundarySeeds.end(), [this](int a, int b) {
        auto aIt = clusteringCoefficients.find(a);
        auto bIt = clusteringCoefficients.find(b);
        double aClust = (aIt != clusteringCoefficients.end()) ? aIt->second : 0.0;
        double bClust = (bIt != clusteringCoefficients.end()) ? bIt->second : 0.0;
        
        int aDegree = graph.getDegree(a);
        int bDegree = graph.getDegree(b);
        
        double aScore = 0.7 * aDegree + 0.3 * aClust * 100;
        double bScore = 0.7 * bDegree + 0.3 * bClust * 100;
        
        return aScore > bScore;
    });

    // Limit the number of boundary seeds to use
    int boundaryCount = std::min((int)boundarySeeds.size(), numSeeds);
    boundarySeeds.resize(boundaryCount);

    // Also add high k-core vertices as potential seeds
    if (boundarySeeds.size() < static_cast<size_t>(numSeeds)) {
        std::vector<int> additionalSeeds = enhancedKCoreSeedSelection(numSeeds - boundarySeeds.size());
        for (int seed : additionalSeeds) {
            // Check if seed is already in boundary seeds
            if (std::find(boundarySeeds.begin(), boundarySeeds.end(), seed) == boundarySeeds.end()) {
                boundarySeeds.push_back(seed);
                if (boundarySeeds.size() >= static_cast<size_t>(numSeeds)) break;
            }
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
            std::vector<int> solution = initialSolution;
            if (std::find(solution.begin(), solution.end(), seed) == solution.end()) {
                solution.push_back(seed);
            }
            
            // Continue expanding from here using improved method
            std::vector<int> expanded = improvedExpansionMethod(solution);
            
            // Update best solution if better
            if (expanded.size() > bestSolutionOverall.size() && 
                isQuasiClique(expanded) && isConnected(expanded)) {
                std::lock_guard<std::mutex> lock(bestSolutionMutex);
                if (expanded.size() > bestSolutionOverall.size()) {
                    bestSolutionOverall = expanded;
                    solutionFound = true;
                    
                    // Save progress
                    std::ofstream solutionFile("solution_in_progress.txt");
                    for (int v : bestSolutionOverall) {
                        solutionFile << v << std::endl;
                    }
                    solutionFile.close();
                    
                    std::cout << "New best solution found: " << bestSolutionOverall.size() << " vertices" << std::endl;
                }
            }
            
            // Add to candidate solutions for phase 2
            if (expanded.size() > initialSolution.size() && 
                isQuasiClique(expanded) && isConnected(expanded)) {
                std::lock_guard<std::mutex> lock(candidateSolutionsMutex);
                candidateSolutions.push_back(expanded);
            }
            
            completedSeeds++;
        });
    }

    // Wait for all expansions to complete
    while (completedSeeds < totalSeeds && !terminationRequested) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        std::cout << "Progress: " << completedSeeds << "/" << totalSeeds 
            << " seeds processed, candidate solutions: " << candidateSolutions.size() << std::endl;
    }

    // Sort candidate solutions by size
    std::sort(candidateSolutions.begin(), candidateSolutions.end(), 
        [](const std::vector<int>& a, const std::vector<int>& b) { return a.size() > b.size(); });

    // Keep only top solutions to limit computational complexity in phase 2
    if (candidateSolutions.size() > 300) {
        std::cout << "Currently, we have " << candidateSolutions.size() << " candidate solutions.\n";
        std::cout << "Limiting to top 300 candidate solutions for phase 2" << std::endl;
        candidateSolutions.resize(300);
    }

    // Phase 2: Refine and merge solutions
    phase2_refineSolutions();

    // Random Restart Logic 
    // Check if the best solution so far didn't improve much compared to the initial solution.
    if (bestSolutionOverall.size() <= initialSolution.size() + 5) {
        std::cout << "***************Solution not improving significantly. Trying multi-round exploration..." << std::endl;
        
        // Use the multi-round exploration to try different approaches
        multiRoundExploration(numSeeds, 2, numThreads);
        
        // Run another round of merging after multi-round exploration
        std::cout << "Running additional merge phase with exploration solutions..." << std::endl;
        phase2_refineSolutions();
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