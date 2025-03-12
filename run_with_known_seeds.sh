#!/bin/bash
# Run the two-phase algorithm using seeds from the known solution

# Colors for output formatting
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default parameters
KNOWN_SOLUTION="data/known_solution.txt"
FLYWIRE_EDGES="data/flywire_edges_converted.txt"
ID_MAPPING="data/id_mapping.csv"
SEED_FILE="known_solution_seeds.txt"
NUM_SEEDS=1000
KNOWN_SEEDS=0  # Will be set based on file content

# Auto-detect number of threads
AVAILABLE_THREADS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
NUM_THREADS=$AVAILABLE_THREADS

# Function to print section headers
print_section() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
}

# Function to print progress messages
print_progress() {
    echo -e "${GREEN}[PROGRESS]${NC} $1"
}

# Function to print warnings
print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to print errors
print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --known-solution)
        KNOWN_SOLUTION="$2"
        shift
        shift
        ;;
        --edges)
        FLYWIRE_EDGES="$2"
        shift
        shift
        ;;
        --mapping)
        ID_MAPPING="$2"
        shift
        shift
        ;;
        --seeds)
        NUM_SEEDS="$2"
        shift
        shift
        ;;
        --threads)
        NUM_THREADS="$2"
        shift
        shift
        ;;
        *)
        print_error "Unknown option: $1"
        echo "Usage: $0 [--known-solution FILE] [--edges FILE] [--mapping FILE] [--seeds NUM] [--threads NUM]"
        exit 1
        ;;
    esac
done

# Check if the known solution file exists
print_section "CHECKING FILES"
if [ ! -f "$KNOWN_SOLUTION" ]; then
    print_error "Known solution file not found: $KNOWN_SOLUTION"
    exit 1
fi

if [ ! -f "$FLYWIRE_EDGES" ]; then
    print_error "Edge file not found: $FLYWIRE_EDGES"
    exit 1
fi

if [ ! -f "$ID_MAPPING" ]; then
    print_warning "ID mapping file not found: $ID_MAPPING"
    print_warning "Will try to use original IDs from the known solution"
fi
print_progress "All required files found"

# Step 1: Generate seed file from known solution
print_section "GENERATING SEED FILE FROM KNOWN SOLUTION"
python3 scripts/generate_seed_file.py "$KNOWN_SOLUTION" "$SEED_FILE" "$ID_MAPPING"
if [ ! -f "$SEED_FILE" ]; then
    print_error "Failed to generate seed file"
    exit 1
fi

# Count seed file lines and use all available seeds
KNOWN_SEEDS=$(wc -l < "$SEED_FILE")
print_progress "Generated seed file with $KNOWN_SEEDS seeds"
print_progress "Using all $KNOWN_SEEDS seeds from the known solution"

# Calculate normal seeds
NORMAL_SEEDS=$((NUM_SEEDS - KNOWN_SEEDS))
if [ $NORMAL_SEEDS -lt 0 ]; then
    NORMAL_SEEDS=0
    print_warning "Known seeds exceed total seeds, using only known seeds"
fi

print_progress "Configuration:"
echo "- Known solution file: $KNOWN_SOLUTION"
echo "- Edge file: $FLYWIRE_EDGES"
echo "- ID mapping file: $ID_MAPPING"
echo "- Total seeds: $NUM_SEEDS ($NORMAL_SEEDS normal + $KNOWN_SEEDS from known solution)"
echo "- Threads: $NUM_THREADS"
echo

# Step 2: Create a modified version of the two_phase_solver.cpp that prioritizes these seeds
print_section "CREATING MODIFIED SOLVER"
MODIFIED_SOLVER="src/two_phase_solver_with_known_seeds.cpp"

# Copy the original solver
cp src/two_phase_solver.cpp "$MODIFIED_SOLVER"
print_progress "Created modified solver: $MODIFIED_SOLVER"

# Step 3: Modify the selectSeedsWithCommunityAwareness function to include known seeds
print_progress "Modifying the seed selection function..."
cat > seed_selection_func.tmp << 'EOF'
    // Sort seed vertices based on community structure, but prioritize known seeds
    vector<int> selectSeedsWithCommunityAwareness(const vector<int>& potentialSeeds, int numSeeds) {
        if (numSeeds >= (int)potentialSeeds.size()) {
            return potentialSeeds;
        }
        
        // First, try to load known seeds from file
        vector<int> knownSeeds;
        ifstream seedFile("known_solution_seeds.txt");
        if (seedFile.is_open()) {
            int seed;
            while (seedFile >> seed) {
                knownSeeds.push_back(seed);
            }
            seedFile.close();
            cout << "Loaded " << knownSeeds.size() << " seeds from known solution" << endl;
        }
        
        // Calculate how many regular seeds we need
        int regularSeedsNeeded = numSeeds - knownSeeds.size();
        
        // If we have enough known seeds, just return those
        if (regularSeedsNeeded <= 0) {
            return knownSeeds;
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
        vector<int> selectedSeeds = knownSeeds; // Start with known seeds
        selectedSeeds.reserve(numSeeds);
        
        // First, get boundary vertices (vertices connecting different communities)
        vector<int> boundaryVertices = communityDetector.findBoundaryVertices();
        
        // Sort boundary vertices by potential
        sort(boundaryVertices.begin(), boundaryVertices.end(), [this](int a, int b) {
            return calculateVertexPotential(a) > calculateVertexPotential(b);
        });
        
        // Take top 20% of remaining seeds from boundary vertices
        int boundaryCount = min(regularSeedsNeeded / 5, (int)boundaryVertices.size());
        for (int i = 0; i < boundaryCount; i++) {
            selectedSeeds.push_back(boundaryVertices[i]);
        }
        
        // Allocate remaining seeds to communities proportionally
        int remainingSeeds = regularSeedsNeeded - boundaryCount;
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
EOF

# Replace the function in the file
sed -i '/vector<int> selectSeedsWithCommunityAwareness(const vector<int>& potentialSeeds, int numSeeds)/,/return selectedSeeds;/ {
    /vector<int> selectSeedsWithCommunityAwareness(const vector<int>& potentialSeeds, int numSeeds)/,/return selectedSeeds;/ d
}' "$MODIFIED_SOLVER"

# Place the new function where the old one was
sed -i '/double calculateCommunityAwareVertexPotential(int v, int targetCommunity) const {/a \
'"$(cat seed_selection_func.tmp)"'' "$MODIFIED_SOLVER"

rm seed_selection_func.tmp

# Step 4: Modify phase1_findCandidateSolutions to give priority to known seeds
print_progress "Modifying the solution finding function..."
cat > find_solutions_func.tmp << 'EOF'
        for (size_t seedIdx = 0; seedIdx < seeds.size(); seedIdx++) {
            int seed = seeds[seedIdx];
            int community = communityDetector.getCommunity(seed);
            
            // Give higher priority to seeds from known solution
            bool isKnownSeed = seedIdx < knownSeeds.size();
            
            pool.enqueue([this, seed, seedIdx, community, isKnownSeed, knownSeeds]() {
                if (terminationRequested) return;
                
                vector<int> solution;
                if (isKnownSeed) {
                    cout << "Processing known seed " << seedIdx + 1 << "/" << knownSeeds.size() << endl;
                    // For known seeds, use a different growing strategy with more aggressive expansion
                    solution = findQuasiCliqueFromSeedAggressive(seed, seedIdx, community);
                } else {
                    cout << "Processing regular seed " << (seedIdx + 1 - knownSeeds.size()) << "/" << (totalSeeds - knownSeeds.size()) << endl;
                    solution = findQuasiCliqueFromSeed(seed, seedIdx, community);
                }
                
                // Update best solution if better
                if (solution.size() > 5 && isQuasiClique(solution)) {
                    lock_guard<mutex> lock(bestSolutionMutex);
                    if (solution.size() > bestSolutionOverall.size()) {
                        bestSolutionOverall = solution;
                        solutionFound = true;
                        cout << "New best solution found: " << bestSolutionOverall.size() << " vertices" << endl;
                    }
                }
            });
EOF

# Find the original function in the file and replace it
sed -i '/for (size_t seedIdx = 0; seedIdx < seeds.size(); seedIdx++) {/,/});/ {
    /for (size_t seedIdx = 0; seedIdx < seeds.size(); seedIdx++) {/,/});/ d
}' "$MODIFIED_SOLVER"

# Insert the new function where the old one was
sed -i '/vector<int> seeds = selectSeedsWithCommunityAwareness(vertices, numSeeds);/a \
        vector<int> knownSeeds;\
        ifstream seedFile("known_solution_seeds.txt");\
        if (seedFile.is_open()) {\
            int seed;\
            while (seedFile >> seed) {\
                knownSeeds.push_back(seed);\
            }\
            seedFile.close();\
        }\
        cout << "Using " << knownSeeds.size() << " known seeds and " << (seeds.size() - knownSeeds.size()) << " regular seeds" << endl;\
        \
'"$(cat find_solutions_func.tmp)"'' "$MODIFIED_SOLVER"

rm find_solutions_func.tmp

# Step 5: Add the aggressive expansion function
print_progress "Adding aggressive expansion function..."
cat > aggressive_func.tmp << 'EOF'
    // Find a quasi-clique using aggressive expansion (for known seeds)
    vector<int> findQuasiCliqueFromSeedAggressive(int seed, int seedIdx, int targetCommunity = -1) {
        cout << "  Starting from known seed: " << seed 
             << " (degree: " << graph.getDegree(seed) 
             << ", clustering: " << clusteringCoefficients[seed];
        
        if (targetCommunity >= 0) {
            cout << ", community: " << communityDetector.getCommunity(seed) << ")";
        } else {
            cout << ")";
        }
        cout << endl;
        
        vector<int> solution = {seed};
        
        // Expansion phase - aggressive version
        int iteration = 0;
        while (!terminationRequested) {
            iteration++;
            vector<pair<int, double>> candidates;
            
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
                
                // Enhanced score with clustering coefficient and community awareness
                double directRatio = static_cast<double>(connections) / solution.size();
                double candidateClustering = clusteringCoefficients[candidate];
                
                // More aggressive alpha parameter that decreases slower
                double alpha = max(0.6, 0.95 - 0.005 * solution.size());
                double score = alpha * directRatio + (1 - alpha) * candidateClustering;
                
                // Community awareness with stronger boost
                if (targetCommunity >= 0 && communityDetector.getCommunity(candidate) == targetCommunity) {
                    score *= 1.2; // Stronger boost (1.2 instead of 1.1)
                }
                
                // Allow more candidates for evaluation
                vector<int> newSolution = solution;
                newSolution.push_back(candidate);
                
                if (isQuasiClique(newSolution)) {
                    candidates.push_back({candidate, score});
                }
            }
            
            // If no suitable candidates found, break
            if (candidates.empty()) {
                cout << "    No more candidates found after " << iteration << " iterations" << endl;
                break;
            }
            
            // Sort candidates by score
            sort(candidates.begin(), candidates.end(), 
                 [](const pair<int, double>& a, const pair<int, double>& b) {
                     return a.second > b.second;
                 });
            
            // Try to add multiple candidates in each iteration if possible
            bool added = false;
            for (size_t i = 0; i < min(size_t(3), candidates.size()); i++) {
                int candidate = candidates[i].first;
                
                // Check if adding this candidate maintains quasi-clique property
                vector<int> newSolution = solution;
                newSolution.push_back(candidate);
                
                if (isQuasiClique(newSolution)) {
                    solution.push_back(candidate);
                    added = true;
                }
            }
            
            if (!added && !candidates.empty()) {
                // If we couldn't add multiple, at least add the best one
                solution.push_back(candidates[0].first);
            }
            
            // Progress reporting
            if (iteration % 10 == 0) {
                cout << "    Iteration " << iteration << ": solution size = " << solution.size() << endl;
            }
            
            // Check if solution is better than current best, and if so, save it immediately
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
        
        // Add to candidate solutions
        if (solution.size() > 5 && isQuasiClique(solution)) {
            lock_guard<mutex> lock(candidateSolutionsMutex);
            candidateSolutions.push_back(solution);
        }
        
        return solution;
    }
EOF

# Insert the aggressive function before saveSolution function
sed -i '/bool saveSolution(const vector<int>& solution, const string& filename = "solution.txt")/i \
'"$(cat aggressive_func.tmp)"'' "$MODIFIED_SOLVER"

rm aggressive_func.tmp

# Step 6: Modify the phase2_refineSolutions to use more aggressive merging
print_progress "Modifying the solution merging function..."
sed -i 's/if (similarity >= 0.1 && similarity <= 0.8) {/if (similarity >= 0.05 && similarity <= 0.9) {/' "$MODIFIED_SOLVER"
sed -i 's/mergeIterations < 5/mergeIterations < 10/' "$MODIFIED_SOLVER"

# Step 7: Compile the modified solver
print_section "COMPILING MODIFIED SOLVER"
g++ -std=c++17 -Wall -Wextra -O3 -pthread "$MODIFIED_SOLVER" -o build/two_phase_solver_with_known_seeds

if [ $? -ne 0 ]; then
    print_error "Failed to compile modified solver"
    exit 1
fi
print_progress "Successfully compiled modified solver"

# Step 8: Run the algorithm
print_section "RUNNING ALGORITHM WITH KNOWN SEEDS"
print_progress "Running algorithm with $NUM_SEEDS total seeds ($KNOWN_SEEDS from known solution)"
print_progress "Using $NUM_THREADS threads"

./build/two_phase_solver_with_known_seeds "$FLYWIRE_EDGES" "$NUM_SEEDS" "$NUM_THREADS"

if [ $? -ne 0 ]; then
    print_error "Algorithm failed"
    exit 1
fi

print_section "CHECKING RESULTS"
if [ -f "solution.txt" ]; then
    SOLUTION_SIZE=$(wc -l < solution.txt)
    print_progress "Found solution with $SOLUTION_SIZE vertices"
    
    # Compare with known solution
    KNOWN_SIZE=$(wc -l < "$KNOWN_SOLUTION")
    if [ "$SOLUTION_SIZE" -gt "$KNOWN_SIZE" ]; then
        print_progress "Found a larger solution than the known solution ($SOLUTION_SIZE > $KNOWN_SIZE)!"
    elif [ "$SOLUTION_SIZE" -eq "$KNOWN_SIZE" ]; then
        print_progress "Found a solution with the same size as the known solution ($SOLUTION_SIZE)."
    else
        print_warning "Found a smaller solution than the known solution ($SOLUTION_SIZE < $KNOWN_SIZE)."
    fi
else
    print_error "No solution file found"
fi

print_section "COMPLETED"