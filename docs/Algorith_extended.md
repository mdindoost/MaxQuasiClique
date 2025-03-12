Two-Phase Community-Aware Algorithm for Finding Large Quasi-Cliques in Neural Connectome Networks
Abstract
Finding large quasi-cliques in complex networks presents significant computational challenges due to the NP-hard nature of the problem. This paper introduces a novel two-phase algorithm that combines community detection, parallel seed exploration, and strategic solution merging to efficiently identify large quasi-cliques in neuronal connectome graphs. By leveraging network structure through community analysis and employing adaptive growth strategies, our method overcomes limitations of traditional greedy approaches, providing superior results for dense subgraph identification in large-scale biological networks.
1. Introduction and Problem Definition
A quasi-clique in an undirected graph G = (V, E) is defined as a subgraph on k vertices that contains strictly more than (k choose 2)/2 edges - in other words, a subgraph where more than half of all possible connections exist. The problem of finding the largest such subgraph is of significant interest in neuronal connectome analysis, as quasi-cliques may represent functional neural circuits or structural motifs with important biological roles.
Traditional approaches to this problem often employ greedy vertex addition but frequently become trapped in local optima, especially in complex networks with community structure like neuronal connectomes. Our algorithm addresses these limitations through a sophisticated two-phase approach that systematically explores and refines multiple candidate solutions.
2. Two-Phase Quasi-Clique Algorithm
Our algorithm comprises two main phases: (1) generation of multiple candidate solutions through community-aware seed exploration and (2) refinement through strategic solution merging. Both phases incorporate network structural properties to guide the search process.
2.1 Preprocessing and Initialization
Before the main phases, two essential preprocessing steps occur:

Clustering Coefficient Calculation: For each vertex v, we compute its local clustering coefficient:
CopyC(v) = (2 × triangles_containing_v) / (degree(v) × (degree(v) - 1))
This metric quantifies how close v's neighbors are to forming a complete graph, providing a measure of local density.
Community Detection: We employ the Louvain method to identify the natural community structure of the graph. The method iteratively optimizes modularity through:

Local optimization: Moving vertices between communities to increase modularity
Community consolidation: Aggregating vertices in the same community

This process provides essential structural insights that guide our seed selection and expansion strategies.

2.2 Phase 1: Multiple Solution Generation with Community Awareness
The first phase generates multiple candidate quasi-cliques through parallel exploration from strategically selected seed vertices:
2.2.1 Strategic Seed Selection
Rather than selecting seeds arbitrarily, we employ three complementary strategies:

Degree-Based Selection: We prioritize high-degree vertices, as they have greater potential for forming dense subgraphs:
Copyseeds = sort(vertices, by=degree, descending)[:numSeeds]

Community-Balanced Selection: We allocate seeds across communities proportionally to their size, ensuring broad exploration:
Copyseeds_per_community[c] = community_size[c] * numSeeds / total_vertices

Boundary Vertex Inclusion: We include vertices that connect different communities, as they can bridge dense regions:
Copyboundary_seeds = {v | ∃ neighbors u,w of v where community(u) ≠ community(w)}


This multi-strategy approach ensures diverse exploration across the graph structure.
2.2.2 Adaptive Solution Expansion
From each seed, we grow a solution through an enhanced greedy expansion process:

Solution Initialization: Each solution S begins as a singleton set containing one seed vertex.
Boundary Identification: At each iteration, we identify the boundary vertices - those adjacent to at least one vertex in S but not in S themselves:
Copyboundary(S) = {v ∉ S | ∃u ∈ S such that (u,v) ∈ E}

Neighbor-Prioritized Candidate Evaluation: We prioritize neighbors of recently added vertices:

Maintain a sliding window R of the 10 most recently added vertices
First evaluate candidates that connect to vertices in R
Only if no suitable candidates are found, evaluate other boundary vertices


Multi-Factor Scoring Function: For each candidate v, we calculate a composite score:
Copyscore(v) = α × (connections_to_S / |S|) + (1-α) × clustering(v) + β × (connections_to_R / |R|)
where:

The first term measures direct connectivity to the current solution
The second term incorporates local density via clustering coefficient
The third term adds a bonus for connections to recently added vertices
α adapts as the solution grows: α = max(0.5, 0.95 - 0.005 × |S|)
β is a constant (typically 0.2) weighting the recency bonus


Community Awareness: If v belongs to the same community as the seed, its score receives a 10% boost, encouraging community-cohesive growth.
Valid Addition Verification: Before adding a vertex, we verify:

Quasi-clique property: The new subgraph must have >50% of possible edges
Connectivity: The new subgraph must remain connected


Iterative Growth: We add the highest-scoring valid candidate and repeat until no suitable candidates remain.

Each expansion runs in parallel using a thread pool, allowing efficient exploration of multiple starting points.
2.3 Phase 2: Solution Refinement and Merging
The second phase enhances the candidate solutions through a strategic merging process:

Solution Ranking: We sort candidate solutions by size in descending order and select the top 100 for refinement.
Similarity-Based Filtering: For each pair of solutions (S₁, S₂), we calculate their Jaccard similarity:
CopyJ(S₁, S₂) = |S₁ ∩ S₂| / |S₁ ∪ S₂|
We only consider merging solutions with similarity between 0.05 and 0.9, balancing between merging dissimilar solutions (which likely won't form a valid quasi-clique) and nearly identical ones (which would yield minimal gains).
Merge Validation: For promising pairs, we verify if their union forms a valid quasi-clique and is connected.
Iterative Improvement: We continue merging solutions until no further improvements are possible or after 10 iterations. After each iteration:

New merged solutions are added to the candidate pool
Solutions are re-sorted by size
The overall best solution is updated if improved



This merging phase effectively combines complementary solutions, escaping local optima that might trap individual expansions.
2.4 Continuity and Incremental Improvement
A key capability of our algorithm is the ability to start from an existing solution rather than from scratch:

Solution Initialization: When provided with an initial solution S₀, we verify its validity and use it as a starting point.
Boundary-Based Seed Selection: We identify the boundary vertices of S₀ and select the highest-degree ones as seeds.
Parallel Expansion: Each seed launches an independent expansion starting from S₀ ∪ {seed}.
Merging and Refinement: The resulting expanded solutions enter Phase 2 for further refinement.

This allows incremental improvement of previously found solutions, building upon past results rather than restarting the search process.
3. Implementation and Optimizations
Several implementation details are crucial to the algorithm's performance:

Parallel Processing: Both clustering coefficient computation and seed exploration use multi-threading via a custom thread pool implementation.
Adaptive Data Structures: The implementation switches between direct array traversal and set-based operations based on solution size to optimize performance.
Incremental Property Verification: Edge counting and connectivity checks are optimized through incremental updates and early termination.
Progressive Checkpointing: The best solution found is continuously saved to disk, providing resilience against interruptions.
Memory-Efficient Graph Representation: The graph uses compressed adjacency lists with optional vertex ID remapping for non-contiguous IDs.

4. Theoretical Analysis
The time complexity of the algorithm can be analyzed as follows:

Preprocessing:

Clustering coefficient computation: O(|V|·d²) where d is the average degree
Louvain community detection: O(|E|) per pass, typically with a small constant number of passes


Phase 1: For each of the k seeds:

Boundary identification: O(|S|·d) where |S| is the solution size
Candidate evaluation: O(|∂S|·d) where |∂S| is the boundary size
Total worst-case: O(k·|V|²·d)


Phase 2:

Solution pair evaluation: O(c²·|V|) where c is the number of candidate solutions
Merge validation: O(c²·|V|²)



The algorithm balances exploration breadth (through multiple seeds) with exploitation depth (through solution merging), enabling it to effectively navigate the search space of potential quasi-cliques.
5. Limitations and Future Work
While effective, our algorithm has limitations that suggest directions for future research:

Parameter Sensitivity: The performance depends on parameters like the number of seeds and the similarity thresholds for merging.
Community Detection Quality: The effectiveness of the community-aware aspects depends on the quality of the detected communities.
Scalability Challenges: For extremely large graphs (millions of vertices), the quadratic components in solution verification could become prohibitive.

Future work could explore alternative community detection methods, adaptive parameter tuning based on graph properties, and approximation techniques for quasi-clique verification in very large graphs.
6. Conclusion
The Two-Phase Community-Aware Algorithm provides an effective approach to finding large quasi-cliques in complex networks, particularly neural connectomes. By leveraging community structure, employing adaptive expansion strategies, and intelligently merging solutions, it overcomes the limitations of traditional greedy methods. The ability to build upon previous solutions through incremental improvement further enhances its utility for ongoing analysis of neural connectivity patterns.