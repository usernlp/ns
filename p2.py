#1 the General Working and Syntaxes for BA network and its metrics

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from powerlaw import Fit

# Generate Barabási-Albert network
n = 1000  # Number of nodes
m = 3     # Number of edges to attach from a new node to existing nodes
G = nx.barabasi_albert_graph(n, m)


# Basic network metrics
avg_path_length = nx.average_shortest_path_length(G)
avg_clustering = nx.average_clustering(G)
diameter = nx.diameter(G)
assortativity = nx.degree_assortativity_coefficient(G)


# Degree statistics
degrees = [d for n, d in G.degree()]
min_degree = min(degrees)
max_degree = max(degrees)
avg_degree = np.mean(degrees)


# Power-law fit for degree distribution
fit = Fit(degrees, verbose=False)
power_law_alpha = fit.power_law.alpha
power_law_xmin = fit.power_law.xmin


# Degree distribution histogram
hist, bin_edges = np.histogram(degrees, bins=50, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
valid = hist > 0  # Filter for log-log plot


# Centrality measures
betweenness = nx.betweenness_centrality(G)
top_hubs = sorted(betweenness, key=betweenness.get, reverse=True)[:5]


# Percolation analysis (robustness to targeted node removal)
def percolation(G, fraction, targeted=False):
    G_copy = G.copy()
    if targeted:
        nodes_to_remove = sorted(G_copy.degree, key=lambda x: x[1], reverse=True)
    else:
        nodes_to_remove = list(G_copy.nodes())
        np.random.shuffle(nodes_to_remove)
    num_remove = int(fraction * len(G_copy))
    G_copy.remove_nodes_from(nodes_to_remove[:num_remove])
    components = list(nx.connected_components(G_copy))
    return len(max(components, key=len, default=set()))  # Size of largest component


largest_component_random = percolation(G, 0.2, targeted=False)
largest_component_targeted = percolation(G, 0.2, targeted=True)


# Community detection
from networkx.algorithms.community import greedy_modularity_communities
communities = greedy_modularity_communities(G)
num_communities = len(communities)
modularity = nx.community.modularity(G, communities)


# Print results
print("=== Barabási-Albert Network Analysis ===")
print(f"Average Shortest Path Length: {avg_path_length:.3f}")
print(f"Average Clustering Coefficient: {avg_clustering:.3f}")
print(f"Diameter: {diameter}")
print(f"Degree Assortativity Coefficient: {assortativity:.3f}")
print(f"Minimum Degree: {min_degree}")
print(f"Maximum Degree: {max_degree}")
print(f"Average Degree: {avg_degree:.3f}")
print(f"Power-law Exponent: {power_law_alpha:.3f}")
print(f"Power-law xmin: {power_law_xmin}")
print(f"Top 5 Hubs by Betweenness: {top_hubs}")
print(f"Largest Component after 20% Random Removal: {largest_component_random}")
print(f"Largest Component after 20% Targeted Removal: {largest_component_targeted}")
print(f"Number of Communities: {num_communities}")
print(f"Modularity: {modularity:.3f}")


# Plot degree distribution
plt.figure(figsize=(8, 6))
plt.loglog(bin_centers[valid], hist[valid], 'b.', label='Degree distribution')
plt.xlabel('Degree (k)')
plt.ylabel('P(k)')
plt.title('Degree Distribution of Barabási-Albert Network')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()


# Plot network visualization
plt.figure(figsize=(8, 8))
pos = nx.spring_layout(G)
node_sizes = [d * 10 for n, d in G.degree()]  # Scale for visibility
nx.draw(G, pos, node_size=node_sizes, node_color='skyblue', edge_color='gray', alpha=0.6)
plt.title('Barabási-Albert Network Visualization')
plt.show()


import networkx as nx
from networkx_robustness import networkx_robustness as netrob
import matplotlib.pyplot as plt

G = nx.barabasi_albert_graph(50, 2)
initial, frac, apl = netrob.simulate_random_attack(G, attack_fraction=0.2)

plt.figure(figsize=(10, 6))
plt.plot(frac, apl, marker='o', linestyle='-', color='red')
plt.title('Network Robustness under Random Attack')
plt.xlabel('Fraction of Nodes Removed')
plt.ylabel('Average Path Length of Largest Component')
plt.grid(True)
plt.show()

molloy_reed = netrob.molloy_reed(G)
molloy_reed

critical_threshold = netrob.critical_threshold(G)
print(critical_threshold)


initial, frac_random, apl_random = netrob.simulate_random_attack(G, attack_fraction=0.2)
initial, frac_degree, apl_degree = netrob.simulate_degree_attack(G, attack_fraction=0.1, weight=None)
initial, frac_betweenness, apl_betweenness = netrob.simulate_betweenness_attack(G, attack_fraction=0.1, weight=None, normalized=True, k=None, seed=None, endpoints=False)
initial, frac_closeness, apl_closeness = netrob.simulate_closeness_attack(G, attack_fraction=0.1, weight=None, u=None, wf_improved=True)
initial, frac_eigenvector, apl_eigenvector = netrob.simulate_eigenvector_attack(G, attack_fraction=0.1, weight=None, tol=1e-06, max_iter=100, nstart=None)

plt.figure(figsize=(12, 8))


plt.plot(frac_random, apl_random, marker='o', linestyle='-', color='blue', label='Random Attack')
plt.plot(frac_degree, apl_degree, marker='x', linestyle='--', color='green', label='Degree Attack')
plt.plot(frac_betweenness, apl_betweenness, marker='s', linestyle='-.', color='purple', label='Betweenness Attack')
plt.plot(frac_closeness, apl_closeness, marker='^', linestyle=':', color='orange', label='Closeness Attack')
plt.plot(frac_eigenvector, apl_eigenvector, marker='d', linestyle='-', color='red', label='Eigenvector Attack')


plt.title('Network Robustness under Different Attack Strategies')
plt.xlabel('Fraction of Nodes Removed')
plt.ylabel('Average Path Length of Largest Component')
plt.grid(True)
plt.legend()
plt.show()


import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 1000
steps = 20

# Scale-free network (Barabási–Albert model)
G_config = nx.barabasi_albert_graph(N, 3)

# Hierarchical-like graph
def generate_hierarchical(level):
    G = nx.complete_graph(5)
    for _ in range(level-1):
        H = nx.disjoint_union_all([G]*5)
        centers = [list(c)[0] for c in nx.connected_components(H)]
        for c in centers[1:]:
            H.add_edge(centers[0], c)   # connect central nodes
        G = H
    return G

G_hier = generate_hierarchical(3)

# Attack simulation
def simulate_attack(G, metric, steps=20):
    N = len(G)
    fracs = np.linspace(0, 1, steps)
    sizes = []
    if metric == "degree":
        values = dict(G.degree())
    else:
        values = nx.clustering(G)
    nodes_sorted = sorted(values, key=values.get, reverse=True)

    for f in fracs:
        k = int(f*N)
        G_copy = G.copy()
        G_copy.remove_nodes_from(nodes_sorted[:k])
        if len(G_copy) == 0:
            sizes.append(0)
        else:
            largest_cc = max(nx.connected_components(G_copy), key=len)
            sizes.append(len(largest_cc)/N)
    return fracs, sizes

# Run simulations
config_f_deg, config_s_deg = simulate_attack(G_config, "degree", steps)
config_f_clus, config_s_clus = simulate_attack(G_config, "clustering", steps)
hier_f_deg, hier_s_deg = simulate_attack(G_hier, "degree", steps)
hier_f_clus, hier_s_clus = simulate_attack(G_hier, "clustering", steps)

# Plot
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(config_f_deg, config_s_deg, 'r-', label="Degree Attack")
plt.plot(config_f_clus, config_s_clus, 'b-', label="Clustering Attack")
plt.title("Scale-Free (BA Model)")
plt.xlabel("Fraction Removed")
plt.ylabel("Giant Component Size")
plt.legend(); plt.grid(True)

plt.subplot(1,2,2)
plt.plot(hier_f_deg, hier_s_deg, 'r-', label="Degree Attack")
plt.plot(hier_f_clus, hier_s_clus, 'b-', label="Clustering Attack")
plt.title("Hierarchical Model")
plt.xlabel("Fraction Removed")
plt.ylabel("Giant Component Size")
plt.legend(); plt.grid(True)

plt.tight_layout(); plt.show()
