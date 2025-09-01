import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import norm, uniform


def compute_metrics(G, name):
    # Average clustering coefficient
    avg_clustering = nx.average_clustering(G)
    
    # Average shortest path length (for the largest connected component)
    if nx.is_directed(G):
        G_cc = max(nx.strongly_connected_components(G), key=len)
        G_sub = G.subgraph(G_cc).copy()
    else:
        G_cc = max(nx.connected_components(G), key=len)
        G_sub = G.subgraph(G_cc).copy()
    avg_path_length = nx.average_shortest_path_length(G_sub)
    
    # Average degree
    degrees = [d for n, d in G.degree()]
    avg_degree = np.mean(degrees)
    
    print(f"\nMetrics for {name}:")
    print(f"Average Clustering Coefficient: {avg_clustering:.4f}")
    print(f"Average Shortest Path Length: {avg_path_length:.4f}")
    print(f"Average Degree: {avg_degree:.4f}")
    return avg_clustering, avg_path_length, avg_degree, degrees

'''#additional metrics
# 1. Centralities
deg_cent = nx.degree_centrality(G)
bet_cent = nx.betweenness_centrality(G)
close_cent = nx.closeness_centrality(G)
eig_cent = nx.eigenvector_centrality(G)

# 2. Show top nodes by degree
sorted_deg = sorted(deg_cent.items(), key=lambda x: x[1], reverse=True)[:5]
print("Top 5 nodes by degree centrality:", sorted_deg)

# 3. Connectivity
components = sorted(nx.connected_components(G), key=len, reverse=True)
GCC = G.subgraph(components[0])
print(f"Connected components: {len(components)}")
print(f"GCC size: {GCC.number_of_nodes()} nodes")'''

def visualize_network(G, name, degrees):
    # Compute clustering values
    clustering_values = list(nx.clustering(G).values())
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # --- 1. Network layout ---
    pos = nx.spring_layout(G, seed=42)  # Consistent layout
    nx.draw(G, pos, node_size=50, node_color='skyblue', edge_color='gray', 
            with_labels=False, ax=axes[0])
    axes[0].set_title(f"{name} Layout")
    


    # --- 2. Degree distribution with fits ---
    axes[1].hist(degrees, bins=20, density=True, color='salmon',
                edgecolor='black', alpha=0.6, label="Observed Degrees")

    # Normal fit
    mu, sigma = np.mean(degrees), np.std(degrees)
    x = np.linspace(min(degrees), max(degrees), 200)
    axes[1].plot(x, norm.pdf(x, mu, sigma), 'b-', lw=2, label="Normal fit")

    # Uniform fit
    a, b = min(degrees), max(degrees)
    axes[1].plot(x, uniform.pdf(x, a, b-a), 'g--', lw=2, label="Uniform fit")

    # Power-law fit (using MLE for exponent)
    kmin = max(1, min(degrees))  # avoid zero
    degrees_pl = [k for k in degrees if k >= kmin]
    n = len(degrees_pl)
    alpha = 1 + n / sum(np.log(k / (kmin - 0.5)) for k in degrees_pl)

    # Normalized power-law PDF over observed range
    x_pl = np.linspace(kmin, max(degrees), 200)
    C = (alpha - 1) * kmin**(alpha - 1)   # normalization constant
    pdf_pl = C * (x_pl**(-alpha))

    axes[1].plot(x_pl, pdf_pl, 'r-.', lw=2, label=f"Power-law fit (α={alpha:.2f})")

    # Labels
    axes[1].set_title(f"{name} Degree Distribution")
    axes[1].set_xlabel("Degree")
    axes[1].set_ylabel("Probability Density")
    axes[1].legend()

    
    # --- 3. Clustering coefficient distribution ---
    axes[2].hist(clustering_values, bins=20, density=True, color='purple', 
                 edgecolor='black', alpha=0.7)
    axes[2].set_title(f"{name} Clustering Coefficient Distribution")
    axes[2].set_xlabel("Clustering Coefficient")
    axes[2].set_ylabel("Probability Density")
    
    # Final layout
    plt.tight_layout()
    plt.show()


#Edge List
G = nx.read_edgelist("graph_edges.txt")
avg_clust, avg_path, avg_deg, degrees_data = compute_metrics(G, "Input Network")
visualize_network(G, "Input Network", degrees_data)

#Adjacency List
G = nx.read_adjlist("graph_adjlist.txt")
avg_clust, avg_path, avg_deg, degrees_data = compute_metrics(G, "Input Network")
visualize_network(G, "Input Network", degrees_data)

#Adjacency Matrix
A = np.loadtxt("adj_matrix.txt", dtype=int)
G = nx.from_numpy_array(A)
avg_clust, avg_path, avg_deg, degrees_data = compute_metrics(G, "Input Network")
visualize_network(G, "Input Network", degrees_data)

#GML
G = nx.read_gml("network.gml")
avg_clust, avg_path, avg_deg, degrees_data = compute_metrics(G, "Input Network")
visualize_network(G, "Input Network", degrees_data)
print(f"Graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
print("Nodes:", list(G.nodes()))
print("Edges:", list(G.edges()))

#number of nodes
n = G.number_of_nodes()
#number of edges
m = G.number_of_edges()
# Estimate edge probability for Erdős-Rényi
p = (2 * m) / (n * (n - 1)) if not nx.is_directed(G) else m / (n * (n - 1))

#or
n = 10
p = 0.3
m = 7

# 1. Erdős-Rényi Graph
G = nx.erdos_renyi_graph(n, p)
avg_clust_er, avg_path_er, avg_deg_er, degrees_er = compute_metrics(G, "Erdős-Rényi Graph")
visualize_network(G, "Erdős-Rényi Graph", degrees_er)

# 2. Random Graph, n->nodes, m->edges
G = nx.gnm_random_graph(n, m)
avg_clust_er, avg_path_er, avg_deg_er, degrees_er = compute_metrics(G, "Random Graph")
visualize_network(G, "Random Graph", degrees_er)

# 3. Barabási–Albert (Scale-Free Network)
G = nx.barabasi_albert_graph(n, m)
#G = nx.scale_free_graph(n)
avg_clust_er, avg_path_er, avg_deg_er, degrees_er = compute_metrics(G, "Scale Free Network")
visualize_network(G, "Scale Free Network", degrees_er)

# 4. Watts–Strogatz (Small-World Network)
#k is the number of neighbors each node is connected to in the initial ring lattice.
k = 4
G = nx.watts_strogatz_graph(n, k, p)
avg_clust_er, avg_path_er, avg_deg_er, degrees_er = compute_metrics(G, "Watts–Strogatz")
visualize_network(G, "Watts–Strogatz", degrees_er)

#critical threshold
def critical_threshold(G):
    degrees = np.array([d for n, d in G.degree()])
    k_avg = degrees.mean()
    k2_avg = (degrees**2).mean()
    return k_avg / (k2_avg - k_avg)

pc_er = critical_threshold(G)
print("Critical Threshold : ", pc_er)





#pseudocode -> Similar to Wattz schtrogat
N = 100
m = 2
p_rewire = 0.1

G = nx.cycle_graph(5)


for new_node in range(5, N):
    G.add_node(new_node)

    # Compute degrees and total degree
    degrees = np.array([G.degree(n) for n in G.nodes()])
    total_deg = degrees.sum()

    # Preferential attachment: select m nodes proportional to degree
    targets = set()
    while len(targets) < m:
        rand_node = np.random.choice(G.nodes(), p=degrees/total_deg)
        if rand_node != new_node:
            targets.add(rand_node)

    # Connect new node
    for t in targets:
        G.add_edge(new_node, t)

        # Step 3: Rewire with probability p_rewire
        if random.random() < p_rewire:
            G.remove_edge(new_node, t)
            possible_nodes = list(set(G.nodes()) - {new_node})
            new_target = random.choice(possible_nodes)
            G.add_edge(new_node, new_target)

# --- Analysis ---
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())
print("Average clustering coefficient:", nx.average_clustering(G))

# Diameter of largest component
largest_cc_nodes = max(nx.connected_components(G), key=len)
largest_cc = G.subgraph(largest_cc_nodes)
print("Diameter (largest component):", nx.diameter(largest_cc))


# --- Degree Distribution ---
degrees = [d for n, d in G.degree()]
plt.figure(figsize=(8,5))
plt.hist(degrees, bins=range(max(degrees)+1), alpha=0.7, color='orange')
plt.xlabel("Degree k")
plt.ylabel("Number of nodes")
plt.title("Degree Distribution of Hybrid Network")
plt.show()

# --- Visualize network ---
plt.figure(figsize=(8,8))
nx.draw_spring(G, node_size=50, node_color='skyblue', edge_color='gray')
plt.title("Hybrid Network Visualization")
plt.show()





#pseudocode -> seperate communities

# Parameters
N = 120       # total nodes
C = 3         # number of communities
n0 = 5        # initial nodes per community
m = 2         # edges per new node
p_inter = 0.05  # probability of inter-community edge

G = nx.Graph()
community_map = {}  # node -> community

# Step 1: Create seed communities
node_id = 0
for c in range(C):
    H = nx.erdos_renyi_graph(n0, 0.5, seed=random.randint(0, 1000))
    mapping = {n: n + node_id for n in H.nodes()}
    H = nx.relabel_nodes(H, mapping)
    G = nx.compose(G, H)
    for n in H.nodes():
        community_map[n] = c
    node_id += n0

# Step 2: Add new nodes
while G.number_of_nodes() < N:
    new_node = node_id
    G.add_node(new_node)

    # Pick a community randomly
    c = random.randint(0, C - 1)

    # Assign community BEFORE attachment
    community_map[new_node] = c

    # Preferential attachment inside community
    community_nodes = [n for n in G.nodes() if community_map[n] == c and n != new_node]
    degrees = np.array([G.degree(n) for n in community_nodes])
    total_deg = degrees.sum()

    targets = set()
    while len(targets) < m and community_nodes:
        if total_deg == 0:
            targets.add(random.choice(community_nodes))
        else:
            t = np.random.choice(community_nodes, p=degrees / total_deg)
            targets.add(t)

    # Connect new node
    for t in targets:
        if random.random() < p_inter:
            # Rewire to another community
            other_com = random.choice([i for i in range(C) if i != c])
            other_nodes = [n for n in G.nodes() if community_map[n] == other_com]
            if other_nodes:
                t = random.choice(other_nodes)
        G.add_edge(new_node, t)

    node_id += 1

# --- Analysis ---
avg_clust_er, avg_path_er, avg_deg_er, degrees_er = compute_metrics(G, "Community Network with Preferential Attachment")

# Visualize communities
pos = nx.spring_layout(G, seed=42)
colors = [community_map[n] for n in G.nodes()]
plt.figure(figsize=(8, 8))
nx.draw_networkx(G, pos, node_color=colors,
                 node_size=100, with_labels=False)
plt.title("Community Network with Preferential Attachment")
plt.show()
