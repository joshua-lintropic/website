import networkx as nx
import matplotlib.pyplot as plt

K = 2394
M = 12

fiber_indices = [1, 2, 3, K-2, K-1, K]
fiber_labels  = [f"$f_{{{i}}}$" for i in fiber_indices]
ellipsis_f    = "..."

class_indices = [1, 2, M-1, M]
class_labels  = [f"$c_{{{i}}}$" for i in class_indices]
ellipsis_c    = "..."

# build graph
G = nx.Graph()
for f in fiber_labels:
    G.add_node(f, part="fiber")
G.add_node(ellipsis_f, part="fiber_ellipsis")

for c in class_labels:
    G.add_node(c, part="class")
G.add_node(ellipsis_c, part="class_ellipsis")

# connect fiber nodes to class nodes
for f in fiber_labels:
    for c in class_labels:
        G.add_edge(f, c)

# layout coordinates
pos = {}

# create fiber coordinates
y_f_top = [2.5, 2.0, 1.5]
y_f_bot = [-1.5, -2.0, -2.5]
for f, y in zip(fiber_labels[:3], y_f_top):
    pos[f] = (-1.0, y)
for f, y in zip(fiber_labels[3:], y_f_bot):
    pos[f] = (-1.0, y)
pos[ellipsis_f] = (-1.0, 0.0)

# create class coordinates
y_c_top = [1.0, 0.5]
y_c_bot = [-0.5, -1.0]
for c, y in zip(class_labels[:2], y_c_top):
    pos[c] = (1.0, y)
for c, y in zip(class_labels[2:], y_c_bot):
    pos[c] = (1.0, y)
pos[ellipsis_c] = (1.0, 0.0)

# draw nodes 
plt.figure(figsize=(8, 6), dpi=600)
nx.draw_networkx_nodes(G, pos,
                       nodelist=fiber_labels,
                       node_color="#ffcccc",
                       node_size=600)
plt.text(-1.0, 0.0, ellipsis_f, fontsize=18, ha="center", va="center")
nx.draw_networkx_nodes(G, pos,
                       nodelist=class_labels,
                       node_color="#ccccff",
                       node_size=600)
plt.text(1.0, 0.0, ellipsis_c, fontsize=18, ha="center", va="center")

# draw edges
edge_list = [(f, c) for f in fiber_labels for c in class_labels]
nx.draw_networkx_edges(G, pos, edgelist=edge_list, alpha=0.6)

# plot metadata
nx.draw_networkx_labels(G, pos, font_size=9)
plt.axis('off')
plt.title(f"PFS Class-Level Bipartite Graph", pad=5, fontsize=18)
plt.tight_layout()
plt.savefig('../assets/images/class_bipartite.webp')