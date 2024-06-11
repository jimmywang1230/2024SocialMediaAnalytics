import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import leidenalg as la
import igraph as ig

# 讀取數據
links_df = pd.read_csv('stack_network_links.csv')
nodes_df = pd.read_csv('stack_network_nodes.csv')

# 建立NetworkX圖
G = nx.Graph()

# 添加節點
for _, row in nodes_df.iterrows():
    G.add_node(row['name'], group=row['group'], nodesize=row['nodesize'])

# 添加邊
for _, row in links_df.iterrows():
    G.add_edge(row['source'], row['target'], weight=row['value'])

# 將NetworkX圖轉換為iGraph圖
ig_graph = ig.Graph.TupleList(G.edges(data=True), weights=True, directed=False)

# 應用Leiden算法
partition = la.find_partition(ig_graph, la.ModularityVertexPartition)

# 獲取社群數量
num_communities = len(partition)

print(f'Number of communities found: {num_communities}')

# 可視化結果
# 將社群分配回NetworkX節點
for i, community in enumerate(partition):
    for node in community:
        G.nodes[list(G.nodes)[node]]['community'] = i

# 生成顏色映射
community_colors = [G.nodes[node]['community'] for node in G.nodes]

plt.figure(figsize=(15, 15))
pos = nx.spring_layout(G, seed=5, k=1.50)  # k 值可以調整節點之間的距離
nx.draw_networkx_nodes(G, pos, node_color=community_colors, cmap=plt.cm.rainbow, node_size=[G.nodes[node]['nodesize'] for node in G])
nx.draw_networkx_edges(G, pos, alpha=0.5)

# 添加標籤
labels = {node: node for node in G.nodes()}
nx.draw_networkx_labels(G, pos, labels, font_size=7)

plt.title('Leiden Community Detection')
plt.show()
