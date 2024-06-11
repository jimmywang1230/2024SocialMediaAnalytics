import pandas as pd
import networkx as nx
from networkx.algorithms.link_prediction import resource_allocation_index, jaccard_coefficient, adamic_adar_index

# 讀取CSV文件
links_df = pd.read_csv('stack_network_links.csv')
nodes_df = pd.read_csv('stack_network_nodes.csv')

# 創建無向圖
G = nx.Graph()

# 添加節點
for _, row in nodes_df.iterrows():
    G.add_node(row['name'], group=row['group'], nodesize=row['nodesize'])

# 添加邊
for _, row in links_df.iterrows():
    G.add_edge(row['source'], row['target'], weight=row['value'])

# 打印圖的信息
print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

# 定義 Preferential Attachment 鏈結預測方法
def preferential_attachment(G):
    return ((u, v, G.degree(u) * G.degree(v)) for u, v in nx.non_edges(G))


# 選擇鏈結預測方法
methods = {
    'Resource Allocation Index': resource_allocation_index,
    'Jaccard Coefficient': jaccard_coefficient,
    'Adamic-Adar Index': adamic_adar_index,
    'Preferential Attachment': preferential_attachment,
}

# 執行鏈結預測並打印結果
for method_name, method in methods.items():
    print(f"\n{method_name} Predictions:")
    preds = method(G)
    sorted_preds = sorted(preds, key=lambda x: x[2], reverse=True)[:10]  # 只打印前500個預測結果
    for u, v, p in sorted_preds:
        print(f"({u}, {v}) -> {p:.4f}")
