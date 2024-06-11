import pandas as pd
import networkx as nx

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

print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

import numpy as np
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 計算共同鄰居數量
def common_neighbors(G, u, v):
    return len(list(nx.common_neighbors(G, u, v)))

# 計算兩點間最短路徑
def shortest_path_length(G, u, v):
    try:
        return nx.shortest_path_length(G, u, v)
    except nx.NetworkXNoPath:
        return np.inf

# 計算Adamic-Adar指數
def adamic_adar(G, u, v):
    return sum(1 / np.log(G.degree(w)) for w in nx.common_neighbors(G, u, v))

# 生成正樣本（已存在的邊）
positive_samples = [(u, v, 1) for u, v in G.edges()]

# 生成負樣本（不存在的邊）
non_edges = list(nx.non_edges(G))
negative_sample_indices = np.random.choice(len(non_edges), size=len(positive_samples), replace=False)
negative_samples = [(non_edges[i][0], non_edges[i][1], 0) for i in negative_sample_indices]

# 合併樣本
samples = positive_samples + negative_samples

# 為每個樣本生成特徵
features = []
for u, v, label in samples:
    common_neighbors_count = common_neighbors(G, u, v)
    shortest_path = shortest_path_length(G, u, v)
    adamic_adar_index = adamic_adar(G, u, v)
    features.append([u, v, common_neighbors_count, shortest_path, adamic_adar_index, label])

# 轉換為DataFrame
features_df = pd.DataFrame(features, columns=['source', 'target', 'common_neighbors', 'shortest_path', 'adamic_adar', 'label'])

# 去掉無窮大的最短路徑
features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
features_df.dropna(inplace=True)

# 特徵和標籤
X = features_df[['common_neighbors', 'shortest_path', 'adamic_adar']]
y = features_df['label']

# 標準化特徵
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 訓練邏輯迴歸模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 預測
y_pred = model.predict(X_test)

# 評估模型
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
