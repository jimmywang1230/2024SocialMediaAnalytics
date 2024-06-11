# Readme_img/Stack Overflow Tag Network Analysis

### Work Distribution

M11202260 陳伯軒: Leiden + Link Prediction

M11207508 沈煜彬: Centrality Analysis

M11207509 王佑強: Dataset + Community Detection

## Dataset

1. Readme_img/Stack_network_links.csv：
    
    
    | source | target | value |
    | --- | --- | --- |
    | http://asp.net/ | .net | 48.40702996199019 |
    | entity-framework | .net | 24.37090250532431 |
    | devops | amazon-web-services | 24.98353120101788 |
    | docker | amazon-web-services | 32.198071014100535 |
    | ios | android | 39.77803622570551 |
    | android-studio | android | 33.661083176336234 |
    | android | android-studio | 33.661083176336234 |
    
    描述技術標籤之間的連接，哪些技術標籤經常一起出現在開發者的履歷中，分析不同技術標籤之間的關係和關聯強度
    
    欄位：
    
    - source：與目標技術標籤相關的來源技術標籤
    - target：與來源技術標籤相關的目標技術標籤
    - value：表示這對技術標籤之間連接強度的數值。這個值可能表示這對技術標籤共同出現的頻率或相關度
2. Readme_img/Stack_network_nodes.csv：
    
    
    | name | group | nodesize |
    | --- | --- | --- |
    | html | 6 | 272.45 |
    | css | 6 | 341.17 |
    | hibernate | 8 | 29.83 |
    | spring | 8 | 52.84 |
    
    網路中每個技術標籤的詳細訊息，分析技術標籤在整個網路中的分佈和重要性
    
    欄位：
    
    - name：技術標籤的名稱
    - group：技術標籤所屬的群組，表示相關技術標籤的分組這些群組是通過群集算法
    - nodesize：技術標籤的大小，表示該標籤被使用的頻率或流行度
    
    ### Overview:
    
    總共有115個nodes, 490個links
    
    ![截圖 2024-06-11 下午4.52.18.png](Readme_img/Stack%20Overflow%20Tag%20Network%20Analysis%2036a2eef368dd4661b694676383f1fc3f/%25E6%2588%25AA%25E5%259C%2596_2024-06-11_%25E4%25B8%258B%25E5%258D%25884.52.18.png)
    
    ### **Nodes group distribution**
    
    ![Untitled](Readme_img/Stack%20Overflow%20Tag%20Network%20Analysis%2036a2eef368dd4661b694676383f1fc3f/Untitled.png)
    

## Purpose

對於程式學習新手來說，要從哪一個程式語言開始起手、最適合新手的程式語言是什麼等問題是經典月經文，有些人學習完一個程式語言後往往不知道下一步該學習什麼技術。因此我們透過Readme_img/Stack overflow的技術標籤關聯網路分析相關技術標籤之間的關係，找出最為流行的數個程式語言或技術，以及與這些相關的技術，建立初學者的學習路徑。

1. 社群偵測：使用 `Leiden`、`Infomap` 演算法做社群檢測，找出哪些技術標籤經常一起出現形成一個社群，幫助理解可能需要一併學習的技術
2. 中心性分析：可以分析哪些技術標籤在網路中的位置最重要，也就是哪些技術標籤有最多的連接。這可以幫助我們找出最受歡迎或最重要的技術。ex: 程度中心性(`Degree Centrality`)、接近中心性(`Clossness Centrality`)、間接中間性(`betweenness centrality`)以及特徵向量中心性(`Eigenvector Centrality`)。
3. 鏈結預測分析方法：可以分析哪些技術標籤之間的連接最強，也就是哪些技術標籤最常一起出現。分析推薦的順序。

---

## Community Detection_M11207509 王佑強

此次使用了三種Community detection演算法：`Infomap`, `Louvain`, `Leiden`

### Infomap

基於最小化網路上random walk的描述長度。它使用information-theoretic approach壓縮描述網路上的flow來尋找模組。

- **過程**：Infomap 通過模擬網絡上的random walk，將random walk經常訪問的節點分組到同一社群中，並使用 Map Equation，最小化描述random walk軌跡所需的資訊。
- **優點**：在資料流或活動明顯的網絡中較有效。
- (a)random walk; (b)根據random walk的機率直接建構huffman編碼; (c)層次編碼; (d)層次編碼中的類別編碼。最下方顯示了對應的編碼序列，可以看到層次編碼的編碼序列更短
    
    ![Untitled](Readme_img/Stack%20Overflow%20Tag%20Network%20Analysis%2036a2eef368dd4661b694676383f1fc3f/Untitled%201.png)
    

### Louvain

- Louvain是一種迭代的貪婪優化算法，最大化網路的modularity來偵測社群。modularity 用來衡量社群內與社群間鏈接密度的指標。
- **過程**：Modularity優化和社群集合。從每個節點作為一個社群開始，然後反覆移動節點以增加modularity。一旦達到局部最大值，它會將屬於同一社群的節點集合並重複這個過程。
- **優點**：非常快速，能夠有效處理大規模網絡。

### Leiden

- Leiden 優化了 Louvain 方法，減少限制並最大化模塊度。
- **過程**：與 Louvain 類似，Leiden 分階段迭代。確保生成的社群連接品質，優化流程以避免 Louvain 方法中可能出現的不良連接社群。
- **優點**：確保社群連接良好，通常能找到更有意義的社群結構。同時，它的效率也很高。
- **缺點**：不適用於複雜的網絡。

### Community Detection Code (Infomap, Louvain)

1. **匯入所需的模組**：
    
    ```python
    import pandas as pd
    import networkx as nx
    from networkx.algorithms import community
    import matplotlib.pyplot as plt
    import leidenalg as la
    import community as community_louvain
    from infomap import Infomap
    import igraph as ig
    
    import numpy as np
    import math
    from random import sample
    import random
    from IPython.display import display 
    import statistics as stat
    from collections import Counter
    ```
    
2. **讀取數據**：
    - 讀取 CSV 文件，這些文件包含了網絡的節點和邊資料。
    
    ```python
    df_links=pd.read_csv('dataset/Readme_img/Stack_network_links.csv')
    df_nodes=pd.read_csv('dataset/Readme_img/Stack_network_nodes.csv')
    ```
    
3. 載入邊與權重
    
    ```python
    edges = df_links[['source', 'target']].values.tolist()
    weights = [float(l) for l in df_links.value.values.tolist()]
    node_size_dict = dict(zip(df_nodes['name'], df_nodes['nodesize']))
    ```
    
4. **建立 NetworkX 圖**：
    - 建立一個無向圖 `G` 並加入權重`weight`。
    
    ```python
    G = nx.Graph()
    G.add_edges_from(edges)
    for cnt, a in enumerate(G.edges(data=True)):
        G.edges[(a[0], a[1])]['weight'] = weights[cnt]
    ```
    
5. **定義 Infomap 函式**：
    
    ```python
    def simple_Infomap(G):
        # Create a mapping from string nodes to integer identifiers
        node_to_int = {node: i for i, node in enumerate(G.nodes())}
        int_to_node = {i: node for node, i in node_to_int.items()}
    
        # Running Infomap for community detection
        infomap = Infomap()
        for e in G.edges(data=True):
            node1, node2 = node_to_int[e[0]], node_to_int[e[1]]
            weight = float(e[2].get('weight', 1))  # Ensure weight is a float
            infomap.addLink(node1, node2, weight)
    
        infomap.run()
    
        # Extracting the community assignments
        infomap_communities = {int_to_node[node]: infomap.getModules()[node] for node in node_to_int.values()}
    
        max_k_w = []
        for com in set(infomap_communities.values()):
            list_nodes = [nodes for nodes in infomap_communities.keys() if infomap_communities[nodes] == com]
            max_k_w = max_k_w + [list_nodes]
    
        node_mapping = {}
        map_v = 0
        for node in G.nodes():
            node_mapping[node] = map_v
            map_v += 1
    
        community_num_group = len(max_k_w)
        color_list_community = [[] for i in range(len(G.nodes()))]
    
        # color
        for i in G.nodes():
            for j in range(community_num_group):
                if i in max_k_w[j]:
                    color_list_community[node_mapping[i]] = j
    
        pos = graphviz_layout(G)
    
        return G, pos, color_list_community, community_num_group, max_k_w
    ```
    
6. **定義 Louvian 函式**
    
    ```python
    def simple_Louvain(G):
        partition = community_louvain.best_partition(G)
        pos = graphviz_layout(G)
        
        max_k_w = []
        for com in set(partition.values()):
            list_nodes = [nodes for nodes in partition.keys()
                          if partition[nodes] == com]
            max_k_w = max_k_w + [list_nodes]
    
        
        node_mapping = {}
        map_v = 0
        for node in G.nodes():
            node_mapping[node] = map_v
            map_v += 1
    
        community_num_group = len(max_k_w)
        color_list_community = [[] for i in range(len(G.nodes()))]
        
        # color
        for i in G.nodes():
            for j in range(community_num_group):
                if i in max_k_w[j]:
                    color_list_community[node_mapping[i]] = j
        
        return G, pos, color_list_community, community_num_group, max_k_w
    ```
    
7. **可視化結果**：
    - 講演算法結果是覺化呈現。
    - 根據顏色區分不同社群。
    - 使用 `matplotlib` 繪製圖形，其中節點根據社群顏色著色，邊按權重連接。
    - 添加節點標籤以顯示節點名稱。
    
    ```python
    # Visualizing the network with nodes colored by community
    edges = G.edges()
    # node_size = 70
    node_size = [node_size_dict.get(node, 70) for node in G.nodes()]
    
    # Visualizing Infomap communities
    fig = plt.figure(figsize=(20, 10))
    im = nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=infomap_color_list_community, cmap='jet', vmin=0, vmax=infomap_community_num_group)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, font_size=12, font_color="black")
    plt.xticks([])
    plt.yticks([])
    plt.title('Readme_img/Stack Overflow Tag Network - Infomap Community Detection')
    plt.show(block=False)
    
    # Visualizing Louvain communities
    fig = plt.figure(figsize=(20, 10))
    im = nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=louvian_color_list_community, cmap='jet', vmin=0, vmax=louvian_community_num_group)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, font_size=12, font_color="black")
    plt.xticks([])
    plt.yticks([])
    plt.title('Readme_img/Stack Overflow Tag Network - Louvian Community Detection')
    plt.show(block=False)
    
    # Compare community detection results
    print("Infomap detected", len(set(infomap_color_list_community)), "communities.")
    print("Louvain method detected", len(set(louvian_color_list_community)), "communities.")
    
    ```
    

### 輸出結果：

- **節點顏色**：節點顏色代表不同的社群，每個社群由 Leiden 演算法檢測出來。
- **節點大小**：節點大小根據 `nodesize` 屬性設定，通常代表該節點的重要性或連接數。
- **標籤**：每個節點的標籤顯示其名稱，方便識別。

本次分析中，Infomap演算法找到了7個社群(communities), Louvian演算法找到14個社群。

1. **社群的概念**：
    - 在社群檢測（community detection）中，社群是一組節點，這些節點彼此之間有較多的連接（邊），而與其他社群的節點連接較少。
    - 換句話說，社群內的節點更加緊密地連接在一起，而社群之間的連接相對稀疏。
2. **Leiden演算法**：
    - Leiden演算法是一種基於圖（graph）的社群檢測方法，它改進了著名的Louvain演算法，在社群結構上有更好的精確度和穩定性。
    - 它通過最大化模塊度（modularity）來劃分社群，模塊度是一個衡量社群結構的指標。

### 分析結果：

Communities found: Infomap→7, Louvian→14。

![截圖 2024-06-10 下午3.31.16.png](Readme_img/Stack%20Overflow%20Tag%20Network%20Analysis%2036a2eef368dd4661b694676383f1fc3f/%25E6%2588%25AA%25E5%259C%2596_2024-06-10_%25E4%25B8%258B%25E5%258D%25883.31.16.png)

![Untitled](Readme_img/Stack%20Overflow%20Tag%20Network%20Analysis%2036a2eef368dd4661b694676383f1fc3f/Untitled%202.png)

---

## Leiden+Link Prediction_M11202260 陳伯軒

### Leiden 演算法分析

1. **匯入所需的模組**：
    
    ```python
    import pandas as pd
    import networkx as nx
    import matplotlib.pyplot as plt
    import leidenalg as la
    import igraph as ig
    ```
    
2. **讀取數據**：
    - 讀取 CSV 文件，這些文件包含了網絡的節點和邊數據。
    
    ```python
    links_df = pd.read_csv('Readme_img/Stack_network_links.csv')
    nodes_df = pd.read_csv('Readme_img/Stack_network_nodes.csv')
    ```
    
3. **建立 NetworkX 圖**：
    - 建立一個無向圖 `G`。
    - 從 `nodes_df` 中添加節點，每個節點具有名稱 (`name`)、群組 (`group`) 和節點大小 (`nodesize`) 屬性。
    - 從 `links_df` 中添加邊，邊具有源 (`source`)、目標 (`target`) 和權重 (`weight`) 屬性。
    
    ```python
    G = nx.Graph()
    for _, row in nodes_df.iterrows():
        G.add_node(row['name'], group=row['group'], nodesize=row['nodesize'])
    for _, row in links_df.iterrows():
        G.add_edge(row['source'], row['target'], weight=row['value'])
    ```
    
4. **將 NetworkX 圖轉換為 iGraph 圖**：
    - 使用 `igraph` 將 `NetworkX` 圖轉換成 `iGraph` 圖。
    
    ```python
    ig_graph = ig.Graph.TupleList(G.edges(data=True), weights=True, directed=False)
    ```
    
5. **應用 Leiden 演算法**：
    - 使用 Leiden 演算法進行社群檢測。
    - 獲取並打印社群數量。
    
    ```python
    partition = la.find_partition(ig_graph, la.ModularityVertexPartition)
    num_communities = len(partition)
    print(f'Number of communities found: {num_communities}')
    ```
    
6. **可視化結果**：
    - 將社群分配回 NetworkX 節點。
    - 生成顏色映射。
    - 使用 `matplotlib` 繪製圖形，其中節點根據社群顏色著色，邊按權重連接。
    - 添加節點標籤以顯示節點名稱。
    
    ```python
    for i, community in enumerate(partition):
        for node in community:
            G.nodes[list(G.nodes)[node]]['community'] = i
    
    community_colors = [G.nodes[node]['community'] for node in G.nodes]
    
    plt.figure(figsize=(15, 15))
    pos = nx.spring_layout(G, seed=5, k=1.50)
    nx.draw_networkx_nodes(G, pos, node_color=community_colors, cmap=plt.cm.rainbow, node_size=[G.nodes[node]['nodesize'] for node in G])
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    
    labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=7)
    
    plt.title('Leiden Community Detection')
    plt.show()
    ```
    

### 輸出結果：

- **節點顏色**：節點顏色代表不同的社群，每個社群由 Leiden 演算法檢測出來。
- **節點大小**：節點大小根據 `nodesize` 屬性設定，通常代表該節點的重要性或連接數。
- **標籤**：每個節點的標籤顯示其名稱，方便識別。

本次分析中，Leiden演算法找到了13個社群（communities）。

1. **社群的概念**：
    - 在社群檢測（community detection）中，社群是一組節點，這些節點彼此之間有較多的連接（邊），而與其他社群的節點連接較少。
    - 換句話說，社群內的節點更加緊密地連接在一起，而社群之間的連接相對稀疏。
2. **Leiden演算法**：
    - Leiden演算法是一種基於圖（graph）的社群檢測方法，它改進了著名的Louvain演算法，在社群結構上有更好的精確度和穩定性。
    - 它通過最大化模塊度（modularity）來劃分社群，模塊度是一個衡量社群結構的指標。
3. **13個社群**：
    - 本次分析中的網絡圖被劃分成13個不同的社群。
    - 每個社群內的節點在某種程度上是更加密切相關的，例如SQL與PHP、Query、MYPHP之間的連結十分緊密，現實生活中也是如此，通常都會一起使用與討論。

### 示例解釋：

數據來自於Readme_img/Stack Overflow技術標籤的共現網絡，這13個社群可能代表不同技術領域的群體：

- 一個社群可能包含所有與前端開發相關的標籤，如HTML、CSS、JavaScript等。
- 另一個社群可能包含與後端開發相關的標籤，如Node.js、Express、MongoDB等。
- 另一個社群可能包含數據科學相關的標籤，如Python、Machine Learning、Pandas等。

本次分析中，社群幫助識別在技術標籤之間的主要關聯和技術領域的分佈。

![Untitled](Readme_img/Stack%20Overflow%20Tag%20Network%20Analysis%2036a2eef368dd4661b694676383f1fc3f/Untitled%203.png)

### Link Prediction

這些鏈結預測分析分數是用來評估圖中節點之間潛在連接的可能性。每種鏈結預測方法根據不同的理論基礎來計算節點之間的相似度或親近度，從而得出分數。

- **匯入必要的模組**：
    
    ```python
    import pandas as pd
    import networkx as nx
    from networkx.algorithms.link_prediction import resource_allocation_index, jaccard_coefficient, adamic_adar_index
    ```
    
    - `pandas` 用於處理 CSV 文
    - `networkx` 用於創建和操作圖結構。
    - `networkx.algorithms.link_prediction` 提供鏈結預測的不同方法。
- **讀取 CSV 文件**：
    
    ```python
    links_df = pd.read_csv('Readme_img/Stack_network_links.csv')
    nodes_df = pd.read_csv('Readme_img/Stack_network_nodes.csv')
    ```
    
    - `links_df` 包含邊的數據，描述了節點之間的連接。
    - `nodes_df` 包含節點的數據，描述了節點的屬性。
- **創建無向圖**：
    
    ```python
    G = nx.Graph()
    ```
    
    - 創建一個空的無向圖。
- **添加節點**：
    
    ```python
    for index, row in nodes_df.iterrows():
        G.add_node(row['name'], group=row['group'], nodesize=row['nodesize'])
    ```
    
    - 從 `nodes_df` 中迭代每一行，並將節點添加到圖中，節點包含名稱、組別和節點大小等屬性。
- **添加邊**：
    
    ```python
    for index, row in links_df.iterrows():
        G.add_edge(row['source'], row['target'], weight=row['value'])
    ```
    
    - 從 `links_df` 中迭代每一行，並將邊添加到圖中，邊包含來源節點、目標節點和權重。
- **打印圖的信息**：
    
    ```python
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    ```
    
    - 打印圖的節點數和邊數。
    
    ### 定義鏈結預測方法
    
    ```python
    # 定義 Preferential Attachment 
    def preferential_attachment(G):
        preds = ((u, v, G.degree(u) * G.degree(v)) for u, v in nx.non_edges(G))
        return preds
    ```
    
    這裡自定義的鏈結預測方法：
    
    - **Preferential Attachment**：根據節點度的乘積來進行預測。
- **選擇鏈結預測方法**：
    
    ```python
    methods = {
        'Resource Allocation Index': resource_allocation_index,
        'Jaccard Coefficient': jaccard_coefficient,
        'Adamic-Adar Index': adamic_adar_index
        'Preferential Attachment': preferential_attachment
    }
    ```
    
    - 定義一個 methods 來存儲鏈結預測的前三種方法。
- **執行鏈結預測並打印結果**：
    
    ```python
    for method_name, method in methods.items():
        print(f"\n{method_name} Predictions:")
        preds = method(G)
        preds = sorted(preds, key=lambda x: x[2], reverse=True)[:10]
        for u, v, p in preds:
            print(f"({u}, {v}) -> {p:.4f}")
    ```
    
    - 對於每種鏈結預測方法：
        - 打印方法名稱。
        - 使用該方法在圖上進行鏈結預測，得到一個包含預測結果的生成器。
        - 將預測結果按預測分數排序並取前 10 個。
        - 打印每個預測結果，包含節點對和預測分數。

### 1. 資源分配指數 (Resource Allocation Index)

資源分配指數是一種基於資源分配的概念來預測鏈結的方法。它的基本思想是，如果兩個節點之間有共同的鄰居，那麼它們之間形成連接的可能性會更高，該指數計算它們的共通鄰居數，並考慮共通鄰居的度數。具體計算方式是：

![Untitled](Readme_img/Stack%20Overflow%20Tag%20Network%20Analysis%2036a2eef368dd4661b694676383f1fc3f/Untitled%204.png)

**分數意義**：

- 分數越高，表示節點 u 和 v 之間有更多共同的鄰居且這些鄰居的連接數較少，則 u 和 v 之間形成鏈結的可能性越高，該指數衡量了資源在網絡中的分配情況。共通鄰居數較多且度數較小的節點對，連接的可能性更高。
- 高分代表高可能性形成鏈結。

**結果示例**

![Untitled](Readme_img/Stack%20Overflow%20Tag%20Network%20Analysis%2036a2eef368dd4661b694676383f1fc3f/Untitled%205.png)

- **解釋**：這些預測值表示在網絡中具有較高共通鄰居數的節點對，例如，`asp.net-web-api` 和 `.net` 之間的連接可能性很高。這些節點對的共通鄰居可能提供了強有力的支持，說明這些技術可能在開發者社區中有緊密的互動和關聯。分數為0.7095，表明它們有多個共同鄰居且這些鄰居的連接數較少。

### 2. 賈卡德係數 (Jaccard Coefficient)

- **意義**：。

賈卡德係數是一種衡量兩個集合相似度的方法。在鏈結預測中，它被用來衡量兩個節點的鄰居集合之間的相似度。具體計算方式是：

![Untitled](Readme_img/Stack%20Overflow%20Tag%20Network%20Analysis%2036a2eef368dd4661b694676383f1fc3f/Untitled%206.png)

**分數意義**：

- 該係數測量了節點對之間的共同鄰居數與總鄰居數之比，反映了節點對之間的相似度。值越高，節點 u 和 v 之間的鄰居集合重疊程度越高，則 u 和 v 之間形成鏈結的可能性越高。
- 分數範圍在 0 到 1 之間，1 表示完全重疊，0 表示完全不重疊。

**結果示例**

![Untitled](Readme_img/Stack%20Overflow%20Tag%20Network%20Analysis%2036a2eef368dd4661b694676383f1fc3f/Untitled%207.png)

- **解釋**：最高分數（1.0000）的節點對顯示了完全相同的鄰居集合，這表明這些技術通常一起使用。例如，`visual-studio` 和 `unity3d` 經常一起出現在開發環境中，這對於預測這些技術之間可能的合作關係非常有用。

### 3. Adamic-Adar 指數 (Adamic-Adar Index)

Adamic-Adar 指數考慮到共同鄰居數量的同時，也考慮了共同鄰居的稀有性。具體計算方式是：

![Untitled](Readme_img/Stack%20Overflow%20Tag%20Network%20Analysis%2036a2eef368dd4661b694676383f1fc3f/Untitled%208.png)

**分數意義**：

- 該指數考慮了共通鄰居的稀有性，稀有共通鄰居的權重更高，因此能更有效地捕捉稀有連接的潛力。分數越高，表示節點 u 和 v 之間有更多的共同鄰居，且這些鄰居的連接數量較少（即更稀有），則 u 和 v 之間形成鏈結的可能性越高。
- 高分表示高可能性形成鏈結，並且考慮了共同鄰居的影響力。

**結果示例**

![Untitled](Readme_img/Stack%20Overflow%20Tag%20Network%20Analysis%2036a2eef368dd4661b694676383f1fc3f/Untitled%209.png)

- **解釋**：高分數的節點對如 `asp.net-web-api` 和 `.net` 顯示了它們有多個共通鄰居，這些鄰居的稀有性增加了它們之間形成連接的可能性。這些結果指出了在開發者社區中，某些技術組合的罕見但強大的合作關係。

### 4. Preferential Attachment

Preferential Attachment 是指新加入的節點更傾向於連接到已經具有較多連接的節點。這一概念來自於無尺度網絡（scale-free network）的理論，該理論認為網絡中部分節點的度分佈遵循冪次法則。

### 定義和計算

優先連接機制表明，某個節點 被選中作為新連接的概率，與該節點的度 k 成正比，偏好連接是基於節點度的乘積來進行預測的方法

![Untitled](Readme_img/Stack%20Overflow%20Tag%20Network%20Analysis%2036a2eef368dd4661b694676383f1fc3f/Untitled%2010.png)

### 意義

- **高優先連接概率**：度數高的節點（樞紐節點）更有可能吸引新的連接，這導致了網絡中少數節點擁有大量連接，形成所謂的「無尺度」特性，適用於度分佈呈現長尾分佈的網絡。
- **無尺度網絡特性**：無尺度網絡具有「贏者通吃」的特性，少數節點成為網絡的中心，這在網絡韌性和攻擊容忍度方面具有重要意義。

**結果示例**

![Untitled](Readme_img/Stack%20Overflow%20Tag%20Network%20Analysis%2036a2eef368dd4661b694676383f1fc3f/Untitled%2011.png)

- **解釋**：這些高分數預測表明，度數大的節點對如 `c#` 和 `jquery` 更有可能形成新連接。這意味著這些技術在開發者社區中非常受歡迎，並且在未來可能會有更多的交互和合作。

### 分數的意義

- 這些分數都是為了評估節點之間潛在連接的可能性，但使用不同的計算方法和理論基礎。
- 資源分配指數強調共同鄰居的連接稀疏性，揭示了具有高共通鄰居的強連接潛力。
- Jaccard 係數注重集合重疊程度，評估節點之間的相似度。
- Adamic-Adar 指數識別出具有稀有共通鄰居的節點對，指出潛在的稀有但重要的連接。
- Preferential Attachment 則重點考慮節點的度數。

這些預測結果可以幫助這次期末的DATASET有更好地理解技術之間的關聯和互動，從而在開發和合作中做出更明智的決策。

---

## 準確率分析

利用作業一所學的知識去進行準確率分析

- **讀取數據：**

```python
import pandas as pd
import networkx as nx

# 讀取CSV文件
links_df = pd.read_csv('Readme_img/Stack_network_links.csv')
nodes_df = pd.read_csv('Readme_img/Stack_network_nodes.csv')

# 創建無向圖
G = nx.Graph()

# 添加節點
for _, row in nodes_df.iterrows():
    G.add_node(row['name'], group=row['group'], nodesize=row['nodesize'])

# 添加邊
for _, row in links_df.iterrows():
    G.add_edge(row['source'], row['target'], weight=row['value'])

print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
```

- **生成特徵：**
    
    為每對潛在邊生成特徵。包括共同鄰居數量、兩點間最短路徑、Adamic-Adar指數等。
    

```python
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
negative_samples = [(u, v, 0) for u, v in np.random.choice(non_edges, size=len(positive_samples), replace=False)]

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
```

- **分割數據集：**

```python
# 特徵和標籤
X = features_df[['common_neighbors', 'shortest_path', 'adamic_adar']]
y = features_df['label']

# 標準化特徵
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
```

- **訓練和評估模型：**
    
    使用邏輯迴歸來訓練模型，並評估其性能。
    

```python
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
```

**結果示例**

![Untitled](Readme_img/Stack%20Overflow%20Tag%20Network%20Analysis%2036a2eef368dd4661b694676383f1fc3f/Untitled%2012.png)

此期末專案進行二分類問題的模型評估用的指標有準確率（Accuracy）、精確率（Precision）、召回率（Recall）、F1值（F1 Score）和ROC AUC值。

### 1. 準確率（Accuracy）

**公式**: 

![Untitled](Readme_img/Stack%20Overflow%20Tag%20Network%20Analysis%2036a2eef368dd4661b694676383f1fc3f/Untitled%2013.png)

- **TP**（True Positive）：真陽性，即正樣本被正確預測為正樣本的數量
- **TN**（True Negative）：真陰性，即負樣本被正確預測為負樣本的數量
- **FP**（False Positive）：假陽性，即負樣本被錯誤預測為正樣本的數量
- **FN**（False Negative）：假陰性，即正樣本被錯誤預測為負樣本的數量

**解釋**：準確率表示模型預測正確的樣本所佔的比例。我的準確率為0.9474，表示在測試數據中，94.74%的樣本被正確分類。

### 2. 精確率（Precision）

**公式**: 

![Untitled](Readme_img/Stack%20Overflow%20Tag%20Network%20Analysis%2036a2eef368dd4661b694676383f1fc3f/Untitled%2014.png)

**解釋**：精確率表示被預測為正樣本的樣本中，實際為正樣本的比例。我的精確率為0.9157，表示被預測為正樣本的樣本中，91.57%是真正的正樣本。

### 3. 召回率（Recall）

**公式**: 

![Untitled](Readme_img/Stack%20Overflow%20Tag%20Network%20Analysis%2036a2eef368dd4661b694676383f1fc3f/Untitled%2015.png)

**解釋**：召回率表示所有實際正樣本中，被正確預測為正樣本的比例。我的召回率為1.0000，表示所有的正樣本都被正確預測出來了，沒有遺漏。

### 4. F1值（F1 Score）

**公式**: 

![Untitled](Readme_img/Stack%20Overflow%20Tag%20Network%20Analysis%2036a2eef368dd4661b694676383f1fc3f/Untitled%2016.png)

**解釋**：F1值是精確率和召回率的調和平均數，兼顧兩者的平衡。我的F1值為0.9560，表示模型在精確率和召回率之間取得了良好的平衡。

### 5. ROC AUC值

**公式**：ROC AUC值是在ROC曲線下的面積

- **ROC曲線**（Receiver Operating Characteristic Curve）：是通過改變分類閾值來反映真陽性率（TPR）和假陽性率（FPR）之間權衡的曲線。
- **AUC**（Area Under Curve）：表示ROC曲線下的面積，值越大越好，範圍在0.5到1之間。

**解釋**：ROC AUC值衡量模型區分正樣本和負樣本的能力。我的ROC AUC值為0.9386，表示模型在區分正負樣本方面具有很高的能力。

### 總結

- **高準確率（0.9474）**：模型在大多數情況下能正確分類樣本。
- **高精確率（0.9157）**：預測為正的樣本中，大多數是正樣本。
- **高召回率（1.0000）**：所有正樣本都被成功檢測出來。
- **高F1值（0.9560）**：模型在精確率和召回率之間取得了良好平衡。
- **高ROC AUC值（0.9386）**：模型在區分正負樣本方面性能非常好。

---

## Centrality Anaylsis_M11207508 沈煜彬

中心節點普遍被認為在社群網路中扮演重要角色，在本專題中我們以四種方法作為中心性衡量指標：程度中心性(Degree Centrality)、接近中心性(Clossness Centrality)、間接中間性(Betweenness Centrality)以及特徵向量中心性(Eigenvector Centrality)。

- 程度中心性：將「擁有愈多鄰居的節點」定義為中心節點，一個節點的程度中心性越高，表示它與越多的節點直接相連，可能在網路中扮演重要的角色
- 接近中心性：將「與其他節點之平均距離越短的節點」定義為中心，接近中心性越高的節點，表示它到其他節點的平均距離越短，代表它在網路中的位置更中心
- 間接中心性：將「頻繁被其他節點間之最短路徑經過的節點」定義為中心，間接中間性越高的節點，表示它在網路中的許多最短路徑上都出現，代表可能在網路中扮演**橋樑**的角色
- 特徵向量中心性：除了要有較高的程度中心性之外，還要與許多重要節點相連，因此該指標還計算此節點之鄰居對其重要程度的貢獻

在每種中心性指標中，我們分別找出前五名的中心節點與其鄰居，並將該子網路視覺化呈現於網路圖中，以利觀察何種技術與中心節點的相關度最高。

### **程度中心性(Degree Centrality)**

在程度中心性部分，我們得出前五名最具程度中心性的節點為：`jquery`、`c#`、`css`、`asp.net`、`angularjs`，這些節點在網路中與其他節點的連結數最多，可能在網路中扮演重要角色。

| Index | Central Node | Value |
| --- | --- | --- |
| 1 | jquery | 0.2807 |
| 2 | c# | 0.2456 |
| 3 | css | 0.2456 |
| 4 | asp.net | 0.2281 |
| 5 | angularjs | 0.2281 |

![bar_chart.png](Readme_img/Stack%20Overflow%20Tag%20Network%20Analysis%2036a2eef368dd4661b694676383f1fc3f/bar_chart.png)

而各中心節點的子網路圖如下：

![jquery_graph.png](Readme_img/Stack%20Overflow%20Tag%20Network%20Analysis%2036a2eef368dd4661b694676383f1fc3f/jquery_graph.png)

![c_sharp_graph.png](Readme_img/Stack%20Overflow%20Tag%20Network%20Analysis%2036a2eef368dd4661b694676383f1fc3f/c_sharp_graph.png)

![css_graph.png](Readme_img/Stack%20Overflow%20Tag%20Network%20Analysis%2036a2eef368dd4661b694676383f1fc3f/css_graph.png)

![angularjs_graph.png](Readme_img/Stack%20Overflow%20Tag%20Network%20Analysis%2036a2eef368dd4661b694676383f1fc3f/angularjs_graph.png)

![asp.net_graph.png](Readme_img/Stack%20Overflow%20Tag%20Network%20Analysis%2036a2eef368dd4661b694676383f1fc3f/asp.net_graph.png)

### **接近中心性(Closeness Centrality)**

在接近中心性部分，我們得出前五名最具接近中心性的節點為：`jquery`、`mysql`、`ajax`、`css`、`javascript`，這些節點到其他節點的平均距離最短，可能在網路中的位置較中心。

| Index | Central Node | Value |
| --- | --- | --- |
| 1 | jquery | 0.2896 |
| 2 | mysql | 0.2779 |
| 3 | ajax | 0.2586 |
| 4 | css | 0.2579 |
| 5 | javascript | 0.2571 |

![bar_chart.png](Readme_img/Stack%20Overflow%20Tag%20Network%20Analysis%2036a2eef368dd4661b694676383f1fc3f/bar_chart%201.png)

而各中心節點的子網路圖如下：

![jquery_graph.png](Readme_img/Stack%20Overflow%20Tag%20Network%20Analysis%2036a2eef368dd4661b694676383f1fc3f/jquery_graph%201.png)

![ajax_graph.png](Readme_img/Stack%20Overflow%20Tag%20Network%20Analysis%2036a2eef368dd4661b694676383f1fc3f/ajax_graph.png)

![javascript_graph.png](Readme_img/Stack%20Overflow%20Tag%20Network%20Analysis%2036a2eef368dd4661b694676383f1fc3f/javascript_graph.png)

![mysql_graph.png](Readme_img/Stack%20Overflow%20Tag%20Network%20Analysis%2036a2eef368dd4661b694676383f1fc3f/mysql_graph.png)

![css_graph.png](Readme_img/Stack%20Overflow%20Tag%20Network%20Analysis%2036a2eef368dd4661b694676383f1fc3f/css_graph%201.png)

### **間接中間性(Betweenness Centrality)**

在間接中間性部分，我們得出前五名最具間接中間性的節點為：`jquery`、`linux`、`mysql`、`asp.net`、`apache`，這些節點在網路中的許多最短路徑上都出現，可能在網路中扮演橋樑的角色。

| Index | Central Node | Value |
| --- | --- | --- |
| 1 | jquery | 0.2555 |
| 2 | linux | 0.2084 |
| 3 | mysql | 0.1977 |
| 4 | asp.net | 0.1741 |
| 5 | apache | 0.1309 |

![bar_chart.png](Readme_img/Stack%20Overflow%20Tag%20Network%20Analysis%2036a2eef368dd4661b694676383f1fc3f/bar_chart%202.png)

而各中心節點的子網路圖如下：

![jquery_graph.png](Readme_img/Stack%20Overflow%20Tag%20Network%20Analysis%2036a2eef368dd4661b694676383f1fc3f/jquery_graph%202.png)

![mysql_graph.png](Readme_img/Stack%20Overflow%20Tag%20Network%20Analysis%2036a2eef368dd4661b694676383f1fc3f/mysql_graph%201.png)

![apache_graph.png](Readme_img/Stack%20Overflow%20Tag%20Network%20Analysis%2036a2eef368dd4661b694676383f1fc3f/apache_graph.png)

![linux_graph.png](Readme_img/Stack%20Overflow%20Tag%20Network%20Analysis%2036a2eef368dd4661b694676383f1fc3f/linux_graph.png)

![asp.net_graph.png](Readme_img/Stack%20Overflow%20Tag%20Network%20Analysis%2036a2eef368dd4661b694676383f1fc3f/asp.net_graph%201.png)

### **特徵向量中心性(Eigenvector Centrality)**

在特徵向量中心性部分，我們得出前五名最具特徵向量中心性的節點為：`jquery`、`css`、`javascript`、`html5`、`php`，這些節點除了與其他節點的連結數最多之外，還與許多重要節點相連，可能在網路中扮演重要角色。

| Index | Central Node | Value |
| --- | --- | --- |
| 1 | jquery | 0.3658 |
| 2 | css | 0.3387 |
| 3 | javascript | 0.3256 |
| 4 | html5 | 0.2681 |
| 5 | php | 0.2653 |

![bar_chart.png](Readme_img/Stack%20Overflow%20Tag%20Network%20Analysis%2036a2eef368dd4661b694676383f1fc3f/bar_chart%203.png)

而各中心節點的子網路圖如下：

![jquery_graph.png](Readme_img/Stack%20Overflow%20Tag%20Network%20Analysis%2036a2eef368dd4661b694676383f1fc3f/jquery_graph%203.png)

![javascript_graph.png](Readme_img/Stack%20Overflow%20Tag%20Network%20Analysis%2036a2eef368dd4661b694676383f1fc3f/javascript_graph%201.png)

![php_graph.png](Readme_img/Stack%20Overflow%20Tag%20Network%20Analysis%2036a2eef368dd4661b694676383f1fc3f/php_graph.png)

![css_graph.png](Readme_img/Stack%20Overflow%20Tag%20Network%20Analysis%2036a2eef368dd4661b694676383f1fc3f/css_graph%202.png)

![html5_graph.png](Readme_img/Stack%20Overflow%20Tag%20Network%20Analysis%2036a2eef368dd4661b694676383f1fc3f/html5_graph.png)

### 綜合比較

| Index | Degree Centrality | Closeness Centrality | Betweenness Centrality | Eigenvector Centrality |
| --- | --- | --- | --- | --- |
| 1 | jquery | jquery | jquery | jquery |
| 2 | c# | mysql | linux | css |
| 3 | css | ajax | mysql | javascript |
| 4 | asp.net | css | asp.net | html5 |
| 5 | angularjs | javascript | apache | php |

可以看出這四種分析法的最大中心節點都是jquery，但是其他的中心節點就有些不同，這些分析法可以讓我們從不同的角度看待中心節點的重要性，提供更客觀的數據。
