import numpy as np
import networkx as nx
import sys
import os
import math
import glob

feather_path = os.path.dirname(os.path.abspath(__file__))

if feather_path not in sys.path:
    sys.path.append(feather_path)

try:
    from FEATHER.src.feather import FEATHER as OrigFeatherNode
    from FEATHER.src.feather import FEATHERG as OrigFeatherGraph
except ImportError as e:
    print(f"Import error: {e}")
    print("Current sys.path:", sys.path)
    print("Checking if files exist:")
    print(glob.glob(os.path.join(feather_path, 'FEATHER/src/*')))

class CustomFeatherGraph:

    def __init__(
        self,
        theta_max=2.5,
        eval_points=25,
        order=5,
        pooling="mean",
        seed=42
    ):
        self.theta_max = theta_max
        self.eval_points = eval_points
        self.order = order
        self.pooling = pooling
        self.seed = seed
        self.embedding = None

    def _extract_features(self, graph):
        num_nodes = graph.number_of_nodes()

        features = np.zeros((num_nodes, 2))

        clustering_coeffs = nx.clustering(graph)

        for node in range(num_nodes):
            features[node, 0] = math.log(graph.degree(node) + 1.0)

            features[node, 1] = clustering_coeffs.get(node, 0.0)

        return features

    def fit(self, graphs):
        np.random.seed(self.seed)

        processed_graphs = []
        for graph in graphs:
            graph = nx.convert_node_labels_to_integers(graph)
            graph = graph.copy()
            for node in graph.nodes():
                if not graph.has_edge(node, node):
                    graph.add_edge(node, node)
            processed_graphs.append(graph)

        all_embeddings = []

        for graph in processed_graphs:
            node_features = self._extract_features(graph)

            edge_list = [(int(u), int(v)) for u, v in graph.edges()]

            feather_node = OrigFeatherNode(theta_max=self.theta_max,
                                           eval_points=self.eval_points,
                                           order=self.order)

            feather_node.fit(graph, node_features)
            node_embeddings = feather_node.get_embedding()

            if self.pooling == "mean":
                graph_embedding = np.mean(node_embeddings, axis=0)
            elif self.pooling == "max":
                graph_embedding = np.max(node_embeddings, axis=0)
            elif self.pooling == "min":
                graph_embedding = np.min(node_embeddings, axis=0)
            else:
                graph_embedding = np.mean(node_embeddings,
                                          axis=0)  # Default to mean

            all_embeddings.append(graph_embedding)

        self.embedding = np.array(all_embeddings)

        if len(processed_graphs) > 0:
            print(f"Generated embedding dimension: {self.embedding.shape[1]}")

            expected_dim = 2 * self.eval_points * 2 * self.order
            print(f"Expected dimension: {expected_dim}")

        return self

    def get_embedding(self):
        return self.embedding
