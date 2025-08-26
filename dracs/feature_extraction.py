import networkx as nx
import time, warnings
import numpy as np
from functools import wraps
from dracs.aif_graph_utils import GraphAIF
from copy import deepcopy

def log_execution_time(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        # Check if self.verbose is True before printing
        if hasattr(self, 'verbose') and self.verbose:
            print(f"Function '{func.__name__}' executed in {execution_time:.4f} seconds.")
        return result
    return wrapper


class AIFTimeFillingFeatureExtractor:
    def __init__(self, aif_method='binary', remove_node_types=["TA", "L"], verbose=False):
        self.aif = {
            "nodes": [],
            "edges": []
        }

        self.nodeID_to_num = {}
        self.vector_names = []

        self.nx_graph = nx.DiGraph()
        self.graph_aif = {
            "nodes": [],
            "edges": []
        }
        self.aif_method = aif_method
        self.remove_node_types = remove_node_types

        self.vector_dims_features = [
            self.edges_stats_dims,
            self.nodes_degrees_stats_dim,
            self.path_stats_dims,
            self.core_stats_dims,
            self.num_leafs_stats_dims,
            self.connected_components_stats_dims,
            self.isolated_i_nodes_stats_dims,
        ]

        self.relation_types_text = [
            ("YA", "Asserting"),
            ("RA", "Default Inference"),
            ("YA", "Arguing"),
            ("MA", "Default Rephrase"),
            ("YA", "Restating"),
            ("YA", "Analysing"),
            ("CA", "Default Conflict"),
            ("YA", "Disagreeing"),
            ("YA", "Default Illocuting"),
            ("YA", "Rhetorical Questioning"),
            ("YA", "Pure Questioning"),
            ("YA", "Agreeing")
        ]

        self.verbose = verbose

        # assign self.vector_names with self.get_features_vector
        self.vector_names = []
        self.get_features_vector()


    def is_ya(self, nodeID):
        for node_dict in self.aif['nodes']:
            if node_dict['nodeID'] == nodeID:
                return node_dict['type'] == "YA"
        return False
    def is_isolated(self, node_dict):
        if node_dict['type'] != "I":
            return False

        for edge_dict in self.aif['edges']:
            if edge_dict['fromID'] == node_dict['nodeID']:
                return False
            if edge_dict['toID'] == node_dict['nodeID']:
                if not self.is_ya(edge_dict['fromID']):
                    return False
        return True

    def isolated_i_nodes_stats_dims(self, **kwargs):
        n_isolated = 0
        for node_dict in self.aif['nodes']:
            if self.is_isolated(node_dict):
                n_isolated += 1

        return [n_isolated], ['N Isolated I']


    def is_leaf(self, node_dict):
        if node_dict['type'] != "I":
            return False

        for edge_dict in self.aif['edges']:
            if edge_dict['fromID'] == node_dict["nodeID"]:
                return False
        return True

    def num_leafs_stats_dims(self, vectors_history, **kwargs):

        n_leafs = 0
        for node_dict in self.aif['nodes']:
            if self.is_leaf(node_dict):
                n_leafs += 1

        return [n_leafs], ['N Leafs I']


    def edges_stats_dims(self, vectors_history, **kwargs):
        features = [
            # len(self.aif['edges']),
            len(self.aif['edges'])/(len(self.aif['nodes'])**2) if len(self.aif['nodes']) else 0.0  # density
        ]
        names = [
            "Density"
        ]
        return features, names

    def average_clustering_stats_dim(self):
        features = [
            nx.average_clustering(G=self.nx_graph)
        ]
        names = ["AVG Clustering"]
        return features, names


    def ya_node(self, nodeID):
        for node_dict in self.aif['nodes']:
            if node_dict['nodeID'] == nodeID:
                return node_dict['type'] == "YA"
        return False

    def nodes_degrees_stats_dim(self, **kwargs):

        i_nodes_degrees = {}


        last_node_in_degree = 0.0
        last_node_out_degree = 0.0
        max_text_occ_idx = -1.0

        for node_dict in self.aif['nodes']:
            if node_dict['type'] == "I":
                in_degree = len(
                    [edge_dict for edge_dict in self.aif['edges'] if edge_dict['toID'] == node_dict['nodeID'] and not self.ya_node(edge_dict['fromID'])]
                )
                out_degree = len(
                    [edge_dict for edge_dict in self.aif['edges'] if edge_dict['fromID'] == node_dict['nodeID']]
                )
                i_nodes_degrees[node_dict['nodeID']] = {
                    "In-Degree": in_degree,
                    "Out-Degree": out_degree
                }
                if "text_occ_idx" in node_dict:
                    if node_dict['text_occ_idx'] > max_text_occ_idx:
                        max_text_occ_idx = node_dict['text_occ_idx']
                        last_node_in_degree = in_degree
                        last_node_out_degree = out_degree

        avg_in_degree = sum([x['In-Degree'] for x in i_nodes_degrees.values()])/len(i_nodes_degrees) if len(i_nodes_degrees) else 0.0
        avg_out_degree = sum([x['Out-Degree'] for x in i_nodes_degrees.values()])/len(i_nodes_degrees) if len(i_nodes_degrees) else 0.0
        avg_degree = sum([x['Out-Degree'] + x['In-Degree'] for x in i_nodes_degrees.values()])/len(i_nodes_degrees) if len(i_nodes_degrees) else 0.0

        max_in_degree = max([x['In-Degree'] for x in i_nodes_degrees.values()]) if len(i_nodes_degrees) else 0.0
        max_out_degree = max([x['Out-Degree'] for x in i_nodes_degrees.values()]) if len(i_nodes_degrees) else 0.0

        features = [
            avg_in_degree,
            avg_out_degree,
            avg_degree,
            max_in_degree,
            max_out_degree,
            last_node_in_degree,
            last_node_out_degree
        ]
        names = [
            "AVG In-Degree",
            "AVG Out-Degree",
            "AVG Degree",
            "MAX In-Degree",
            "MAX Out-Degree",
            "LAST NODE In-Degree",
            "LAST NODE Out-Degree"
        ]

        return features, names




    def build_graph(self, nodes_dicts, edges_dicts):
        self.aif = {
            "nodes": [],
            "edges": []
        }

        self.nodeID_to_num = {}

        self.nx_graph = nx.DiGraph()

        self._set_nodes(nodes_dicts)
        self._set_edges(edges_dicts)

        self.graph_aif = GraphAIF(aif=self.aif, method=self.aif_method, remove_node_types=self.remove_node_types)

    def _set_nodes(self, nodes_dicts):

        for node_dict in nodes_dicts:
            if node_dict['type'] not in ['TA', "L"]:
                if node_dict['nodeID'] not in self.nodeID_to_num:
                    self.nodeID_to_num[node_dict['nodeID']] = len(self.nodeID_to_num)
                    self.aif['nodes'].append(node_dict)

                    self.nx_graph.add_node(self.nodeID_to_num[node_dict['nodeID']])

    def _set_edges(self, edges_dicts):
        for edge_dict in edges_dicts:
            if edge_dict['fromID'] in self.nodeID_to_num:
                if edge_dict['toID'] in self.nodeID_to_num:
                    self.aif['edges'].append(edge_dict)
                    self.nx_graph.add_edge(
                        self.nodeID_to_num[edge_dict['fromID']],
                        self.nodeID_to_num[edge_dict['toID']]
                    )

    @log_execution_time
    def wiener_index_stats_dims(self):
        wiener_index = nx.wiener_index(G=self.nx_graph)
        return [wiener_index], ["Wiener Index"]

    @log_execution_time
    def core_stats_dims(self, vectors_history, **kwargs):
        core_number_dict = nx.core_number(G=self.nx_graph)
        core_number = sum(core_number_dict.values())/len(core_number_dict) if len(core_number_dict) > 0 else 0.0

        return [core_number], ["Core AVG Num"]

    @log_execution_time
    def communicability_stats_dims(self):
        communicability_dict = nx.communicability(G=self.nx_graph)
        avg_communicability = sum(communicability_dict.values())/len(communicability_dict)

        return [avg_communicability], ["AVG Communicability"]

    @log_execution_time
    def connected_components_stats_dims(self, vectors_history, previous_aifs, previous_nx_graphs, **kwargs):
        w_connected_components = list(nx.weakly_connected_components(G=self.nx_graph))

        num_w_components = len(w_connected_components)
        if w_connected_components:
            avg_components_size = sum([len(comp) for comp in w_connected_components])/len(w_connected_components)
        else:
            avg_components_size = 0


        features = [
            num_w_components,
            avg_components_size
        ]
        names = [
            "Num W Comps",
            "AVG Len W Comps"
        ]

        return features, names

    @log_execution_time
    def centrality_stats_dims(self):
        degree_centrality_dict = nx.degree_centrality(G=self.nx_graph)
        avg_degree_centrality = sum(degree_centrality_dict.values())/len(degree_centrality_dict)

        in_degree_centrality_dict = nx.in_degree_centrality(G=self.nx_graph)
        avg_in_degree_centrality = sum(in_degree_centrality_dict.values())/len(in_degree_centrality_dict)

        out_degree_centrality_dict = nx.out_degree_centrality(G=self.nx_graph)
        avg_out_degree_centrality = sum(out_degree_centrality_dict.values())/len(out_degree_centrality_dict)


        features = [
            avg_degree_centrality,
            avg_in_degree_centrality,
            avg_out_degree_centrality
        ]
        names = [
            "AVG Degree",
            "AVG In-Degree Centrality",
            "AVG Out-Degree Centrality"
        ]
        return features, names

    @log_execution_time
    def bridges_stats_dims(self):
        bridges = nx.bridges(G=self.nx_graph)

        num_none_i_nodes = len([x for x in self.aif['nodes'] if x['type'] != "I"])
        avg_num_bridges = len(bridges) / num_none_i_nodes if num_none_i_nodes else 0.0

        return [avg_num_bridges], ['AVG N Bridges']

    @log_execution_time
    def cycles_stats_dims(self):
        simple_cycles = nx.simple_cycles(G=self.nx_graph)
        simple_cycles_lens = [len(x) for x in simple_cycles]


        return [
            sum(simple_cycles_lens)/len(simple_cycles_lens) if len(simple_cycles_lens) else 0.0,
            # len(simple_cycles)
        ], ['AVG Simple Cycles']

    @log_execution_time
    def path_stats_dims(self, vectors_history, **kwargs):
        try:
            longest_path = nx.dag_longest_path(
                G=self.nx_graph
            )
        except nx.exception.NetworkXUnfeasible:
            warnings.warn('It seems like there is a cycle. Longest path will be searched from the longest cycle')
            longest_cycle = sorted(
                list(nx.simple_cycles(G=self.nx_graph)),
                key=lambda s: len(s)
            )
            longest_path = longest_cycle[-1]

        return [
            len(longest_path)/len(self.aif['nodes']) if len(self.aif['nodes']) else 0.0,
            # len(longest_path)
        ], [
            "LEN/NODES Longest Path",
            # 'LEN Longest Path'
        ]

    @log_execution_time
    def node_types_count_stats_dims(self, vectors_history, **kwargs):
        new_count_dims = []
        names = []

        num_relation_nodes = len([x for x in self.aif['nodes'] if x['type'] in ["CA", "RA", "MA", "YA"]])
        for node_type in ['I', 'CA', "RA", "MA", "YA"]:
            new_count_dims.append(
                len([x for x in self.aif['nodes'] if x['type'] == node_type])/num_relation_nodes if num_relation_nodes>0 else 0.0
            )
            names.append(f"{node_type} / (N relat nodes)")

        for node_type, node_text in self.relation_types_text:
            new_count_dims.append(
                len([x for x in self.aif['nodes'] if x['type'] == node_type and x['text'] == node_text])/num_relation_nodes if num_relation_nodes>0 else 0.0
            )
            names.append(f"{node_type} + {node_text} / (N relat nodes)")

        return new_count_dims, names


    def get_features_vector(self, vectors_history=[], previous_aifs=[], previous_nx_graphs=[]):

        vector_names = []
        vector_features = []

        for i, dim_feature_func in enumerate(self.vector_dims_features):
            features, names = dim_feature_func(
                vectors_history=vectors_history,
                previous_aifs=previous_aifs,
                previous_nx_graphs=previous_nx_graphs
            )
            # print(names)
            assert len(features) == len(names)
            vector_features += features
            vector_names += names

        self.vector_names = vector_names

        return vector_features


    def get_features_vectors(self, ts_nodes, ts_edges):

        vectors_history = []
        previous_aifs = []
        previous_nx_graphs = []

        for nodes, edges in zip(ts_nodes, ts_edges):
            self.build_graph(
                nodes_dicts=nodes,
                edges_dicts=edges
            )
            previous_aifs.append(deepcopy(self.aif))
            previous_nx_graphs.append(deepcopy(self.nx_graph))

            vector_features = self.get_features_vector(
                vectors_history=vectors_history,
                previous_aifs=previous_aifs,
                previous_nx_graphs=previous_nx_graphs
            )
            vectors_history.append(vector_features)

        return vectors_history




class AIFTimeChunkSplitter:

    @classmethod
    def is_connecting_existing_i(cls, relation_node_id, idx_i_nodes_ids, edges):
        connects_to = [edge_dict['toID'] for edge_dict in edges if edge_dict['fromID'] == relation_node_id]
        connects_to = [n for n in connects_to if n in idx_i_nodes_ids]

        if not connects_to:
            return False

        connects_from = [edge_dict['fromID'] for edge_dict in edges if edge_dict['toID'] == relation_node_id]
        if not connects_from:
            return True

        connects_from = [n for n in connects_from if n in idx_i_nodes_ids]
        if not connects_from:
            return False

        return True

    @classmethod
    def get_kept_nodes_edges(cls, kept_nodes_ids, edges):
        kept_edges = []

        for edge_dict in edges:
            if edge_dict['fromID'] in kept_nodes_ids:
                if edge_dict['toID'] in kept_nodes_ids:
                    kept_edges.append(edge_dict)

        return kept_edges

    @classmethod
    def get_ya_nodes(cls, i_nodes, aif):
        ya_edges, ya_nodes = [], []

        for node_dict in aif['nodes']:
            if node_dict['type'] == "YA":
                for edge_dict in aif['edges']:
                    if edge_dict['fromID'] == node_dict['nodeID']:
                        if edge_dict['toID'] in i_nodes:
                            ya_edges.append(edge_dict)
                            ya_nodes.append(node_dict)

        return ya_edges, ya_nodes

    @classmethod
    def ya_connects(cls, ya_node_id, idx_i_nodes_ids, edges):
        for edge_dict in edges:
            if edge_dict['fromID'] == ya_node_id:
                if edge_dict['toID'] in idx_i_nodes_ids:
                    return True
        return False

    @classmethod
    def gen_prepare_order_list_nodes(cls, aif):
        all_idx_occ = [x['text_occ_idx'] for x in aif['nodes'] if 'text_occ_idx' in x and x['type'] == "I"]
        all_idx_occ = list(sorted(set(all_idx_occ)))

        all_i_nodes = [
            node_dict for node_dict in aif['nodes'] if node_dict['type'] == "I"
        ]

        for idx in all_idx_occ:
            # try:
            idx_i_nodes = [
                node_dict for node_dict in all_i_nodes if node_dict['text_occ_idx'] <= idx
            ]
            # except:
            #     print()

            idx_i_nodes_ids = [x['nodeID'] for x in idx_i_nodes]
            kept_nodes = deepcopy(idx_i_nodes)

            for node_dict in aif['nodes']:
                if node_dict['type'] not in ['I', "L", "TA"]:

                    if cls.is_connecting_existing_i(
                            relation_node_id=node_dict['nodeID'],
                            idx_i_nodes_ids=idx_i_nodes_ids,
                            edges=aif['edges']
                    ):
                        kept_nodes.append(node_dict)

                if node_dict['type'] == "YA":
                    if cls.ya_connects(
                            ya_node_id=node_dict['nodeID'],
                            idx_i_nodes_ids=idx_i_nodes_ids,
                            edges=aif['edges']
                    ):
                        kept_nodes.append(node_dict)


            kept_nodes_ids = [x['nodeID'] for x in kept_nodes]
            kept_edges = cls.get_kept_nodes_edges(kept_nodes_ids, aif['edges'])

            # print(idx, sorted(kept_nodes_ids))

            yield kept_nodes, kept_edges

    @classmethod
    def get_time_features(cls, aif, aif_method='binary', remove_node_types=["TA", "L"], verbose=False, feature_extractor=AIFTimeFillingFeatureExtractor):

        extractor = feature_extractor(
            aif_method=aif_method,
            remove_node_types=remove_node_types,
            verbose=verbose
        )

        ts_kept_nodes, ts_kept_edges = [],[]
        for kept_nodes, kept_edges in cls.gen_prepare_order_list_nodes(aif):
            ts_kept_nodes.append(kept_nodes)
            ts_kept_edges.append(kept_edges)

        vector_ts = extractor.get_features_vectors(
            ts_nodes=ts_kept_nodes,
            ts_edges=ts_kept_edges
        )

        return np.array(vector_ts), extractor.vector_names