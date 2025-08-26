import numpy as np
from collections import Counter


class GraphAIF:
    def __init__(self, aif, method='binary', remove_node_types=["L", "TA"]):
        self.aif = aif
        self.remove_node_types = remove_node_types
        self._remove_info_nodes()

        self.method = method

        self.nodeID2nodeDict = {
            node_dict['nodeID']: node_dict for node_dict in self.aif['nodes']
        }
        self.connection_pairs_types_weights = self._get_connection_pairs_types_weights()

        self.adjacency_matrix_r = self._aif2adjacency_R()
        self.degree_matrix_r = self._get_degree_matrix_R()


    def _remove_info_nodes(self):
        l_nodes = [node_dict for node_dict in self.aif['nodes'] if node_dict['type'] in self.remove_node_types]
        drop_nodes_idx = []
        nodeID_to_dict = {
            node_dict['nodeID']: node_dict for node_dict in self.aif['nodes']
        }

        self.aif['edges'] = [edge_dict for edge_dict in self.aif['edges'] if edge_dict['toID'] in nodeID_to_dict and edge_dict['fromID'] in nodeID_to_dict]

        for l_node in l_nodes:
            attached_ya_node_id = [
                edge_dict['toID'] for edge_dict in self.aif['edges']
                if edge_dict['fromID'] == l_node['nodeID']
                   and nodeID_to_dict[edge_dict['toID']]['type'] == 'YA'
            ]
            if not len(attached_ya_node_id):
                drop_nodes_idx.append(l_node['nodeID'])

        self.aif['nodes'] = [
            node_dict for node_dict in self.aif['nodes'] if node_dict['nodeID'] not in drop_nodes_idx
        ]
        self.aif['edges'] = [
            edge_dict for edge_dict in self.aif['edges']
            if edge_dict['toID'] not in drop_nodes_idx and edge_dict['fromID'] not in drop_nodes_idx
        ]


    def _get_degree_matrix_R(self):
        diag_matrix = np.zeros(self.adjacency_matrix_r.shape)
        for i in range(len(self.adjacency_matrix_r)):
            diag_matrix[i, i] += sum(self.adjacency_matrix_r[i, :]) + sum(self.adjacency_matrix_r[:, i])
        return diag_matrix


    def _get_connection_pairs_types_weights(self):

        all_pairs_connections_types = []
        for edge_dict in self.aif['edges']:
            if edge_dict['toID'] in self.nodeID2nodeDict and edge_dict['fromID'] in self.nodeID2nodeDict:
                if self.nodeID2nodeDict[edge_dict['toID']]['type'] not in ["L", "TA"]:
                    if self.nodeID2nodeDict[edge_dict['fromID']]['type'] not in ["L", "TA"]:
                        all_pairs_connections_types.append(
                            self.nodeID2nodeDict[edge_dict['toID']]['type'] + "+++" + self.nodeID2nodeDict[edge_dict['fromID']]['type']
                        )

        freq_dict = dict(Counter(all_pairs_connections_types))

        if len(freq_dict) > 1:
            mean = np.mean(list(freq_dict.values()))
            std = np.std(list(freq_dict.values()))

            for k in freq_dict:
                freq_dict[k] = (freq_dict[k] - mean)/std
        else:
            for k in freq_dict:
                freq_dict[k] = 1.0

        return freq_dict


    def calc_edge_score(self, edge_dict):
        if self.method == 'binary':
            return 1.0

        if self.method == "pairs_st_weighted":
            return self.connection_pairs_types_weights[self.nodeID2nodeDict[edge_dict['toID']]['type'] + "+++" + self.nodeID2nodeDict[edge_dict['fromID']]['type']]




    def _aif2adjacency_R(self):

        r_nodes = [
            node_dict['nodeID'] for node_dict in self.aif['nodes'] if node_dict['type'] not in ["L", "TA"]
        ]
        r_nodes = list(set(r_nodes))
        r_nodes2idx = {
            nodeID: enum_i for enum_i, nodeID in enumerate(r_nodes)
        }

        adjacency_matrix_r = np.zeros((len(r_nodes2idx), len(r_nodes2idx)))

        for edge_dict in self.aif['edges']:

            if edge_dict["fromID"] in r_nodes2idx and edge_dict['toID'] in r_nodes2idx:

                edge_score = self.calc_edge_score(edge_dict)
                # try:
                adjacency_matrix_r[
                    r_nodes2idx[edge_dict["fromID"]],
                    r_nodes2idx[edge_dict['toID']]
                ] = edge_score
                # except:
                #     print()

        return adjacency_matrix_r


def aif_calculate_spectral_distance(aif_1, aif_2):
    laplacian_1 = aif_1.degree_matrix_r - aif_1.adjacency_matrix_r
    laplacian_2 = aif_2.degree_matrix_r - aif_2.adjacency_matrix_r

    eigenvals_1, eigenvectors_1 = np.linalg.eig(laplacian_1)
    eigenvals_2, eigenvectors_2 = np.linalg.eig(laplacian_2)


    min_num_eigenvals = min([len(eigenvals_1), len(eigenvals_2)])

    dist = np.linalg.norm(
        eigenvals_1[:min_num_eigenvals] - eigenvals_2[:min_num_eigenvals]
    )

    return dist