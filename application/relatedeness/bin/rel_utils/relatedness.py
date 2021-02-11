import pandas as pd
import numpy as np


## Template
# itemId, similar_items
# 7733.0, "[109, 121313, 120219, 119605, 122450]"
# num similar :int(len(similarities_for_item.argsort()[0])*cfg.top_k_similar_items)

def evaluate_katz_relatedness(hashmap_shortest_paths, alpha, topk, top_k_similar_items):
    """

    :param hashmap_shortest_paths:
    :param alpha:  the effectiveness of a link between two nodes is governed by a known, constant probability, α
    :param top_k_similar_items: The quartile of items to save
    :return:
    """
    similar_items = pd.DataFrame()

    for target_item in hashmap_shortest_paths.keys():
        rel_target_end_at1, rel_target_end_atk = [], []
        for end_item in hashmap_shortest_paths[target_item].keys():
            top_k_paths = hashmap_shortest_paths[target_item][end_item]
            # -1 since we have the head and tail of the path and the length is the number of edges
            rel_target_end_atk.append(np.mean([np.power(alpha, len(p) - 1) for p in top_k_paths[:topk]]))

        # For now we are storing the list based on katz_rel_at_10
        end_items = list(hashmap_shortest_paths[target_item].keys())
        top_similar_index_ids = np.argsort(rel_target_end_atk)[::-1][
                                :int(len(end_items) * top_k_similar_items)]  # Sort from bigger to smaller
        similar_items = similar_items.append({
            'itemId': int(target_item),
            'similar_items': [end_items[top_similar_index_id] for top_similar_index_id in
                              top_similar_index_ids]
        }, ignore_index=True)

    return similar_items


def evaluate_exclusivity_relatedness(hashmap_shortest_paths, dict_node_to_subject, dict_subject_relation_from_node,
                                     dict_node_relation_to_subject, alpha, topk, top_k_similar_items):
    dict_exclusivity = {}  # Key is the (item_node, edge_type) where edge type comes from df_item_features
    dict_weights = {}

    for target_item in hashmap_shortest_paths.keys():
        dict_weights[target_item] = {}
        for end_item in hashmap_shortest_paths[target_item].keys():
            dict_weights[target_item][end_item] = {}
            top_k_paths = hashmap_shortest_paths[target_item][end_item]
            for path in top_k_paths[:topk]:
                temp_exclusivity = []
                for num_pair_nodes in range(len(path) - 1):
                    abs_x_t_star, abs_star_t_y = 0, 0
                    x, y = path[num_pair_nodes:num_pair_nodes + 2]

                    if (x, y) in dict_node_to_subject:
                        r = dict_node_to_subject[(x, y)]
                    else:
                        r = dict_node_to_subject[(y, x)]
                        x, y = y, x  # reverse The indices since it is a hop from attribute -> item in the catalogue

                    if (x, r, y) not in dict_exclusivity:
                        if (x, r) in dict_node_relation_to_subject:
                            abs_x_t_star = len(dict_node_relation_to_subject[(x, r)])
                        if (r, y) in dict_subject_relation_from_node:
                            abs_star_t_y = len(dict_subject_relation_from_node[(r, y)])
                        dict_exclusivity[(x, r, y)] = (abs_x_t_star + abs_star_t_y - 1) ** -1

                    temp_exclusivity.append(dict_exclusivity[(x, r, y)]**-1)

                dict_weights[target_item][end_item][tuple(path)] = np.power(np.sum(temp_exclusivity), -1)

    similar_items = pd.DataFrame()

    for target_item in dict_weights.keys():
        rel_target_end_atk = []
        for end_item in dict_weights[target_item].keys():
            pass
            positions_weights = np.argsort(list(dict_weights[target_item][end_item].values()))[::-1][:topk]
            p_xy = list(dict_weights[target_item][end_item].keys())
            weights = list(dict_weights[target_item][end_item].values())
            rel_target_end_atk.append(np.mean([np.power(alpha, len(p_xy[pos]))*weights[pos] for pos in positions_weights]))

        # For now we are storing the list based on katz_rel_at_10
        end_items = list(hashmap_shortest_paths[target_item].keys())
        top_similar_index_ids = np.argsort(rel_target_end_atk)[::-1][
                                :int(len(end_items) * top_k_similar_items)]

        similar_items = similar_items.append({
            'itemId': int(target_item),
            'similar_items': [end_items[top_similar_index_id] for top_similar_index_id in
                              top_similar_index_ids]
        }, ignore_index=True)

    return similar_items