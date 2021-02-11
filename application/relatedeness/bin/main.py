import networkx as nx
import pandas as pd
import os
from ast import literal_eval
import time
import sys

from rel_utils.io_util import save_obj
from rel_utils.relatedness import evaluate_katz_relatedness, evaluate_exclusivity_relatedness
from rel_utils.timer import timer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

similarities_file = '{0}_{1}_{2}_similarities.txt'

yahoo_movies = 'yahoo_movies'
yahoo_movies_2_hops = 'yahoo_movies_2_hops'
small_library_thing = 'SmallLibraryThing'
small_library_thing_2_hops = 'SmallLibraryThing2Hops'

project_dir = os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..'))
selection_types = ['categorical', 'ontological', 'factual']
datasets = [small_library_thing, small_library_thing_2_hops, yahoo_movies, yahoo_movies_2_hops]
dataset = datasets[1]
topk = 10  # The threshold on which we build the relatedness measures
topn = 100  # The number of extracted top shortest paths (Not useful in Katz)
top_k_similar_items = 0.25

# Parameters
alpha = 0.25

# Which measures?
is_katz = 1
is_exclusivity = 1


# 0.25 is the best value in:
# Path-based Semantic Relatedness on Linked Data and its use to Word and Entity Disambiguation
# authored by Ioana Hulpu¸s, Narumol Prangnawarat, Conor Hayes


def build():
    print(dataset)
    print('\n')
    df_map = pd.read_csv(os.path.join(project_dir, 'data', dataset, 'df_map.csv'))
    df_selected_features = pd.read_csv(os.path.join(project_dir, 'data', dataset, 'selected_features.csv'))
    df_features = pd.read_csv(os.path.join(project_dir, 'data', dataset, "features.tsv"), sep='\t', header=None)
    df_target_items = pd.read_csv(os.path.join(project_dir, 'data', dataset, "target_items.csv"))
    df_ratings = pd.read_csv(os.path.join(project_dir, 'data', dataset, "ratings.csv"))
    df_ratings = df_ratings.iloc[:, :3]
    df_ratings.columns = ['userId', 'itemId', 'rating']

    items = df_ratings.itemId.unique()
    df_map = df_map[df_map['item'].isin(items)]

    for i in range(len(df_selected_features)):
        start_type = time.time()
        id_features, selection_type = literal_eval(df_selected_features.iloc[i].features), df_selected_features.iloc[
            i].type
        print(selection_type)
        # Take all available features
        df_feature_uris = df_features[df_features[0].isin(id_features)]
        df_item_features = df_map[df_map['feature'].isin(id_features)]
        # Create Graph
        G = nx.Graph()

        # Add items nodes
        for item in items:
            G.add_node(item)

        # Add Object Nodes
        indexer_subjects = {}  # Indexer to Simply the MAP Creation
        subject_id = 0
        for i, row in df_feature_uris.iterrows():
            uris = row[1].split('><')
            for i in range(int(len(uris)/2)):
                uri = uris[1+2*i][:-1] if (i+1) == len(uris)/2 else uris[1+2*i]
                if uri not in indexer_subjects:
                    while subject_id in items:
                        subject_id += 1
                    indexer_subjects[uri] = subject_id
                    # print(subject_node)
                    G.add_node(subject_id)
                    subject_id += 1
            # Else it means that we have already a subject with the same URI but coming from a different property

        # Add edges
        dict_node_relation_to_subject = {}
        dict_subject_relation_from_node = {}
        dict_node_to_subject = {}
        for i, row in df_item_features.iterrows():
            item_id, feature_id = int(row['item']), int(row['feature'])
            uris = df_feature_uris[df_feature_uris[0] == feature_id][1].values[0].split('><')
            for i in range(int(len(uris) / 2)):
                rel = uris[2*i] if (i+1) == len(uris)/2 else uris[2*i][1:]
                uri = uris[1+2*i][:-1] if (i+1) == len(uris)/2 else uris[1+2*i]
                subject_id = indexer_subjects[uri]
                G.add_edge(item_id, subject_id)
                if (item_id, rel) not in dict_node_relation_to_subject:
                    dict_node_relation_to_subject[(item_id, rel)] = [subject_id]
                else:
                    dict_node_relation_to_subject[(item_id, rel)].append(subject_id)
                dict_node_to_subject[(item_id, subject_id)] = rel

                if (rel, subject_id) not in dict_subject_relation_from_node:
                    dict_subject_relation_from_node[(rel, subject_id)] = [item_id]
                dict_subject_relation_from_node[(rel, subject_id)].append(item_id)

                item_id = subject_id

        print('Start Path Exploration of {}'.format(selection_type))
        start_path_exploration = time.time()
        # Evaluate the Shortest Path with Dijkstra
        hashmap_shortest_paths = {}
        for target_item_id in df_target_items.itemId.tolist():
            start_target_item_id = time.time()
            print("\tTarget Item: {}".format(target_item_id))
            hashmap_shortest_paths[target_item_id] = {}
            for num_item, item in enumerate(df_item_features[df_item_features['item'] != target_item_id].item.unique()):
                # print("\t{}".format(item))
                item = int(item)
                hashmap_shortest_paths[target_item_id][item] = []
                all_simple_paths = nx.shortest_simple_paths(G, target_item_id, item)
                # for num, path in enumerate(reversed_iterator(all_simple_paths)):
                for num, path in enumerate(all_simple_paths):
                    hashmap_shortest_paths[target_item_id][item].append(path)
                    if num == topk:
                        break

                if num_item % 500 == 0 and num_item > 0:
                    print("\t\tExplored: {0}/{1}".format(num_item, len(df_item_features.item.unique()) - 1))

            print("\t--> End Exploration in {}".format(timer(start_target_item_id, time.time())))
        print("--> End Path Exploration of {} in {}".format(selection_type, timer(start_path_exploration, time.time())))

        # Store Object

        save_obj(hashmap_shortest_paths,
                 os.path.join(project_dir, 'data', dataset, 'similarities',
                              'top{0}_dict_{1}_exploration'.format(topk, selection_type)))
        save_obj(indexer_subjects,
                 os.path.join(project_dir, 'data', dataset, 'similarities',
                              'top{0}_subject_uri_indexer_{1}_exploration'.format(topk, selection_type)))

        # Measure Metrics

        ## Katz Relatedness
        if is_katz:
            start_katz = time.time()
            similar_items = evaluate_katz_relatedness(hashmap_shortest_paths, alpha, topk, top_k_similar_items)
            similar_items.to_csv(os.path.join(project_dir, 'data', dataset, 'similarities',
                                              similarities_file.format('katz-a{0}-top{1}'.format(alpha, topk), 'target',
                                                                       selection_type)), index=None)
            print("\n\n{0} Katz relatedness file WRITTEN on {1} in {2}".format(selection_type, dataset, timer(start_katz, time.time())))

        ## Exclusivity-based Relatedness
        if is_exclusivity:
            start_exclusivity = time.time()
            similar_items = evaluate_exclusivity_relatedness(hashmap_shortest_paths, dict_node_to_subject, dict_subject_relation_from_node, dict_node_relation_to_subject, alpha, topk,
                                                             top_k_similar_items)
            similar_items.to_csv(os.path.join(project_dir, 'data', dataset, 'similarities',
                                              similarities_file.format('exclusivity-a{0}-top{1}'.format(alpha, topk), 'target',
                                                                       selection_type)), index=None)
            print("\n\n{0} Exclusivity relatedness file WRITTEN on {1} in {2}".format(selection_type, dataset, timer(start_exclusivity, time.time())))

        print("**** END in {} ****".format(timer(start_type, time.time())))


if __name__ == '__main__':
    build()
