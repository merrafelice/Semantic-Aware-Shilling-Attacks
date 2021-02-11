import os

import pandas as pd
import numpy as np
import config as cfg

from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.metrics.pairwise import cosine_similarity, check_pairwise_arrays


project_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
df_features = pd.DataFrame()


# def asymmetric_cosine_similarity(x, y, asymmetric_alpha=0.5):
#     """
#         This parameterized variant of the cosine similarity
#         was described in [2], by the winning team of the Million Songs Dataset(MSD) challenge.
#         The paper shows that the Cosine similarity can be
#         represented as a product of the square roots of two conditional prob-abilities
#     :param x:
#     :param y:
#     :param asymmetric_alpha     Coefficient alpha for the asymmetric cosine
#     :return:
#     """
#     def square_rooted(x):
#         return np.round((sum([a * a for a in x])) ** (0.5), 3)
#
#     numerator = np.sum(a * b for a, b in zip(x, y))
#     denominator = (square_rooted(x)**asymmetric_alpha) * (square_rooted(y)**(1-asymmetric_alpha))
#     return np.round(np.sum(numerator) / float(denominator), 3)
#
# def evaluate_asymmetric_cosine_similarity(x, mat_y):
#     sims = []
#     for y in mat_y.toarray():
#         sims.append(asymmetric_cosine_similarity(x.toarray(),
#                                                  y))
#     return sims

def asymmetric_cosine_similarity(X, Y=None, asymmetric_alpha=0.5, dense_output=True):
    """Compute cosine similarity between samples in X and Y.

    Cosine similarity, or the cosine kernel, computes similarity as the
    normalized dot product of X and Y:

        K(X, Y) = <X, Y> / (||X||*||Y||)

    On L2-normalized data, this function is equivalent to linear_kernel.

    Read more in the :ref:`User Guide <cosine_similarity>`.

    Parameters
    ----------
    X : ndarray or sparse array, shape: (n_samples_X, n_features)
        Input data.

    Y : ndarray or sparse array, shape: (n_samples_Y, n_features)
        Input data. If ``None``, the output will be the pairwise
        similarities between all samples in ``X``.

    dense_output : boolean (optional), default True
        Whether to return dense output even when the input is sparse. If
        ``False``, the output is sparse if both input arrays are sparse.

        .. versionadded:: 0.17
           parameter ``dense_output`` for dense output.

    Returns
    -------
    kernel matrix : array
        An array with shape (n_samples_X, n_samples_Y).
    """
    # to avoid recursive import

    X, Y = check_pairwise_arrays(X, Y)

    X_normalized = normalize(X, copy=True)
    if X is Y:
        Y_normalized = X_normalized
    else:
        Y_normalized = normalize(Y, copy=True)

    K = safe_sparse_dot(X_normalized, Y_normalized.T,
                        dense_output=dense_output)

    return K


def evaluate_similarities():
    """

    :return:
    """
    global project_dir

    df_map = pd.read_csv(os.path.join(project_dir, cfg.data, cfg.dataset, 'df_map.csv'))
    df_ratings = pd.read_csv(os.path.join(project_dir, cfg.data, cfg.dataset, cfg.training_file))
    df_ratings.columns = ['userId', 'itemId', 'rating']

    df_map = df_map[df_map['item'].isin(df_ratings['itemId'].unique())]

    from ast import literal_eval
    df_selected_features = pd.read_csv(os.path.join(project_dir, cfg.data, cfg.dataset, cfg.selected_features))

    for selection_type in cfg.selection_types:
        df_item_features = pd.DataFrame()
        if selection_type == 'full':
            df_item_features = df_map.pivot_table(
                index='item',
                columns='feature',
                values='value'
            ).fillna(0)
        elif selection_type == 'categorical':
            df_item_features = df_map[df_map['feature'].isin(literal_eval(df_selected_features[df_selected_features['type'] == 'categorical']['features'].values[0]))].pivot_table(
                index='item',
                columns='feature',
                values='value'
            ).fillna(0)
        elif selection_type == 'ontological':
            df_item_features = df_map[df_map['feature'].isin(literal_eval(df_selected_features[df_selected_features['type'] == 'ontological']['features'].values[0]))].pivot_table(
                index='item',
                columns='feature',
                values='value'
            ).fillna(0)
        elif selection_type == 'factual':
            # FS = FULL - OS
            df_item_features = df_map[df_map['feature'].isin(literal_eval(df_selected_features[df_selected_features['type'] == 'factual']['features'].values[0]))].pivot_table(
                index='item',
                columns='feature',
                values='value'
            ).fillna(0)

        mat_item_features = csr_matrix(df_item_features.values)

        # NOTE THAT ALL THE MODEL ARE ATTACKING THE SAME ITEM SER FOR EACH DATASET
        target_items = \
            pd.read_csv(os.path.join(project_dir, cfg.data, cfg.dataset, cfg.target_items),
                        usecols=['itemId'])['itemId']

        item_index_in_csr = pd.DataFrame(df_item_features.reset_index()['item'])

        print("\t\t\tStart Cosine")
        similar_items = pd.DataFrame()

        for target_item in target_items:
            index = item_index_in_csr[item_index_in_csr['item'] == target_item].index[0]
            similarities_for_item = cosine_similarity(mat_item_features.getrow(index), mat_item_features)
            top_similar_index_ids = similarities_for_item.argsort()[0][::-1][1:int(len(similarities_for_item.argsort()[0])*cfg.top_k_similar_items)]
            similar_items = similar_items.append({
                'itemId': int(target_item),
                'similar_items': [int(item) for item in
                                  list(item_index_in_csr.iloc[list(top_similar_index_ids)].item)],
                'score': np.sort(similarities_for_item)[0][::-1][1:int(len(similarities_for_item.argsort()[0])*cfg.top_k_similar_items)]
            }, ignore_index=True)

        similar_items.to_csv(os.path.join(project_dir, cfg.data, cfg.dataset, 'similarities',
                                          cfg.similarities_file.format('cosine', 'target', selection_type)), index=None)

        # Take also Most Popular Items
        df_popular_items = pd.read_csv(os.path.join(project_dir, cfg.data, cfg.dataset, cfg.training_file), header=None)
        df_popular_items = df_popular_items.iloc[:, :3]
        df_popular_items.columns = ['userId', 'itemId', 'rating']
        df_popular_items = df_popular_items.groupby(['itemId']).size().reset_index(name='counts')
        df_popular_items = df_popular_items.sort_values('counts', axis=0, ascending=False)
        popular_items = df_popular_items['itemId'].iloc[:100].to_list()

        del df_popular_items

        for popular_item in popular_items:
            index = item_index_in_csr[item_index_in_csr['item'] == popular_item].index[0]
            similarities_for_item = cosine_similarity(mat_item_features.getrow(index), mat_item_features)
            top_similar_index_ids = similarities_for_item.argsort()[0][::-1][1:int(len(similarities_for_item.argsort()[0])*cfg.top_k_similar_items)]
            similar_items = similar_items.append({
                'itemId': int(target_item),
                'similar_items': [int(item) for item in
                                  list(item_index_in_csr.iloc[list(top_similar_index_ids)].item)],
                'score': np.sort(similarities_for_item)[0][::-1][1:int(len(similarities_for_item.argsort()[0])*cfg.top_k_similar_items)]
            }, ignore_index=True)

        similar_items.to_csv(os.path.join(project_dir, cfg.data, cfg.dataset, 'similarities',
                                          cfg.similarities_file.format('cosine', 'popular', selection_type)), index=None)

        print("\t\t\tEnd Cosine")

        ##############################################


        print("\n{0} Similarities file WRITTEN on {1}".format(selection_type, cfg.dataset))
