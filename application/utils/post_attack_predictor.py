import pandas as pd
import numpy as np
import random
from surprise import Dataset
from surprise.reader import Reader
from collections import defaultdict
import multiprocessing as mp
import os
from os import listdir
import re
import pickle

from application.utils.manage_dir import check_dir_results
import config as cfg

# Seed For Reproducibility
my_seed = 123
random.seed(my_seed)
np.random.seed(my_seed)

# Global Variables
# df_post_predictions = pd.DataFrame(
#     columns=['sample', 'userId', 'itemId', 'score', 'position'])

dict_post_predictions = {}
full_recommendation_post_prediction = {}
project_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def catch_positions(lista, target):
    try:
        return lista.index(target) + 1
    except ValueError as e:
        return None


def catch_scores(lista, position):
    try:
        return lista[position - 1]
    except TypeError as e:
        return None


def get_rec_list_faster(predictions, target_items, shilling_ids):
    """
    Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
        :param predictions: 
        :param target_items: 
    """

    # First map the predictions to each user.
    top = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        if uid in shilling_ids:
            a = 0
        else:
            top[uid].append((iid, est))

    final_positions = {}
    final_scores = {}

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        list_positions = [ele[0] for ele in user_ratings]
        list_scores = [ele[1] for ele in user_ratings]
        final_positions[int(uid)] = [catch_positions(list_positions, target_item) for target_item in [target_items]]
        final_scores[int(uid)] = [catch_scores(list_scores, position) for position in final_positions[uid]]
        # top[uid] = user_ratings

    # return top, final_positions, final_scores
    return final_positions, final_scores


def get_rec_list(predictions):
    """
    Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    # First map the predictions to each user.
    top = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top[uid] = user_ratings

    return top


def get_algo(train=None):
    """

    :return: Resturn the Algor
    """
    # Set The Model
    if cfg.model == 'SVD':
        from surprise import SVD
        algo = SVD()
    elif cfg.model == 'UserkNN':
        from surprise import KNNBaseline
        sim_options = {'name': 'pearson',
                       'user_based': True  # compute  similarities between users
                       }
        algo = KNNBaseline(sim_options=sim_options, verbose=False)
    elif cfg.model == 'ItemkNN':
        from surprise import KNNBaseline
        sim_options = {'name': 'pearson',
                       'user_based': False  # compute  similarities between items
                       }
        algo = KNNBaseline(sim_options=sim_options, verbose=False)
    elif cfg.model == 'NCF':
        from models.ncf_algo import NCF
        algo = NCF(train)

    return algo


def evaluate_prediction(df, sample_num, dir_attack_profiles, attack):
    """

    :param df:
    :param sample_num:
    :param dir_attack_profiles:
    :param attack:
    :return: the elaborated dataframe and the sample name
    """

    print("\t\t\t\tAttack {0}".format(attack))
    target_item_id = int(attack.split('_')[2].split('.')[0])
    df_attack = pd.read_csv(os.path.join(dir_attack_profiles, attack))

    # Reduce the number of shilling profiles with respect to the maximum
    perc_of_shilling_users = round(cfg.attackSizePercentage / max(cfg.size_of_attacks), 2)
    shilling_users = df_attack.userId.unique()
    df_attack = df_attack[df_attack.userId.isin(shilling_users[:int(len(shilling_users) * perc_of_shilling_users)])]

    shilling_ids = list(df_attack['userId'].unique())

    df_attack = df_attack.append(df, ignore_index=True).reset_index()

    algo = get_algo(df_attack)

    # First train a Recommender Algorithm on the sample dataset.
    if cfg.model in [cfg.ncf]:
        algo.fit()
    else:
        reader = Reader(line_format='user item rating', rating_scale=cfg.rating_scale.get(cfg.dataset))
        data = Dataset.load_from_df(df_attack[['userId', 'itemId', 'rating']], reader)
        trainset = data.build_full_trainset()
        algo.fit(trainset)

    # Than predict ratings for all pairs (u, i) that are NOT in the training set.
    # predictions = algo.test(testset)

    # We are Evaluating on a Single Item
    # rec_list, final_positions, final_scores = get_rec_list_faster(predictions, target_item_id)

    # final_positions, final_scores = get_rec_list_faster(predictions, target_item_id, shilling_ids)
    print('\t\t\t\tEvaluating post prediction')

    if cfg.model in [cfg.ncf]:
        final_positions, final_scores = algo.test([target_item_id], shilling_ids)
    else:
        testset = trainset.build_anti_testset()
        predictions = algo.test(testset)
        final_positions, final_scores = get_rec_list_faster(predictions, target_item_id, shilling_ids)
    print('\t\t\t\tEnd Evaluation of post prediction')

    return final_positions, final_scores, sample_num, target_item_id


def store_predictions(r):
    """
    CallBack Function for store parameters
    :param r: Return Parameter of  evaluate_prediction method
    :return:
    """
    final_positions, final_scores, sample_num, item_id = r
    print("\t\t\t\tStore Post-Attack Prediction on Sample: {0} Item: {1}".format(sample_num, item_id))
    global dict_post_predictions
    # global full_recommendation_post_prediction
    dict_post_predictions[item_id] = {
        'final_positions': final_positions,
        'final_scores': final_scores
    }
    # full_recommendation_post_prediction[item_id] = rec_list


def generate_post_prediction():
    """
    Run in Parallel The Execution of final Prediction
    In this Stage We Will Identify The Items Under Attack for Each Data Samples.
    Note that The output file  is saved in the data_samples directory  (final_predictions.csv)
    The Columns are: 'sample', 'userId', 'itemId', 'score', 'position'
    :return:
    """

    global dict_post_predictions
    global full_recommendation_post_prediction

    # Need to be Restored When We Start a New Execution
    dict_post_predictions = {}
    full_recommendation_post_prediction = {}

    # Load All the Shilling Profile for the specific Attack
    sim_file_name = ""
    if cfg.semantic:
        if cfg.similarity_type == cfg.cosine:
            sim_file_name = 'cosine'
        elif cfg.similarity_type == cfg.katz:
            sim_file_name = 'katz-a{0}-top{1}'.format(cfg.alpha, cfg.topk)
        elif cfg.similarity_type == cfg.exclusivity:
            sim_file_name = 'exclusivity-a{0}-top{1}'.format(cfg.alpha, cfg.topk)

        # Load Semantic
        list_attacks = listdir(os.path.join(project_dir, cfg.model, cfg.shilling_profiles, cfg.dataset,
                                            "{0}_{1}_{2}_{3}_{4}".format(cfg.attack_type,
                                                                         'push' if cfg.push else 'nuke',
                                                                         sim_file_name,
                                                                         cfg.semantic_attack_type, cfg.selection_type)))
    else:
        list_attacks = listdir(os.path.join(project_dir, cfg.model, cfg.shilling_profiles, cfg.dataset,
                                            "{0}_{1}".format(cfg.attack_type, 'push' if cfg.push else 'nuke')))
    # Start Process in Parallel
    pool = mp.Pool(processes=cfg.number_processes)

    check_dir_results(os.path.join(project_dir, cfg.model, cfg.results))

    list_samples = listdir(os.path.join(project_dir, cfg.data_samples, cfg.dataset))
    list_samples.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))

    for path_sample_file in list_samples:

        sample = re.findall(r'\d+', path_sample_file)[0]

        # Load Dataset.
        df = pd.read_csv(os.path.join(project_dir, os.path.join(cfg.data_samples, cfg.dataset, path_sample_file)))

        if cfg.semantic:
            dir_attack_profiles = os.path.join(project_dir, cfg.model, cfg.shilling_profiles, cfg.dataset,
                                               "{0}_{1}_{2}_{3}_{4}".format(cfg.attack_type,
                                                                            'push' if cfg.push else 'nuke',
                                                                            sim_file_name, cfg.semantic_attack_type,
                                                                            cfg.selection_type))
        else:
            dir_attack_profiles = os.path.join(project_dir, cfg.model, cfg.shilling_profiles, cfg.dataset,
                                               "{0}_{1}".format(cfg.attack_type, 'push' if cfg.push else 'nuke'))

        list_attacks_filter = list(filter(lambda x: x.split('_')[1] == sample, list_attacks))
        for attack in list_attacks_filter:
            pool.apply_async(evaluate_prediction, args=(df, sample, dir_attack_profiles, attack,),
                             callback=store_predictions)

    pool.close()
    pool.join()

    if cfg.semantic:
        save_obj(dict_post_predictions, os.path.join(project_dir, cfg.model, cfg.results, cfg.dataset,
                                                     "{0}_{1}_{2}_{3}_{4}_{5}_{6}".format(cfg.attack_type,
                                                                                          'push' if cfg.push else 'nuke',
                                                                                          sim_file_name,
                                                                                          cfg.semantic_attack_type,
                                                                                          cfg.selection_type,
                                                                                          int(
                                                                                              cfg.attackSizePercentage * 100),
                                                                                          cfg.post_prediction)))
        if cfg.save_full_rec_list:
            save_obj(full_recommendation_post_prediction, os.path.join(project_dir, cfg.model, cfg.results, cfg.dataset,
                                                                       "Full_{0}_{1}_{2}_{3}_{4}_{5}_{6}".format(
                                                                           cfg.attack_type,
                                                                           'push' if cfg.push else 'nuke',
                                                                           sim_file_name,
                                                                           cfg.semantic_attack_type,
                                                                           cfg.selection_type,
                                                                           int(
                                                                               cfg.attackSizePercentage * 100),
                                                                           cfg.post_prediction)))
    else:
        save_obj(dict_post_predictions, os.path.join(project_dir, cfg.model, cfg.results, cfg.dataset,
                                                     "{0}_{1}_{2}_{3}_{4}".format(cfg.attack_type,
                                                                                  'push' if cfg.push else 'nuke',
                                                                                  'baseline',
                                                                                  int(
                                                                                      cfg.attackSizePercentage * 100),
                                                                                  cfg.post_prediction)))

        if cfg.save_full_rec_list:
            save_obj(full_recommendation_post_prediction, os.path.join(project_dir, cfg.model, cfg.results, cfg.dataset,
                                                                       "Full_{0}_{1}_{2}_{3}_{4}".format(
                                                                           cfg.attack_type,
                                                                           'push' if cfg.push else 'nuke',
                                                                           'baseline',
                                                                           int(
                                                                               cfg.attackSizePercentage * 100),
                                                                           cfg.post_prediction)))
