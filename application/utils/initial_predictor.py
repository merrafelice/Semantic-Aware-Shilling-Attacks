import pandas as pd
import numpy as np
import random
from surprise import Dataset
from surprise.reader import Reader
from collections import defaultdict
import multiprocessing as mp
from .manage_dir import check_dir_results
import config as cfg
import os
from os import listdir
import re
import time
from .timer import timer
import pickle

# Seed For Reproducibility
my_seed = 123
random.seed(my_seed)
np.random.seed(my_seed)

# Global Variables
df_initial_predictions = pd.DataFrame(
    columns=['sample', 'userId', 'itemId', 'initial_score', 'initial_position'])
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


def get_rec_list(predictions, target_items):
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
        top[uid].append((iid, est))

    initial_positions = {}
    initial_scores = {}

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        list_positions = [ele[0] for ele in user_ratings]
        list_scores = [ele[1] for ele in user_ratings]
        initial_positions[uid] = [catch_positions(list_positions, target_item) for target_item in target_items]
        initial_scores[uid] = [catch_scores(list_scores, position) for position in initial_positions[uid]]
        # top[uid] = user_ratings

    # return top, initial_positions, initial_scores
    return initial_positions, initial_scores


def get_pandas_rec_list(predictions):
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

    top_df = pd.DataFrame()
    length = len(predictions)
    i = 1
    # First map the predictions to each user.
    start = time.time()
    for uid, iid, true_r, est, _ in predictions:
        top_df = top_df.append({
            'userId': uid,
            'itemId': iid,
            'initial_score': est
        }, ignore_index=True)
        i += 1
        if i % 1000 == 0:
            print("\t\t Pred: {0}\{1} in sec: {2}".format(i, length, timer(start, time.time())))
            start = time.time()

    return top_df


def get_algo(train=None):
    # Set The Model
    if cfg.model == 'SVD':
        from surprise import SVD
        algo = SVD()
    elif cfg.model == 'UserkNN':
        from surprise import KNNBaseline
        sim_options = {'name': 'pearson',
                       'user_based': True  # compute  similarities between users
                       }
        algo = KNNBaseline(sim_options=sim_options)
    elif cfg.model == 'ItemkNN':
        from surprise import KNNBaseline
        sim_options = {'name': 'pearson',
                       'user_based': False  # compute  similarities between items
                       }
        algo = KNNBaseline(sim_options=sim_options)
    elif cfg.model == 'NCF':
        from models.ncf_algo import NCF
        algo = NCF(train)
    return algo


def evaluate_prediction(sample_path, sample_num):
    """
    
    :param sample_path: The Absolute Path of the sample useful to read the data samples csv file 
    :param sample_num: the number of sample under analysis
    :return: the elaborated dataframe and the sample name
    """""
    # Load the dataset (download it if needed).

    df = pd.read_csv(os.path.join(project_dir, sample_path))

    target_items = \
        pd.read_csv(os.path.join(project_dir, cfg.data, cfg.dataset, cfg.target_items),
                    usecols=['itemId'])['itemId'].tolist()
    try:
        algo = get_algo(df)
    except Exception as e:
        print(e)

    # First train a Recommender Algorithm on the sample dataset.
    print("\t\t\t\tFit {0}{1}".format(sample_path, sample_num))
    if cfg.model in [cfg.ncf]:
        algo.fit()
    else:
        reader = Reader(line_format='user item rating', rating_scale=cfg.rating_scale.get(cfg.dataset))
        data = Dataset.load_from_df(df[['userId', 'itemId', 'rating']], reader)
        trainset = data.build_full_trainset()
        algo.fit(trainset)
    print("\t\t\t\tEND - Fit {0}{1}".format(sample_path, sample_num))
    # Than predict ratings for all pairs (u, i) that are NOT in the training set.

    print("\t\t\t\tPredict {0}{1}".format(sample_path, sample_num))
    if cfg.model in [cfg.ncf]:
        initial_positions, initial_scores = algo.test(target_items[:])
    else:
        testset = trainset.build_anti_testset()
        predictions = algo.test(testset)
        initial_positions, initial_scores = get_rec_list(predictions, target_items[:])
    print("\t\t\t\tEND - Predict {0}{1}".format(sample_path, sample_num))

    # rec_list, initial_positions, initial_scores = get_rec_list(predictions, target_items[:])
    print("\t\t\t\tStoring Initial Predictions {0}{1}".format(sample_path, sample_num))

    initial_prediction = {
        'initial_positions': initial_positions,
        'initial_scores': initial_scores
    }
    save_obj(initial_prediction,
             os.path.join(project_dir, cfg.model, cfg.results, cfg.dataset, cfg.initial_prediction))

    # if cfg.save_full_rec_list:
    #     # Save Also FULL REC LIST
    #     save_obj(rec_list,
    #              os.path.join(project_dir, cfg.model, cfg.results, cfg.dataset,
    #                           'Full_{0}'.format(cfg.initial_prediction)))

    print("\t\t\t\tEND - Store Initial Positions {0}{1}".format(sample_path, sample_num))


def store_predictions(r):
    """
    CallBack Function for store parameters
    :param r: Return Parameter of  evaluate_prediction method
    :return:
    """
    df, sample_num = r
    print("\t\t\t\tStore Prediction on Sample: {0}".format(sample_num))
    global df_initial_predictions
    df_initial_predictions = df_initial_predictions.append(df)


def initial_predictor():
    """
    Run in Parallel The Execution of Initial Prediction
    In this Stage We Will Identify The Items Under Attack for Each Data Samples.
    Note that The output file  is saved in the data_samples directory  (initial_predictions.csv)
    The Columns are: sample, quartile, userId, itemId, initial_score, initial_position
    :return:
    """
    global df_initial_predictions

    # Need to be Restored
    df_initial_predictions = pd.DataFrame(
        columns=['sample', 'userId', 'itemId', 'initial_score', 'initial_position'])

    # Start Process in Parallel
    pool = mp.Pool(processes=cfg.number_processes)
    check_dir_results(os.path.join(project_dir, cfg.model, cfg.results))

    list = listdir(os.path.join(project_dir, cfg.data_samples, cfg.dataset))
    list.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
    for path_sample_file in list:
        pool.apply_async(evaluate_prediction, args=(
            os.path.join(cfg.data_samples, cfg.dataset, path_sample_file), re.findall(r'\d+', path_sample_file)[0],))

    pool.close()
    pool.join()
