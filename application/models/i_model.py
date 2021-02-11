from abc import ABC, abstractmethod
from collections import defaultdict
import os
from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise.reader import Reader
import pandas as pd


class IModel(ABC):


    # def index_dataset(self, trainset):
    #     # Indexing
    #     n_users = trainset['userId'].nunique()
    #     n_items = trainset['itemId'].nunique()
    #
    #     self.users_index = dict(zip(sorted(trainset['userId'].unique()), range(0, n_users)))
    #     self.items_index = dict(zip(sorted(trainset['itemId'].unique()), range(0, n_items)))
    #
    #     trainset['userId'] = trainset['userId'].map(self.users_index)
    #     trainset['itemId'] = trainset['itemId'].map(self.items_index)
    #
    #     self.URM_train = sps.csr_matrix((trainset['rating'], (trainset['userId'], trainset['itemId'])))
    #
    #     self._train = sps.dok_matrix(self.URM_train)
    #     self.n_users, self.n_items = self.URM_train.shape
    #
    #     self._item_indices = np.arange(0, self.n_items, dtype=np.int)
    #     self._user_ones_vector = np.ones_like(self._item_indices)

    @abstractmethod
    def do_something(self):
        print("Some implementation!")

    @abstractmethod
    def train(self):
        print("Train!")

    @abstractmethod
    def predict(self):
        print("Predict!")

    def get_top_n(predictions, n=10):
        '''Return the top-N recommendation for each user from a set of predictions.

        Args:
            predictions(list of Prediction objects): The list of predictions, as
                returned by the test method of an algorithm.
            n(int): The number of recommendation to output for each user. Default
                is 10.

        Returns:
        A dict where keys are user (raw) ids and values are lists of tuples:
            [(raw item id, rating estimation), ...] of size n.
        '''

        # First map the predictions to each user.
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            top_n[uid].append((iid, est))

        # Then sort the predictions for each user and retrieve the k highest ones.
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = user_ratings[:n]

        return top_n
