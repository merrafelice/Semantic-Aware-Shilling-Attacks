from surprise import AlgoBase
import numpy as np
import pandas as pd
import tensorflow as tf

from application.utils.indexers import ColumnIndexer
from application.models.ncf import NeuralCF
from application.models.ncf_config import config


class NCF(AlgoBase):

    def __init__(self, train):
        # Parameter Of The Model
        # Always call base method before doing anything.
        AlgoBase.__init__(self)
        self.training_args = config['training_args']

        # instantiate the column index for both user and items
        self.indexer = ColumnIndexer(train, ['userId', 'itemId'])

        # index the train set
        self.train = self.indexer.transform(train)

        # get the number of distinct users and items
        self.number_of_users = len(set(train['userId'].values))
        self.number_of_items = len(set(train['itemId'].values))

        # create user item rating tuples
        train_users_items_ratings = ((
                                         train['userId' + '_indexed'].values,
                                         train['itemId' + '_indexed'].values),
                                     train['rating'].values)

        # instantiate the tf datasets
        self.train_dataset = tf.data.Dataset.from_tensor_slices(
            train_users_items_ratings)

        self.ncf = NeuralCF(self.number_of_users, self.number_of_items, self.training_args.user_dim,
                            self.training_args.item_dim, self.training_args.hidden1_dim,
                            self.training_args.hidden2_dim, self.training_args.hidden3_dim, self.training_args.hidden4_dim)
        self.ncf.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                         loss=tf.keras.losses.MeanAbsoluteError())

    def fit(self):
        """
        The fit method is called e.g. by the cross_validate function at each fold of a cross-validation process
        :param trainset: Training Dataset
        :return:
        """
        print("\t\t\t\t Inside New FIT")
        train_batches = self.train_dataset.shuffle(1000).batch(self.training_args.batch_size).prefetch(1)

        self.ncf.fit(train_batches, epochs=self.training_args.num_epochs)

    def test(self, target_items=None, shilling_ids=[]):

        print('\t\t\t\t**** START The FULL PREDICTION OF NCF ******')

        full_predictions = self.ncf.predict_all()

        predictions = np.reshape(full_predictions.numpy(), newshape=(self.number_of_users, self.number_of_items))
        predictions *= -1  # To Simplify the Ordering

        # position_predictions = predictions.argsort(axis=1)
        # predictions.sort(axis=1)

        print('\t\t\t\t**** Queried The TF MODEL The FULL PREDICTION OF NCF ******')

        positions = {}
        scores = {}

        # Convert Shilling IDS into new indexer
        shilling_ids = [self.indexer.indexers['userId'][shilling_id] for shilling_id in shilling_ids]

        for u in range(self.number_of_users):
            if u not in shilling_ids:
                # Remove the Training Data
                predictions[u][self.train[self.train['userId_indexed'] == u]['itemId_indexed'].tolist()] = np.inf

                # Sort Predictions
                position_predictions = np.argsort(predictions[u])

                # Score the Target Item Position
                positions[self.indexer.reverse_indexers['userId'][u]] = [
                    position_predictions.tolist().index(self.indexer.indexers['itemId'][target_item]) + 1 for
                    target_item in target_items]

                # Get the Score
                scores[self.indexer.reverse_indexers['userId'][u]] = [np.sort(predictions[u]).tolist()[position - 1] * -1 for
                                                                              position in positions[
                                                                                  self.indexer.reverse_indexers[
                                                                                      'userId'][u]]
                                                                      ]

                # for i, s in enumerate(scores[self.indexer.reverse_indexers['userId'][u]]):
                #     if np.abs(s) == np.inf:
                #         if target_items[i] not in self.train[self.train['userId_indexed'] == u]['itemId'].tolist():
                #             print('Item {} - Score {} - User {}'.format(target_items[i], s, u))

            if u % 1000 == 0:
                print("\t\t\t\t\tUser {}/{}".format(u, self.number_of_users))

        print('\t\t\t\t**** END The FULL PREDICTION OF NCF ******')

        return positions, scores

    def test_batch(self, testset):

        test = pd.DataFrame(columns=['userId', 'itemId', 'rating'])

        for i, (uid, iid, rating) in enumerate(testset):
            test.loc[i] = [uid, iid, rating]

            if i == 31:
                break

        test = self.indexer.transform(test)

        test_users_items_ratings = ((test['userId' + '_indexed'].values,
                                     test['itemId' + '_indexed'].values),
                                    test['rating'].values)

        test_dataset = tf.data.Dataset.from_tensor_slices(test_users_items_ratings)

        test_batches = test_dataset.batch(self.training_args.batch_size).prefetch(1)

        return self.ncf.predict(test_batches)
