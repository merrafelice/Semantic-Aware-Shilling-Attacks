import pandas as pd
import numpy as np
import config as cfg
import os
from .manage_dir import check_dir_data_samples

# Seed For Reproducibility
np.random.seed(123)


def generate_random_samples(n_min=100, n_max=1000):
    """

    :param df: input dataframe
    :param min: Minimum Number of user and items
    :param max: Maximum Number of users and number
    :return: Generation of Samples
    """

    # Read The Dataset

    project_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
    check_dir_data_samples(os.path.join(project_dir, cfg.data_samples))

    df = pd.read_csv(os.path.join(project_dir, cfg.training_file))
    df = df.iloc[:, :3]

    df.columns = ['userId', 'itemId', 'rating']

    n_items = df['itemId'].nunique()
    n_users = df['userId'].nunique()

    for sample_df_num in range(0, cfg.num_data_samples):
        if (sample_df_num+1) % 50 == 0:
            print("\t\tSample {0}/{1}".format(sample_df_num+1, cfg.num_data_samples))
        # Random Shuffle
        df_shuffle = df.sample(frac=1)

        # Randomly generate two integeres
        n_rand_users = np.random.randint(low=min(n_min, n_users), high=min(n_max, n_users), size=1)
        n_rand_items = np.random.randint(low=min(n_min, n_items), high=min(n_max, n_items), size=1)

        # Randomly extraction and Removal of Empty Rows and Columns
        #   users
        users = np.random.choice(np.unique(df_shuffle['userId']), n_rand_users, replace=False)
        #   items
        items = np.random.choice(np.unique(df_shuffle['itemId']), n_rand_items, replace=False)

        # Shuffle
        df_sample = df_shuffle[(df_shuffle['userId'].isin(users)) & (df_shuffle['itemId'].isin(items))]

        df_sample.astype({'itemId': 'int32'})
        # Store
        df_sample.to_csv(os.path.join(project_dir, cfg.data_samples, cfg.dataset, 'sample_{0}.csv'.format(sample_df_num+1)),  index=False)






