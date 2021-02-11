import pandas as pd
import numpy as np
import config as cfg
import os
import multiprocessing as mp

from application.utils.manage_dir import check_dir_data_samples
from application.utils.sendmail import sendmail

# Seed For Reproducibility
# np.random.seed(123)
project_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
n_min = 100
n_max = 2500
# Global
df = pd.DataFrame()
n_data_samples = cfg.num_data_samples  # 600 in PRODUCTION EXPERIMENT

np.random.seed(1234)


def generation(n_users, n_items, data_sample_number):
    """
    n_users in the generazion
    :param n_items:
    :param n_users:
    :param data_sample_number: data_sample number
    :return:
    """

    local_proc_rand_gen = np.random.RandomState()

    # print("\t\t\t Sample {0}".format(data_sample_number))
    global df

    counter = 0
    while True:
        counter += 1
        # print(counter)
        # Random Shuffle    df_sample = pd.DataFrame()  # Initialize The Object

        df_shuffle = df.sample(frac=1)

        # Randomly generate two integers
        n_rand_users = local_proc_rand_gen.randint(low=min(n_min, n_users), high=min(n_max, n_users), size=1)
        n_rand_items = local_proc_rand_gen.randint(low=min(n_min, n_users), high=min(n_items, n_max), size=1)

        # Randomly extraction of Users
        #   users  items
        users = local_proc_rand_gen.choice(np.unique(df_shuffle['userId']), n_rand_users,
                                           replace=False)
        items = local_proc_rand_gen.choice(np.unique(df_shuffle['itemId']), n_rand_items,
                                           replace=False)

        df_temp = df_shuffle[(df_shuffle['userId'].isin(users))]

        df_temp = df_temp[df_temp['itemId'].isin(items)]

        density = len(df_temp) / (df_temp['userId'].nunique() * df_temp['itemId'].nunique())

        if 0.0005 < density < 0.1:  # 0.05% to 10%
            print("\t\tSample {0} - Density {1} - Users {2} - Items {3}".format(data_sample_number, density,
                                                                                df_temp['userId'].nunique(),
                                                                                df_temp['itemId'].nunique()))

            # Store
            df_temp.to_csv(
                os.path.join(project_dir, cfg.data_samples, cfg.dataset,
                             'sample_{0}.csv'.format(data_sample_number)),
                index=False)

            # Exit from the loop
            break

        if counter > 100:
            sendmail('SERVER - Experiment Journal',
                     'PROBLEM IN THE GENERATION OF DATA SAMPLES WITH THE CURRENT METHOD')

    return data_sample_number


def return_generation(i):
    """
    Call Back FUnction
    :param i: data sample number (1 to NUM DS)
    :return:
    """
    global n_data_samples
    n_data_samples = n_data_samples - 1
    if n_data_samples % 50:
        print("\tRemains {0}".format(n_data_samples))
    # print("\t\t End Generation Sample {0}".format(i))


def generate_random_samples():
    """

    :param df: input dataframe
    :param min: Minimum Number of user and items
    :param max: Maximum Number of users and number
    :return: Generation of Samples
    """

    # Read The Dataset

    check_dir_data_samples(os.path.join(project_dir, cfg.data_samples))

    global df

    df = pd.read_csv(os.path.join(project_dir, cfg.data, cfg.dataset, cfg.training_file))
    df = df.iloc[:, :3]

    df.columns = ['userId', 'itemId', 'rating']

    n_users = df['userId'].nunique()
    n_items = df['itemId'].nunique()

    pool = mp.Pool(processes=cfg.number_processes)
    for sample_df_num in range(0, cfg.num_data_samples):
        if (sample_df_num + 1) % 50 == 0:
            print("\t\tStart Generation on Sample {0}".format(sample_df_num + 1))
        pool.apply_async(generation, args=(n_users, n_items, sample_df_num + 1,), callback=return_generation)
        # pool.apply(generation, args=(n_users, n_items, sample_df_num + 1,))

    pool.close()
    pool.join()


def move_samples():
    """

    :param df: input dataframe
    :param min: Minimum Number of user and items
    :param max: Maximum Number of users and number
    :return: Generation of Samples
    """

    # Read The Dataset

    check_dir_data_samples(os.path.join(project_dir, cfg.data_samples))

    global df

    df = pd.read_csv(os.path.join(project_dir, cfg.data, cfg.dataset, cfg.training_file))
    df = df.iloc[:, :3]

    df.columns = ['userId', 'itemId', 'rating']

    # Store
    df.to_csv(
        os.path.join(project_dir, cfg.data_samples, cfg.dataset,
                     'sample_{0}.csv'.format(1)),
        index=False)

    # df_target_items = pd.DataFrame()
    # if cfg.target_quartile_sensitive:
    #     # Identification of Items Under Attack from Quartiles
    #     num = int(df.itemId.nunique() * cfg.item_size)
    #     if num < 5:
    #         num = 4
    #     counts = df.groupby(['itemId']).size().reset_index(name='counts')
    #     # quartiles = counts.counts.quantile([0.25, 0.5, 0.75])
    #
    #     items = {}
    #     quantiles = pd.qcut(counts['counts'], cfg.num_quartiles, duplicates='drop')
    #     quantiles.name = 'quantiles'
    #     counts = pd.DataFrame(counts).join(quantiles)
    #
    #     for quartile_number, quantile in enumerate(counts.quantiles.unique()):
    #         quartile = quartile_number
    #         for number_q, eleId in enumerate(
    #                 list(counts[counts.quantiles == quantile].sample(cfg.num_quartiles_target_items, replace=False).itemId)):
    #             if items.get(eleId) is None:
    #                 items[eleId] = quartile + 1
    #                 df_target_items = df_target_items.append({
    #                     'itemId': eleId,
    #                     'quartile': quartile + 1,
    #                     'sample': 0
    #                 }, ignore_index=True)
    #
    #     if len(items) == 0:
    #         print("************************* ALERT num: {0}**********************".format(num))
    #     else:
    #         print("Store Target Items . LENGTH: {0}".format(df_target_items['itemId'].nunique()))
    #
    # else:
    #     items_id = df['itemId'].unique()
    #     targets = np.random.choice(items_id, cfg.num_target_items, replace=False)
    #     for eleId in targets:
    #         df_target_items = df_target_items.append({
    #             'itemId': eleId,
    #             'quartile': 0,
    #             'sample': 0
    #         }, ignore_index=True)
    #     print("\n******* Store Target Items . LENGTH: {0} *******".format(df_target_items['itemId'].nunique()))
    #
    # filename = os.path.join(project_dir, cfg.data, cfg.dataset, cfg.target_items)
    # df_target_items = df_target_items.astype({'itemId': 'int32'})
    # df_target_items.to_csv(filename, index=False)
