def get_size(n_users, n_items):
    """
    A Size Metric
    Size = |U|*|I|
    :return: The Size
    """
    return n_users*n_items


def get_shape(n_users, n_items):
    """
    Shape: Useful to understand if the n of users is greater, or less, than the n of items
    $\log_{10}(|U|/|I|)$
    :param df: Data Sample
    :return: The Shape of the Data Samples
    """
    return n_users/n_items


def get_density(n_users, n_items, n_ratings):
    """
    $\dfrac{|R|}{|U|*|I|}$
    :param df: Data Samples
    :return: The Density
    """
    return n_ratings/(n_users*n_items)


def get_frequency_user(df_sample):
    """
    1 - 2\sum_{i=1}^{n} (\dfrac{n+1-i}{n+1}))(x_{i}/totalRatings)
    :param df_sample:
    :return: Gini Index Referred to the User
            Gini = 0 All Users Are Equally Popular
            Gini = 1 One BestSelling User Gives All The RATINGS
    """
    # Ascending Order for Gini Index
    df_sample_asc_ordered = df_sample.groupby(['userId']).size().reset_index(name='counts').sort_values(
        ['counts']).reset_index()
    n = df_sample_asc_ordered.userId.nunique()
    total_ratings = df_sample_asc_ordered.counts.sum()
    summation = 0
    for index, row in df_sample_asc_ordered.iterrows():
        summation += (n + 1 - (index + 1)) / (n + 1) * (row['counts'] / total_ratings)
    return 1 - 2 * summation


def get_frequency_item(df_sample):
    """
    1 - 2\sum_{i=1}^{n} (\dfrac{n+1-i}{n+1}))(x_{i}/totalRatings)
    :param df_sample:
    :return: Gini Index Referred to the Item:
            Gini = 0 All Items Are Equally Popular
            Gini = 1 One BestSelling Item Has All The RATINGS
    """
    # Ascending Order for Gini Index
    df_sample_asc_ordered = df_sample.groupby(['itemId']).size().reset_index(name='counts').sort_values(['counts']).reset_index()
    n = df_sample_asc_ordered.itemId.nunique()
    total_ratings = df_sample_asc_ordered.counts.sum()
    summation = 0
    for index, row in df_sample_asc_ordered.iterrows():
            summation += (n+1-(index+1))/(n+1)*(row['counts']/total_ratings)
    return 1 - 2*summation


def get_rating_variance(df_ratings):
    """

    :param df:
    :return: Return The Standard Daviation of Rating Values
    """
    return df_ratings.std()


def get_item_popularity(df):
    """
    $ |Rating on the Item| $
    :param df:
    :return:
    """
    return 0


def get_user_popularity(df):
    """
    $ |Rating on the User| $
    :param df:
    :return:
    """
    return 0
