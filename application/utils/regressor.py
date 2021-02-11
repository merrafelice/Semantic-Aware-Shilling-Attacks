import statsmodels.api as sm
import pandas as pd
import config as cfg
import os


# CITE
"""
@inproceedings{seabold2010statsmodels,
  title={Statsmodels: Econometric and statistical modeling with python},
  author={Seabold, Skipper and Perktold, Josef},
  booktitle={9th Python in Science Conference},
  year={2010},
}
"""

"""
DV:
    Prediction Shift: The average change in the predicted rating
            for the attacked item before and after the attack
IVs:
    Size
    Shape
    Density
    Gini_User
    Gini_Item
    Rating_StdDev
    Item_Popularity*
    User_Popularity*
"""
project_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
df_results_ops = pd.DataFrame(columns=['coef', 'std err', 't', 'P>|t|', '[0.025', '0.975]', 'experiment'])
df_results_hr = pd.DataFrame(columns=['coef', 'std err', 't', 'P>|t|', '[0.025', '0.975]', 'experiment'])


def regressor_model_ops(log, dataset, model, attack_target, attack, attack_size):
    """
    Build Regressor Model For Each Experiment Given by The Next Identification Variable:
    :param dataset: dataset
    :param model: Model
    :param attack_target: Push Nuke
    :param attack: Type Of Attack
    :param attack_size: Perchentage
    :return:
    """
    # Read Initial Predictions
    initial_prediction = pd.read_csv(os.path.join(project_dir, model, cfg.results, dataset, cfg.initial_prediction))
    data_samples_metrics = pd.read_csv(os.path.join(project_dir, model, cfg.results, dataset, cfg.data_samples_metrics))
    attack_sample = pd.read_csv(os.path.join(project_dir, model, cfg.results, dataset, "{0}_{1}_{2}_{3}".format(attack, 'push' if attack_target else 'nuke', int(attack_size*100), cfg.post_prediction)))
    merge = pd.merge(attack_sample, initial_prediction, on=['sample', 'userId', 'itemId'], how='inner')[['sample', 'userId', 'itemId', 'initial_score', 'score']]
    merge.rename(columns={'score': 'final_score'}, inplace=True)
    # Prediction Shift
    merge['delta_score'] = merge['final_score'] - merge['initial_score']
    prediction_shift = merge.groupby(['sample', 'itemId'])['delta_score'].mean()
    overall_prediction_shift = prediction_shift.groupby(['sample']).mean().reset_index()
    overall_prediction_shift.columns = ['sample', 'overall_prediction_shift']

    # Input Regression
    input_regressor = pd.merge(overall_prediction_shift, data_samples_metrics, on=['sample'], how='inner')

    # Variables
    X = input_regressor[['size', 'shape', 'density',
                         'rating_freq_user', 'rating_freq_item', 'rating_variance']]
    # Center All IVs By Their Means
    X = X.apply(lambda x: x - x.mean())

    y = input_regressor['overall_prediction_shift']

    ## fit a OLS model with intercept on TV and Radio
    X = sm.add_constant(X)

    est = sm.OLS(y, X).fit()

    # Note that tables is a list. The table at index 1 is the "core" table. Additionally, read_html puts dfs in a list, so we want index 0
    results_as_html = est.summary().tables[1].as_html()
    df_result = pd.read_html(results_as_html, header=0, index_col=0)[0]
    df_result['experiment'] = "{0}_{1}_{2}_{3}_{4}".format(dataset, model, attack, 'push' if attack_target else 'nuke', int(attack_size*100))

    # STORE in Global Dataframe
    global df_results_ops
    df_results_ops = df_results_ops.append(df_result)

    text = "*****************************************************************************\n"
    text += "{0}_{1}_{2}_{3}_{4}\n".format(dataset, model, attack, 'push' if attack_target else 'nuke', int(attack_size*100))
    text += est.summary().as_text()
    text += "\n*****************************************************************************\n\n"
    log.write(text)


def regressor_model_hr(log, dataset, model, attack_target, attack, attack_size, k):
    """
    Hit Ratio at K
    Build Regressor Model For Each Experiment Given by The Next Identification Variable:
    :param dataset: dataset
    :param model: Model
    :param attack_target: Push Nuke
    :param attack: Type Of Attack
    :param attack_size: Perchentage
    :return:
    """
    # Read Initial Predictions
    initial_prediction = pd.read_csv(os.path.join(project_dir, model, cfg.results, dataset, cfg.initial_prediction))
    data_samples_metrics = pd.read_csv(os.path.join(project_dir, model, cfg.results, dataset, cfg.data_samples_metrics))
    attack_sample = pd.read_csv(os.path.join(project_dir, model, cfg.results, dataset, "{0}_{1}_{2}_{3}".format(attack, 'push' if attack_target else 'nuke', int(attack_size*100), cfg.post_prediction)))
    merge = pd.merge(attack_sample, initial_prediction, on=['sample', 'userId', 'itemId'], how='inner')[['sample', 'userId', 'itemId', 'initial_position', 'position']]
    merge.rename(columns={'position': 'final_position'}, inplace=True)
    # HR@k
    tot_user = merge.groupby(['sample', 'itemId'])['final_position'].count()
    hr_tot_user = merge[merge.final_position <= k].groupby(['sample', 'itemId'])['final_position'].count()
    hr = hr_tot_user/tot_user
    hr = hr.fillna(0)
    hr = hr.reset_index(name="item_hit_ratio")
    hr = hr.groupby(['sample'])['item_hit_ratio'].mean()
    hr = hr.reset_index(name="hit_ratio")

    # Input Regression
    input_regressor = pd.merge(hr, data_samples_metrics, on=['sample'], how='inner')

    # Variables
    X = input_regressor[['size', 'shape', 'density',
                         'rating_freq_user', 'rating_freq_item', 'rating_variance']]
    # Center All IVs By Their Means
    X = X.apply(lambda x: x - x.mean())

    y = input_regressor['hit_ratio']

    ## fit a OLS model with intercept on TV and Radio
    X = sm.add_constant(X)

    est = sm.OLS(y, X).fit()

    # Note that tables is a list. The table at index 1 is the "core" table. Additionally, read_html puts dfs in a list, so we want index 0
    results_as_html = est.summary().tables[1].as_html()
    df_result = pd.read_html(results_as_html, header=0, index_col=0)[0]
    df_result['experiment'] = "{0}_{1}_{2}_{3}_{4}".format(dataset, model, attack, 'push' if attack_target else 'nuke', int(attack_size*100))

    # STORE in Global Dataframe
    global df_results_hr
    df_results_hr = df_results_hr.append(df_result)

    text = "*****************************************************************************\n"
    text += "{0}_{1}_{2}_{3}_{4}\n".format(dataset, model, attack, 'push' if attack_target else 'nuke', int(attack_size*100))
    text += est.summary().as_text()
    text += "\n*****************************************************************************\n\n"
    log.write(text)


def CUSTOM_regression_analysis(k=10):
    """
    OPS and HR@k
    :param k: For Hit Ratio
    :return:
    """

    with open("CUSTOM.txt", "w+") as log:
        for dataset in cfg.datasets[:1]:
            print("Dataset\t{0}".format(dataset))
            for model in cfg.models[:1]:  # SVD, UserKNN, ItemKNN
                print("\tModel\t{0}".format(model))
                for attack_target in [1, 0][:1]:  # 1 PUSH, 0 NUKE
                    print("\t\tTarget Type\t{0}".format(attack_target))
                    for attack in cfg.attacks:  # Attack Category bandwagon, average, perfect_knowledge, random, love_hate, popular
                        print("\t\t\tAttack Type\t{0}".format(attack))
                        for attack_size in [0.05]:  # 1%    2.5%    5%
                            print("\t\t\t\tSize\t{0}".format(attack_size))
                            regressor_model_hr(log, dataset, model, attack_target, attack, attack_size, 50)
                            # regressor_model_ops(log, dataset, model, attack_target, attack, attack_size)




def execute_regression_analysis(k=10):
    """
    OPS and HR@k
    :param k: For Hit Ratio
    :return:
    """
    # for k in [20, 50]:
    #     global df_results_hr
    #     df_results_hr = pd.DataFrame(columns=['coef', 'std err', 't', 'P>|t|', '[0.025', '0.975]', 'experiment'])
    #     print("Evaluate HR@{0}".format(k))
    #     with open("regressor_tables_HR_{0}.txt".format(k), "w+") as log:
    #         for dataset in cfg.datasets[:1]:
    #             print("Dataset\t{0}".format(dataset))
    #             for model in cfg.models:  # SVD, UserKNN, ItemKNN
    #                 print("\tModel\t{0}".format(model))
    #                 for attack_target in [1, 0][:1]:  # 1 PUSH, 0 NUKE
    #                     print("\t\tTarget Type\t{0}".format(attack_target))
    #                     for attack in cfg.attacks:  # Attack Category bandwagon, average, perfect_knowledge, random, love_hate, popular
    #                         print("\t\t\tAttack Type\t{0}".format(attack))
    #                         for attack_size in cfg.size_of_attacks:  # 1%    2.5%    5%
    #                             print("\t\t\t\tSize\t{0}".format(attack_size))
    #                             regressor_model_hr(log, dataset, model, attack_target, attack, attack_size, k)
    #
    #     df_results_hr.to_csv(os.path.join(project_dir, cfg.final_results.format(cfg.overall_hit_ration_at_k.format(k))),
    #                          index=False)
    #     print("\tEvaluate HR@{0} RESULTS STORED AT PROJECT LEVEL DIRECTORY".format(k))
    #
    print("Evaluate Overall Prediction Shift")
    with open("regressor_tables_OPS.txt", "w+") as log:
        for dataset in cfg.datasets[:1]:
            print("Dataset\t{0}".format(dataset))
            for model in cfg.models: # SVD, UserKNN, ItemKNN
                print("\tModel\t{0}".format(model))
                for attack_target in [1, 0][:1]:  # 1 PUSH, 0 NUKE
                    print("\t\tTarget Type\t{0}".format(attack_target))
                    for attack in cfg.attacks[1:2]:  # Attack Category bandwagon, average, perfect_knowledge, random, love_hate, popular
                        print("\t\t\tAttack Type\t{0}".format(attack))
                        for attack_size in cfg.size_of_attacks: # 1%    2.5%    5%
                            print("\t\t\t\tSize\t{0}".format(attack_size))
                            regressor_model_ops(log, dataset, model, attack_target, attack, attack_size)
    global df_results_ops
    df_results_ops.to_csv(os.path.join(project_dir, cfg.final_results.format(cfg.overall_prediction_shift)), index=False)
    print("\tOverall Prediction Shift RESULTS STORED AT PROJECT LEVEL DIRECTORY")


# def regressor_model_user_item(log, dataset, model, attack_target, attack, attack_size):
#     """
#     Build Regressor Model For Each Experiment Given by The Next Identification Variable:
#     :param dataset: dataset
#     :param model: Model
#     :param attack_target: Push Nuke
#     :param attack: Type Of Attack
#     :param attack_size: Perchentage
#     :return:
#     """
#     # Read Initial Predictions
#     initial_prediction = pd.read_csv(os.path.join(project_dir, model, cfg.results, dataset, cfg.initial_prediction))
#     data_samples_metrics = pd.read_csv(os.path.join(project_dir, model, cfg.results, dataset, cfg.data_samples_metrics))
#     item_data_samples_metrics = pd.read_csv(os.path.join(project_dir, model, cfg.results, dataset, cfg.item_data_samples_metrics))
#     user_data_samples_metrics = pd.read_csv(os.path.join(project_dir, model, cfg.results, dataset, cfg.user_data_samples_metrics))
#     attack_sample = pd.read_csv(os.path.join(project_dir, model, cfg.results, dataset, "{0}_{1}_{2}_{3}".format(attack, 'push' if attack_target else 'nuke', int(attack_size*100), cfg.post_prediction)))
#     merge = pd.merge(attack_sample, initial_prediction, on=['sample', 'userId', 'itemId'], how='inner')[['sample', 'userId', 'itemId', 'initial_position', 'position']]
#     merge = pd.merge(merge, item_data_samples_metrics, on=['sample', 'itemId'], how='inner')
#     merge.rename(columns={'popularity': 'popularity_item', 'log_popularity': 'log_popularity_item'}, inplace=True)
#     merge = pd.merge(merge, user_data_samples_metrics, on=['sample', 'userId'], how='inner')
#     merge.rename(columns={'popularity': 'popularity_user', 'log_popularity': 'log_popularity_user'}, inplace=True)
#     merge.rename(columns={'position': 'final_position'}, inplace=True)
#     # Prediction Shift at Local Level (Only Delta Position)
#     merge['delta_position'] = merge['initial_position'] - merge['final_position']
#
#     # Input Regression
#     input_regressor = pd.merge(merge, data_samples_metrics, on=['sample'], how='inner')
#
#     # Variables
#     X = input_regressor[['size', 'shape', 'density',
#                          'rating_freq_user', 'rating_freq_item', 'rating_variance',
#                          'popularity_item', 'popularity_user']]
#
#     # Center All IVs By Their Means
#     X = X.apply(lambda x: x - x.mean())
#
#     y = input_regressor['delta_position']
#
#     ## fit a OLS model with intercept on TV and Radio
#     X = sm.add_constant(X)
#
#     est = sm.OLS(y, X).fit()
#
#     # Note that tables is a list. The table at index 1 is the "core" table. Additionally, read_html puts dfs in a list, so we want index 0
#     results_as_html = est.summary().tables[1].as_html()
#     df_result = pd.read_html(results_as_html, header=0, index_col=0)[0]
#     df_result['experiment'] = "{0}_{1}_{2}_{3}_{4}".format(dataset, model, attack, 'push' if attack_target else 'nuke', int(attack_size*100))
#
#     # STORE in Global Dataframe
#     global df_results_ops
#     df_results_ops = df_results_ops.append(df_result)
#
#     text = "*****************************************************************************\n"
#     text += "{0}_{1}_{2}_{3}_{4}\n".format(dataset, model, attack, 'push' if attack_target else 'nuke', int(attack_size*100))
#     text += est.summary().as_text()
#     text += "\n*****************************************************************************\n\n"
#     log.write(text)


# def execute_regression_analysis_user_item():
#     with open("regressor_tables_user_item.txt", "w+") as log:
#         for dataset in cfg.datasets:
#             print("Dataset\t{0}".format(dataset))
#             for model in cfg.models: # SVD, UserKNN, ItemKNN
#                 print("\tModel\t{0}".format(model))
#                 for attack_target in [1, 0]:  # 1 PUSH, 0 NUKE
#                     print("\t\tTarget Type\t{0}".format(attack_target))
#                     for attack in cfg.attacks:  # Attack Category bandwagon, average, perfect_knowledge, random, love_hate, popular
#                         print("\t\t\tAttack Type\t{0}".format(attack))
#                         for attack_size in cfg.size_of_attacks: # 1%    2.5%    5%
#                             print("\t\t\t\tSize\t{0}".format(attack_size))
#                             regressor_model_user_item(log, dataset, model, attack_target, attack, attack_size)
#     global df_results_ops
#     df_results_ops.to_csv(os.path.join(project_dir, cfg.final_results_user_item), index=False)
#
#     print("FINAL RESULTS USER-ITEM CHARACTERISTICS STORED AT PROJECT LEVEL DIRECTORY")