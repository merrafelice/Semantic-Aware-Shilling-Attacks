import re
from os import listdir
from scipy import stats

import pandas as pd
import os
import statistics
import numpy as np
import config as cfg
from application.utils.manage_dir import check_dir_metrics_and_plots
from .utils import save_obj, load_obj, save_results_csv


def get_position(string_position):
    """

    :param string_position: '1' -> 0, '2' -> 1, '5' -> 2
    :return: Position in Vector
    """
    if string_position == '1':
        return 0
    elif string_position == '2':
        return 1
    elif string_position == '5':
        return 2


def get_stars(p, attack):
    if p is None:
        # The case of PS
        return ''
    if p[attack] <= 0.001:
        return '***'
    elif p[attack] <= 0.01:
        return '**'
    elif p[attack] <= 0.05:
        return '*'
    else:
        return ''


def get_post_pred_file(post_predictions, analyzed_attack, type_of_attack_similarities, post_pred_file):
    if (len(re.findall(analyzed_attack, post_pred_file)) > 0) \
            and (
            (len(re.findall(type_of_attack_similarities, post_pred_file)) > 0)
            or
            (len(re.findall('baseline', post_pred_file)) > 0)
    ):
        post_predictions[post_pred_file.replace('.pkl', '')] = load_obj(
            os.path.join(cfg.project_dir, cfg.model, cfg.results, cfg.dataset, post_pred_file.replace('.pkl', '')))


def evaluate_hit_ratio(selected_users, target_items_id):
    for model in cfg.models:

        cfg.model = model
        print('EVALUATING Hit Ratio on Model {} for Dataset {}'.format(cfg.model, cfg.dataset))

        results = {}
        results_stddev = {}

        check_dir_metrics_and_plots(os.path.join(cfg.project_dir, cfg.model, cfg.metric))

        for type_of_attack_similarities in [cfg.attack_target_similar]:
            print('\tStart Evaluation Type of Attacks Similarities {0}'.format(type_of_attack_similarities))

            hr_initial = {}

            initial_prediction = load_obj(
                os.path.join(cfg.project_dir, cfg.model, cfg.results, cfg.dataset, cfg.initial_prediction))

            df_target_items_initial_positions = pd.read_csv(
                os.path.join(cfg.project_dir, cfg.data, cfg.dataset, cfg.target_items))

            print('\t\tEvaluate the Metric on the Initial Predictions (Before the Shilling Attack)')
            for index, row in df_target_items_initial_positions.iterrows():
                hr_initial[int(row['itemId'])] = []
                for user_id in selected_users:
                    init_pos, initial_score_of_item = initial_prediction['initial_positions'][user_id][index], \
                                                      initial_prediction['initial_scores'][user_id][index]
                    if init_pos is not None and np.abs(initial_score_of_item) != np.inf:
                        if init_pos <= cfg.top_k_metrics:
                            hr_initial[int(row['itemId'])].append(1)
                        else:
                            hr_initial[int(row['itemId'])].append(0)

            for item_id in hr_initial.keys():
                hr_initial[item_id] = statistics.mean(hr_initial[item_id])
            hr_initial_by_items = hr_initial
            hr_initial = statistics.mean(hr_initial.values())

            print('\t\tCOMPLETED - Evaluate the Metric on the Initial Predictions (Before the Shilling Attack)')

            for analyzed_attack in cfg.attacks:
                print('\t\tEvaluation of Predictions after Attack {}'.format(analyzed_attack))
                list_post_pred = listdir(os.path.join(cfg.project_dir, cfg.model, cfg.results, cfg.dataset))
                post_predictions = {}

                for post_pred_file in list_post_pred:
                    get_post_pred_file(post_predictions, analyzed_attack, type_of_attack_similarities, post_pred_file)

                print('\t\t\t{} Attack Combination Under Evaluations!'.format(len(list(post_predictions.keys()))))

                hr_final = {}
                hr_final_std = {}
                p_values = {}

                for attack in post_predictions.keys():
                    print('\t\t\t Attacks: {}'.format(attack))
                    hr_final[attack] = {}
                    hr_final_std[attack] = {}
                    for item_id in target_items_id:
                        hr_final[attack][item_id] = []
                        item_pp, item_pscore = post_predictions[attack][item_id]['final_positions'], \
                                               post_predictions[attack][item_id]['final_scores']
                        for user_id in item_pp.keys():
                            if user_id in list(selected_users):
                                final_score_of_item = item_pscore[user_id][0]
                                if final_score_of_item is not None and np.abs(final_score_of_item) != np.inf:
                                    final_position_of_item = item_pp[user_id][0]
                                    if final_position_of_item is not None:
                                        # It means that the user did not rate the item
                                        if final_position_of_item <= cfg.top_k_metrics:
                                            # It is in top@k
                                            hr_final[attack][item_id].append(1)
                                        else:
                                            hr_final[attack][item_id].append(0)

                        hr_final[attack][item_id] = statistics.mean(hr_final[attack][item_id])

                    p_values[attack] = stats.ttest_rel(list(hr_initial_by_items.values()),
                                                       list(hr_final[attack].values())).pvalue
                    hr_final[attack], hr_final_std[attack] = statistics.mean(hr_final[attack].values()) \
                        , statistics.stdev(hr_final[attack].values())

                results_name = 'HR{0}_{1}_{2}_{3}_{4}'.format(cfg.top_k_metrics,
                                                              cfg.dataset,
                                                              model,
                                                              analyzed_attack,
                                                              type_of_attack_similarities)

                # Elaborate The Evaluation Metrics
                evaluate_final_results(results_name, results, results_stddev, hr_final, hr_final_std, p_values, hr_initial)

            print('\tEND Evaluation Type of Attacks Similarities {0}'.format(type_of_attack_similarities))

        # Save The Dictionary and CSV with the Results for Both Type of Attacks (Semantic and Baseline) in the ModelName/PS/DatasetName
        save_obj(results, os.path.join(cfg.project_dir, cfg.model, cfg.metric, cfg.dataset,
                                       cfg.results_name.format(cfg.top_k_metrics)))
        save_obj(results_stddev,
                 os.path.join(cfg.project_dir, cfg.model, cfg.metric, cfg.dataset,
                              cfg.results_std_name.format(cfg.top_k_metrics)))

        save_results_csv(results, os.path.join(cfg.project_dir, cfg.model, cfg.metric, cfg.dataset,
                                               cfg.results_name.format(cfg.top_k_metrics)))


def evaluate_prediction_shift(selected_users, target_items_id):
    for model in cfg.models:
        cfg.model = model

        print('EVALUATING Prediction Shifts on Model {} for Dataset {}'.format(cfg.model, cfg.dataset))

        results = {}
        results_stddev = {}

        check_dir_metrics_and_plots(os.path.join(cfg.project_dir, cfg.model, cfg.metric))

        for type_of_attack_similarities in cfg.semantic_attack_types:
            print('\tStart Evaluation Type of Attacks Similarities {0}'.format(type_of_attack_similarities))

            check_dir_metrics_and_plots(os.path.join(cfg.project_dir, cfg.model, cfg.metric))

            print('\t\tEvaluate the Metric on the Initial Predictions (Before the Shilling Attack)')

            initial_prediction = load_obj(
                os.path.join(cfg.project_dir, cfg.model, cfg.results, cfg.dataset, cfg.initial_prediction))

            print('\t\tCOMPLETED - Evaluate the Metric on the Initial Predictions (Before the Shilling Attack)')

            for analyzed_attack in cfg.attacks:
                print('\t\tEvaluation of Predictions after Attack {}'.format(analyzed_attack))
                list_post_pred = listdir(os.path.join(cfg.project_dir, cfg.model, cfg.results, cfg.dataset))
                post_predictions = {}

                for post_pred_file in list_post_pred:
                    get_post_pred_file(post_predictions, analyzed_attack, type_of_attack_similarities,
                                       post_pred_file)

                print('\t\t\t{} Attack Combination Under Evaluations!'.format(len(list(post_predictions.keys()))))

                ps_final = {}
                ps_final_std = {}

                for attack in post_predictions.keys():
                    print('\t\t\t Attacks: {}'.format(attack))
                    ps_final[attack] = {}
                    ps_final_std[attack] = {}
                    for index, item_id in enumerate(target_items_id):
                        ps_final[attack][item_id] = []
                        item_pp = post_predictions[attack][item_id]['final_scores']
                        for user_id in item_pp.keys():
                            if user_id in list(selected_users):
                                final_score_of_item = item_pp[user_id][0]
                                initial_score_of_item = initial_prediction['initial_scores'][user_id][index]
                                if final_score_of_item is not None and np.abs(final_score_of_item) != np.inf:
                                    ps_final[attack][item_id].append(
                                        final_score_of_item - initial_score_of_item)
                                # else:
                                #     # These are the Target Items that are in the Training of the Current User
                                #     print('User: {} T.Item: {} F.score {}, I.score {}'.format(user_id, item_id,
                                #                                                               final_score_of_item,
                                #                                                               initial_score_of_item))

                        ps_final[attack][item_id] = statistics.mean(ps_final[attack][item_id])

                    ps_final[attack], ps_final_std[attack] = statistics.mean(
                        ps_final[attack].values()), statistics.stdev(ps_final[attack].values())

                results_name = 'PS{0}_{1}_{2}_{3}_{4}'.format(cfg.top_k_metrics,
                                                              cfg.dataset,
                                                              model,
                                                              analyzed_attack,
                                                              type_of_attack_similarities)

                # Elaborate The Evaluation Metrics
                evaluate_final_results(results_name, results, results_stddev, ps_final, ps_final_std)

            print('\tEND Evaluation Type of Attacks Similarities {0}'.format(type_of_attack_similarities))

        # Save The Dictionary and CSV with the Results for Both Type of Attacks (Semantic and Baseline) in the ModelName/PS/DatasetName
        save_obj(results, os.path.join(cfg.project_dir, cfg.model, cfg.metric, cfg.dataset,
                                       cfg.results_name.format(cfg.top_k_metrics)))
        save_obj(results_stddev,
                 os.path.join(cfg.project_dir, cfg.model, cfg.metric, cfg.dataset,
                              cfg.results_std_name.format(cfg.top_k_metrics)))

        save_results_csv(results, os.path.join(cfg.project_dir, cfg.model, cfg.metric, cfg.dataset,
                                               cfg.results_name.format(cfg.top_k_metrics)))


def evaluate_final_results(name, results, results_stddev, final, final_std, p_values=None, initial=0):
    # Base Model (No Attacks)
    baseline = [initial, 0, 0, 0]
    baseline_stddev = {}

    # Cosine
    cosine_attack_categorical = [initial, 0, 0, 0]
    cosine_attack_ontological = [initial, 0, 0, 0]
    cosine_attack_factual = [initial, 0, 0, 0]
    cosine_attack_categorical_stddev = {}
    cosine_attack_ontological_stddev = {}
    cosine_attack_factual_stddev = {}

    # Katz
    katz_attack_categorical = [initial, 0, 0, 0]
    katz_attack_ontological = [initial, 0, 0, 0]
    katz_attack_factual = [initial, 0, 0, 0]
    katz_attack_categorical_stddev = {}
    katz_attack_ontological_stddev = {}
    katz_attack_factual_stddev = {}

    # Exclusivity
    exclusivity_attack_categorical = [initial, 0, 0, 0]
    exclusivity_attack_ontological = [initial, 0, 0, 0]
    exclusivity_attack_factual = [initial, 0, 0, 0]
    exclusivity_attack_categorical_stddev = {}
    exclusivity_attack_ontological_stddev = {}
    exclusivity_attack_factual_stddev = {}

    for attack in final.keys():

        if len(re.findall('baseline', attack)) > 0:
            position = get_position(attack.split('_')[3])
            baseline[position + 1] = final[attack]
            baseline_stddev[position + 1] = final_std[attack]

        else:
            position = get_position(attack.split('_')[6])

            if len(re.findall('cosine', attack)) > 0:
                if len(re.findall('categorical', attack)) > 0:
                    cosine_attack_categorical[position + 1] = "{}{}".format(final[attack], get_stars(p_values, attack))
                    cosine_attack_categorical_stddev[position + 1] = final_std[attack]
                elif len(re.findall('ontological', attack)) > 0:
                    cosine_attack_ontological[position + 1] = "{}{}".format(final[attack], get_stars(p_values, attack))
                    cosine_attack_ontological_stddev[position + 1] = final_std[attack]
                elif len(re.findall('factual', attack)) > 0:
                    cosine_attack_factual[position + 1] = "{}{}".format(final[attack], get_stars(p_values, attack))
                    cosine_attack_factual_stddev[position + 1] = final_std[attack]

            elif len(re.findall('katz', attack)) > 0:
                if len(re.findall('categorical', attack)) > 0:
                    katz_attack_categorical[position + 1] = "{}{}".format(final[attack], get_stars(p_values, attack))
                    katz_attack_categorical_stddev[position + 1] = final_std[attack]
                elif len(re.findall('ontological', attack)) > 0:
                    katz_attack_ontological[position + 1] = "{}{}".format(final[attack], get_stars(p_values, attack))
                    katz_attack_ontological_stddev[position + 1] = final_std[attack]
                elif len(re.findall('factual', attack)) > 0:
                    katz_attack_factual[position + 1] = "{}{}".format(final[attack], get_stars(p_values, attack))
                    katz_attack_factual_stddev[position + 1] = final_std[attack]

            elif len(re.findall('exclusivity', attack)) > 0:
                if len(re.findall('categorical', attack)) > 0:
                    exclusivity_attack_categorical[position + 1] = "{}{}".format(final[attack], get_stars(p_values, attack))
                    exclusivity_attack_categorical_stddev[position + 1] = final_std[attack]
                elif len(re.findall('ontological', attack)) > 0:
                    exclusivity_attack_ontological[position + 1] = "{}{}".format(final[attack], get_stars(p_values, attack))
                    exclusivity_attack_ontological_stddev[position + 1] = final_std[attack]
                elif len(re.findall('factual', attack)) > 0:
                    exclusivity_attack_factual[position + 1] = "{}{}".format(final[attack], get_stars(p_values, attack))
                    exclusivity_attack_factual_stddev[position + 1] = final_std[attack]

    results[name], results_stddev[name] = {}, {}

    results[name]['Baseline'] = baseline
    results_stddev[name]['Baseline'] = baseline_stddev

    # Cosine
    results[name]['Cosine'], results_stddev[name]['Cosine'] = {}, {}
    results[name]['Cosine']['Categorical'] = cosine_attack_categorical
    results[name]['Cosine']['Ontological'] = cosine_attack_ontological
    results[name]['Cosine']['Factual'] = cosine_attack_factual
    results_stddev[name]['Cosine']['Categorical'] = cosine_attack_categorical_stddev
    results_stddev[name]['Cosine']['Ontological'] = cosine_attack_ontological_stddev
    results_stddev[name]['Cosine']['Factual'] = cosine_attack_factual_stddev

    # Katz
    results[name]['Katz'], results_stddev[name]['Katz'] = {}, {}
    results[name]['Katz']['Categorical'] = katz_attack_categorical
    results[name]['Katz']['Ontological'] = katz_attack_ontological
    results[name]['Katz']['Factual'] = katz_attack_factual
    results_stddev[name]['Katz']['Categorical'] = katz_attack_categorical_stddev
    results_stddev[name]['Katz']['Ontological'] = katz_attack_ontological_stddev
    results_stddev[name]['Katz']['Factual'] = katz_attack_factual_stddev

    # Exclusivity
    results[name]['Exclusivity'], results_stddev[name]['Exclusivity'] = {}, {}
    results[name]['Exclusivity']['Categorical'] = exclusivity_attack_categorical
    results[name]['Exclusivity']['Ontological'] = exclusivity_attack_ontological
    results[name]['Exclusivity']['Factual'] = exclusivity_attack_factual
    results_stddev[name]['Exclusivity']['Categorical'] = exclusivity_attack_categorical_stddev
    results_stddev[name]['Exclusivity']['Ontological'] = exclusivity_attack_ontological_stddev
    results_stddev[name]['Exclusivity']['Factual'] = exclusivity_attack_factual_stddev

    return results, results_stddev
