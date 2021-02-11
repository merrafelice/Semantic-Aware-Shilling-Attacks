import os
import pandas as pd
import multiprocessing as mp
import re
from os import listdir

import config as cfg
from application.utils.manage_dir import check_dir_attack, check_dir_semantic_attack


def monitor_generation(r):
    sample, target_item_id = r
    print("\t\t\t\tEnd generation for Sample {0} ItemID {1}".format(sample, target_item_id))


def generate_shilling_profiles(r_max, r_min):
    """
    Generation of Shilling Profile
    It is important to have defined:
        DATASET
        TYPE of ATTACK
    :return:
    """

    print("\t\t\t\tAttack Type: {0} - {1}".format(cfg.attack_type, 'Push' if cfg.push else 'Nuke'))

    # Read The List of All Selected Target Items for Each Sample Generate in the STEP 2
    project_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))

    # Read From The Target Items File
    target_items = pd.read_csv(os.path.join(project_dir, cfg.data, cfg.dataset, cfg.target_items),
                               usecols=['itemId'])['itemId']

    # Create The Directory of Movielens to Save Shilling Profiles for the Attack
    if cfg.semantic:
        check_dir_semantic_attack(os.path.join(project_dir, cfg.model, cfg.shilling_profiles))
    else:
        check_dir_attack(os.path.join(project_dir, cfg.model, cfg.shilling_profiles))
    # Start Tool For MultiProcessing
    pool = mp.Pool(processes=cfg.number_processes)

    # Read The List of Sample Datasets
    list = listdir(os.path.join(project_dir, 'data_samples', cfg.dataset))
    list.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))

    for path_sample_file in list:  # It is necessary to update for iterate on each datasample

        print("\t\t\t\tStart Shilling Profiles Generation for Sample {0}".format(path_sample_file))

        # Read The Sample Dataset
        df_sample = pd.read_csv(os.path.join(project_dir, cfg.data_samples, cfg.dataset, path_sample_file))

        if cfg.attack_type == cfg.bandwagon:
            from attack.bandwagon_attack import BandWagonAttack
            attack = BandWagonAttack(df_sample, r_max, r_min)
        elif cfg.attack_type == cfg.random:
            from attack.random_attack import RandomAttack
            random_attack = RandomAttack(df_sample, r_max, r_min)
            attack = random_attack
        elif cfg.attack_type == cfg.average:
            from attack.average_attack import AverageAttack
            attack = AverageAttack(df_sample, r_max, r_min)
        elif cfg.attack_type == cfg.love_hate:
            from attack.love_hate_attack import LoveHateAttack
            attack = LoveHateAttack(df_sample, r_max, r_min)
        elif cfg.attack_type == cfg.perfect_knowledge:
            from attack.perfect_knowledge_attack import PerfectKnowledgeAttack
            attack = PerfectKnowledgeAttack(df_sample, r_max, r_min)
        elif cfg.attack_type == cfg.popular:
            from attack.popular_attack import PopularAttack
            attack = PopularAttack(df_sample, r_max, r_min)
        else:
            raise Exception("Shilling Attack NOT Specified")

        # Read Items Under Attack From Item Predictions
        sample = int(re.findall(r'\d+', path_sample_file)[0])
        # data_sample_target_items = target_items[target_items['sample'] == sample].itemId.unique()

        for target_item_id in target_items:
            pool.apply_async(attack.generate_profile, args=(target_item_id, sample,), callback=monitor_generation)
            # pool.apply(attack.generate_profile, args=(target_item_id, sample, ))

    pool.close()
    pool.join()
