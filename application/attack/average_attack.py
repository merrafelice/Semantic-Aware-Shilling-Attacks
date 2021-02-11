from .i_attack import IAttack
import config as cfg
import pandas as pd
import os
import numpy as np
import random
from ast import literal_eval


# Seed For Reproducibility
my_seed = 123
np.random.seed(my_seed)
random.seed(my_seed)


class AverageAttack(IAttack):

    def __init__(self, dataframe, r_max, r_min):
        super(AverageAttack, self).__init__(dataframe, r_max, r_min)
        self.fillerSize = self.get_filler_size()
        self.selectedSize = self.get_selected_size()
        self.attackSize = self.get_attack_size()
        self.popular_items = list(self.dataframe.groupby('itemId').size().reset_index(name='count').sort_values(by='count', ascending=False )['itemId'])[:10]

    def generate_profile(self, target_item_id, sample):

        start_shilling_user_id = max(list(self.dataframe.userId.unique()))
        shilling_profiles = pd.DataFrame(columns=list(self.dataframe.columns))

        for i in range(self.attackSize):
            start_shilling_user_id += 1

            # ADD SELECTED: Will Be Empty
            selected_items = self.get_selected_items(target_item_id)

            # ADD FILLER:
            #               Average: Mean and Variance of the Filler Item

            filler_items = self.get_filler_items(selected_items, target_item_id)
            for filler_item_id in filler_items:
                shilling_profiles = shilling_profiles.append({
                    'userId': start_shilling_user_id,
                    'itemId': filler_item_id,
                    'rating': self.clamp(int(np.random.normal(
                        self.dataframe[self.dataframe.itemId == filler_item_id].rating.mean(),
                        self.dataframe[self.dataframe.itemId == filler_item_id].rating.std(ddof=0), 1).round()[0]))
                }, ignore_index=True)

            # ADD TARGET ITEM with Rating (Max for Push/mn for Nuke)
            shilling_profiles = shilling_profiles.append({
                'userId': start_shilling_user_id,
                'itemId': target_item_id,
                'rating': self.targetRating
            }, ignore_index=True)

        # Save File Of Shilling Profile in the Directory shilling_profiles -> dataste_name -> attack_name -> sample_NUMBER_TARGETITEM.csv
        file_name = "sample_{0}_{1}.csv".format(sample, int(target_item_id))
        if cfg.semantic:

            sim_file_name = ""
            if cfg.similarity_type == cfg.cosine:
                sim_file_name = 'cosine'
            elif cfg.similarity_type == cfg.katz:
                sim_file_name = 'katz-a{0}-top{1}'.format(cfg.alpha, cfg.topk)
            elif cfg.similarity_type == cfg.exclusivity:
                sim_file_name = 'exclusivity-a{0}-top{1}'.format(cfg.alpha, cfg.topk)

            shilling_profiles.to_csv(os.path.join(self.project_dir, cfg.model, cfg.shilling_profiles, cfg.dataset,
                                                  "{0}_{1}_{2}_{3}_{4}".format(cfg.attack_type,
                                                                               'push' if cfg.push else 'nuke',
                                                                               sim_file_name,
                                                                               cfg.semantic_attack_type,
                                                                               cfg.selection_type), file_name),
                                     index=False)
        else:
            shilling_profiles.to_csv(os.path.join(self.project_dir, cfg.model, cfg.shilling_profiles, cfg.dataset,
                                                  "{0}_{1}".format(cfg.attack_type,
                                                                   'push' if cfg.push else 'nuke'), file_name),
                                     index=False)

        return sample, target_item_id

    def get_filler_size(self):
        """
        |I_{F}|= \frac{\sum_{u \in U} | I_{u |}}{|U|})} - 1
        :return: Filler Size
        """
        fillerSize = int(self.dataframe.shape[0] / self.dataframe.userId.nunique() - 1)
        return fillerSize

    def get_selected_size(self):
        """
        |I_{S}|= 0
        :return: Selected Size
        """
        selectedsize = 0
        return selectedsize

    def get_attack_size(self):
        """
        :return: The number of Added Profiles (A Percentage of The Users in The Data Sample)
        """
        attackSize = int(self.dataframe.userId.nunique() * self.attackSizePercentage)
        return attackSize

    def get_filler_items(self, selectedItems, target_item_id):
        """

        :param selectedItems: List of Already Selected Items
        :return: list of filler items RANDOMLY CHOSEN
        """
        selectedItems.append(target_item_id)

        if cfg.semantic:
            # Get File Name
            sim_file_name = ""
            target_type = 'popular' if cfg.semantic_attack_type == cfg.attack_popular_similar else 'target'
            if cfg.similarity_type == cfg.cosine:
                sim_file_name = cfg.similarities_file.format('cosine', target_type, cfg.selection_type)
            elif cfg.similarity_type == cfg.katz:
                sim_file_name = cfg.similarities_file.format('katz-a{0}-top{1}'.format(cfg.alpha, cfg.topk),
                                                             target_type,
                                                             cfg.selection_type)
            elif cfg.similarity_type == cfg.exclusivity:
                sim_file_name = cfg.similarities_file.format('exclusivity-a{0}-top{1}'.format(cfg.alpha, cfg.topk),
                                                             target_type, cfg.selection_type)

            # Semantic based Implementation
            if cfg.semantic_attack_type == cfg.attack_popular_similar:
                df_similar_items = pd.read_csv(os.path.join(self.project_dir, cfg.data, cfg.dataset, cfg.similarities,
                                                            sim_file_name))
                possible_filler_items = literal_eval(
                    df_similar_items[
                        df_similar_items['itemId'] == np.random.choice(self.popular_items, 1, replace=False)[0]][
                        'similar_items'].values[0])
            elif cfg.semantic_attack_type == cfg.attack_target_similar:
                df_similar_items = pd.read_csv(os.path.join(self.project_dir, cfg.data, cfg.dataset, cfg.similarities,
                                                            sim_file_name))
                possible_filler_items = literal_eval(df_similar_items[df_similar_items['itemId'] == target_item_id]['similar_items'].values[0])

            for selected_element in selectedItems:
                if possible_filler_items.count(selected_element) > 0:
                    possible_filler_items.remove(selected_element)
                    # Test With Random Selection between TOP 200
            # items = possible_filler_items[:self.fillerSize]
            items = list(np.random.choice(possible_filler_items, self.fillerSize, replace=False))

        else:
            # Basic Attack
            items = self.dataframe.itemId.unique()
            items = items[~np.isin(items, selectedItems)]
            items = random.choices(items, k=self.fillerSize)

        return items

    def get_selected_items(self, target_item_id):
        """

        :return: List of Selected Items: EMPTY
        """
        return []
