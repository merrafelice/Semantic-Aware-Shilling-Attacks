from .i_attack import IAttack
import config as cfg
import pandas as pd
import os
import numpy as np
import random

# Seed For Reproducibility
my_seed = 123
np.random.seed(my_seed)
random.seed(my_seed)


class PopularAttack(IAttack):

    def __init__(self, dataframe, r_max, r_min):
        super(PopularAttack, self).__init__(dataframe, r_max, r_min)
        self.selectedSize = self.get_selected_size()
        self.attackSize = self.get_attack_size()

    def generate_profile(self, target_item_id, sample):

        start_shilling_user_id = max(list(self.dataframe.userId.unique()))
        shilling_profiles = pd.DataFrame(columns=list(self.dataframe.columns))

        dict_selected_items = self.get_selected_items(target_item_id)

        for i in range(self.attackSize):

            start_shilling_user_id += 1

            # Rating equal to MAX/min SCORE
            for selected_item_id, rating in dict_selected_items.items():
                shilling_profiles = shilling_profiles.append({
                    'userId': start_shilling_user_id,
                    'itemId': selected_item_id,
                    'rating': rating
                }, ignore_index=True)

            # ADD FILLER:   EMPTY

            # ADD TARGET ITEM with Rating (Max for Push/mn for Nuke)
            shilling_profiles = shilling_profiles.append({
                'userId': start_shilling_user_id,
                'itemId': target_item_id,
                'rating': self.targetRating
            }, ignore_index=True)

        # Save File Of Shilling Profile in the Directory shilling_profiles -> dataste_name -> attack_name -> sample_NUMBER_TARGETITEM.csv
        file_name = "sample_{0}_{1}.csv".format(sample, target_item_id)
        shilling_profiles.to_csv(os.path.join(self.project_dir, cfg.model, cfg.shilling_profiles, cfg.dataset,
                                              "{0}_{1}".format(cfg.attack_type, 'push' if cfg.push else 'nuke'),
                                              file_name), index=False)
        return sample, target_item_id

    def get_filler_size(self):
        """
        |I_{F}|= 0
        :return: Filler Size
        """
        fillerSize = int((self.dataframe.shape[0] / self.dataframe.userId.nunique()) / 2)
        return fillerSize

    def get_selected_size(self):
        """
        |I_{S}|=\(\frac{\sum_{u \in U} | I_{u |}}{|U|}) - 1
        :return: Selected Size
        """
        selectedsize = int((self.dataframe.shape[0] / self.dataframe.userId.nunique()) - 1)
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
        :return: list of filler items EMPTY
        """
        return []

    def get_selected_items(self, target_item_id):
        """
            Push;:
                if  mean(sel_item) < overall_mean
                    ---> r = MIN  # LOW RATED POPULAR ITEMS
                else
                    --->  r = MIN + 1

            Nuke
                if  mean(sel_item) < overall_mean
                    ---> r = MAX - 1
                else
                    --->  r = MIN
        :return: Dictionary of Selected Items: MOST POPULAR ITEMS + ratings
        """

        overall_mean_rating = self.dataframe.rating.mean()
        popular_list = list(
            self.dataframe[self.dataframe.itemId != target_item_id].groupby(['itemId']).size().reset_index(
                name='counts').sort_values(by=['counts'], ascending=False).itemId)[:self.attackSize]

        list_items = dict()
        for selected_item_id in popular_list:
            low_rated_item = self.dataframe[
                                 self.dataframe.itemId == selected_item_id].rating.mean() < overall_mean_rating

            rating = (cfg.r_min if low_rated_item else (cfg.r_min + 1)) if self.target else (
                (cfg.r_max - 1) if low_rated_item else cfg.r_max)

            list_items[selected_item_id] = rating

        return list_items
