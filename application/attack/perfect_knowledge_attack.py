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


class PerfectKnowledgeAttack(IAttack):

    def __init__(self, dataframe, r_max, r_min):
        super(PerfectKnowledgeAttack, self).__init__(dataframe, r_max, r_min)
        self.attackSize = self.get_attack_size()

    def generate_profile(self, target_item_id, sample):

        start_shilling_user_id = max(list(self.dataframe.userId.unique()))
        shilling_profiles = pd.DataFrame(columns=list(self.dataframe.columns))

        users = np.random.choice(self.dataframe.userId.unique(), self.attackSize, replace=False)
        for user_id in users:
            start_shilling_user_id += 1

            items = self.dataframe[self.dataframe.userId == user_id]
            for index, row in items.iterrows():
                if row['itemId'] != start_shilling_user_id:
                    shilling_profiles = shilling_profiles.append({
                        'userId': start_shilling_user_id,
                        'itemId': row['itemId'],
                        'rating': row['rating']
                    }, ignore_index=True)

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
        :return: Filler Size
        """
        return []

    def get_selected_size(self):
        """
        :return: Selected Size
        """
        return 0

    def get_attack_size(self):
        """
        :return: The number of Added Profiles (A Percentage of The Users in The Data Sample)
        """
        attackSize = int(self.dataframe.userId.nunique() * self.attackSizePercentage)
        return attackSize

    def get_filler_items(self, selectedItems, target_item_id):
        """
        :return: list of filler items: EMPTY
        """
        return []

    def get_selected_items(self, target_item_id):
        """
        :return: List of Selected Items: EMPTY
        """
        return []
