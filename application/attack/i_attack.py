from abc import ABCMeta, abstractmethod
import config as cfg
import os

class IAttack:

    def __init__(self, dataframe, r_max, r_min):
        self.dataframe = dataframe
        self.target = cfg.push
        self.r_max = r_max
        self.r_min = r_min
        self.attackSizePercentage = cfg.attackSizePercentage
        self.targetRating = r_max if cfg.push else r_min
        self.fillerRating = int(r_max - r_min)
        self.datasetMean = self.dataframe.rating.mean()
        self.datasetStd = self.dataframe.rating.std()
        self.project_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))


    @classmethod
    def version(self): return "1.0"

    @abstractmethod
    def generate_profile(self, target_item_id, sample): raise NotImplementedError

    @abstractmethod
    def get_filler_items(self, selected, target_item_id): raise NotImplementedError

    @abstractmethod
    def get_selected_items(self, target_item_id): raise NotImplementedError

    @abstractmethod
    def get_filler_size(self): raise NotImplementedError

    @abstractmethod
    def get_selected_size(self): raise NotImplementedError

    @classmethod
    def clamp(self, n, minn=cfg.r_min, maxn =cfg.r_max):
        if n < minn:
            return minn
        elif n > maxn:
            return maxn
        else:
            return n
