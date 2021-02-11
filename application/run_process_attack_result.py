import sys
import os
import pandas as pd
import time
import statistics

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
project_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print(os.getcwd())

from application.parsers.run_mul import parse_run_process, print_args
from application.parsers.config_update import config_update_run_process
from application.utils.attack_metrics import evaluate_hit_ratio, evaluate_prediction_shift
from application.utils.timer import timer

import config as cfg


def run():
    args = parse_run_process()
    config_update_run_process(cfg, args, project_dir)
    print_args(args)

    for dataset in cfg.datasets:
        cfg.dataset = dataset
        start_dataset_process_res = time.time()
        print('Start Results Processing For {}'.format(cfg.dataset))

        # Build User Class
        df = pd.read_csv(os.path.join(project_dir, cfg.data, cfg.dataset, cfg.training_file))
        df = df.iloc[:, :3]
        df.columns = ['userId', 'itemId', 'rating']

        counts = df.groupby('userId').size().reset_index(name='count').sort_values(['count'], ascending=False)
        selected_users = counts['userId'].unique()

        df_target_items = pd.read_csv(os.path.join(cfg.project_dir, cfg.data, cfg.dataset, cfg.target_items))
        target_items_id = df_target_items['itemId'].astype('int32').to_list()
        for metric in cfg.metrics:
            cfg.metric = metric
            if metric == 'HR':
                evaluate_hit_ratio(selected_users, target_items_id)
            elif metric == 'PS':
                evaluate_prediction_shift(selected_users, target_items_id)
            else:
                print('The {0} metric is not implemented!'.format(metric))

        end_dataset_process_res = time.time()
        print('END Results Processing For {} in {}'.format(cfg.dataset, timer(start_dataset_process_res, end_dataset_process_res)))


if __name__ == '__main__':
    run()
