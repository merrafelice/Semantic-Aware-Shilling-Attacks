import pandas as pd
from sendmail import sendmail
import numpy as np
import sys
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from application.parsers.run_mul import parse_run_generate, print_args

np.random.seed(1234)

target_quartile_sensitive = 0

# Quartile Base TArgeting
item_size = 0
num_quartiles = 0
num_quartiles_target_items = 0


def run():
    args = parse_run_generate()
    print_args(args)

    num_target_items = args.num_target_items

    for dataset in args.datasets:
        print('Generate for {0}'.format(dataset))

        df = pd.read_csv('../data/{0}/ratings.csv'.format(dataset))

        df = df.iloc[:, :3]

        df.columns = ['userId', 'itemId', 'rating']

        df_target_items = pd.DataFrame()
        if target_quartile_sensitive:
            # Identification of Items Under Attack from Quartiles
            num = int(df.itemId.nunique() * item_size)
            if num < 5:
                num = 4
            counts = df.groupby(['itemId']).size().reset_index(name='counts')
            # quartiles = counts.counts.quantile([0.25, 0.5, 0.75])

            items = {}
            quantiles = pd.qcut(counts['counts'], num_quartiles, duplicates='drop')
            quantiles.name = 'quantiles'
            counts = pd.DataFrame(counts).join(quantiles)

            for quartile_number, quantile in enumerate(counts.quantiles.unique()):
                quartile = quartile_number
                for number_q, eleId in enumerate(
                        list(
                            counts[counts.quantiles == quantile].sample(num_quartiles_target_items,
                                                                        replace=False).itemId)):
                    if items.get(eleId) is None:
                        items[eleId] = quartile + 1
                        df_target_items = df_target_items.append({
                            'itemId': eleId,
                            'quartile': quartile + 1,
                            'sample': 0
                        }, ignore_index=True)

            if len(items) == 0:
                print("************************* ALERT num: {0}**********************".format(num))
            else:
                print("Store Target Items . LENGTH: {0}".format(df_target_items['itemId'].nunique()))

        else:
            items_id = df['itemId'].unique()
            targets = np.random.choice(items_id, num_target_items, replace=False)
            for eleId in targets:
                df_target_items = df_target_items.append({
                    'itemId': eleId,
                    'quartile': 0,
                    'sample': 0
                }, ignore_index=True)
            print("\n******* Store Target Items . LENGTH: {0} *******".format(df_target_items['itemId'].nunique()))

        filename = '../data/{0}/target_items.csv'.format(dataset)
        df_target_items = df_target_items.astype({'itemId': 'int32'})
        df_target_items.to_csv(filename, index=False)


if __name__ == '__main__':
    run()
