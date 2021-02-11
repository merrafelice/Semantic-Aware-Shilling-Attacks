import argparse


def print_args(args):
    print('**********\nPARAMETERS:')
    for arg in vars(args):
        print("\t --{0}\t{1}".format(arg, getattr(args, arg)))
    print('**********\n')


def parse_run_topological():

    # python3 example.py --list a b c
    parser = argparse.ArgumentParser(description="Generate Target Items.")

    parser.add_argument("--datasets", nargs="+", default=['SmallLibraryThing'], help="You can choose between: "
                                                                                     "'SmallLibraryThing', 'SmallLibraryThing2Hops',"
                                                                                     " 'yahoo_movies', 'yahoo_movies_2_hops' ")

    parser.add_argument('--topk', type=int, default=10, help='The threshold on which we build the relatedness measures')
    parser.add_argument('--topn', type=int, default=100, help='The number of extracted top shortest paths (Not useful in Katz)')
    parser.add_argument('--top_k_similar_items', type=float, default=0.25, help='Top similar items')
    parser.add_argument('--alpha', type=float, default=0.25, help='Alpha parameters')

    parser.add_argument('--is_exclusivity', type=int, default=1, help='Execute Exclusivity')
    parser.add_argument('--is_katz', type=int, default=1, help='Execute Katz')

    return parser.parse_args()


def parse_run_generate():
    # python3 example.py --list a b c
    parser = argparse.ArgumentParser(description="Generate Target Items.")

    # Datasets
    parser.add_argument("--datasets", nargs="+", default=['SmallLibraryThing'], help="You can choose between: "
                                                                                     "'SmallLibraryThing', 'SmallLibraryThing2Hops',"
                                                                                     " 'yahoo_movies', 'yahoo_movies_2_hops' ")
    # num_target_items
    parser.add_argument('--num_target_items', type=int, default=50, help='Number of Target Items')
    return parser.parse_args()


def parse_run_multiple():
    # python3 example.py --list a b c
    parser = argparse.ArgumentParser(description="Run Multiple Execution Server.")

    # GPU
    parser.add_argument('--gpu', type=int, default=-1, help='GPU Number (-1 --> CPU)')

    # Command To Execute
    parser.add_argument('--random_sampling', type=int, default=1, help='Random Sampler')
    parser.add_argument('--initial_predictions', type=int, default=0, help='Initial Predictor')
    parser.add_argument('--evaluate_similarities', type=int, default=0, help='Evaluate Similarities')
    parser.add_argument('--generate_profiles', type=int, default=0, help='Generate Profiles')
    parser.add_argument('--post_predictions', type=int, default=0, help='Evaluate Post Predictions')

    # Similarity Types
    parser.add_argument("--similarity_types", nargs="+", default=["katz", "exclusivity", "cosine"])

    # Semantic Attack Types
    parser.add_argument("--semantic_attack_types", nargs="+", default=['target_similar'],
                        help="The type of attacks. Possible values: 'target_similar', 'baseline'")

    # Parameters for Katz and Exclusivity
    parser.add_argument('--topk', type=int, default=10, help='Top-k Shortest Path used to evaluate the Relatedness')
    parser.add_argument('--alpha', type=float, default=0.25, help='Alpha used in Relatedness-measures')

    # Models
    parser.add_argument("--models", nargs="+", default=["NCF", "SVD", "ItemkNN", "UserkNN"])

    # Similarity Types
    parser.add_argument("--selection_types", nargs="+", default=['categorical', 'ontological', 'factual'])

    # Datasets
    parser.add_argument("--datasets", nargs="+", default=['SmallLibraryThing'], help="You can choose between: "
                                                                                     "'SmallLibraryThing', 'SmallLibraryThing2Hops',"
                                                                                     " 'yahoo_movies', 'yahoo_movies_2_hops' ")

    # Number of Attacked Items For Each Data Sample
    parser.add_argument('--item_size', type=float, default=0.05,
                        help='Equal to the fraction (percentage 5%) of Attacked Items')

    # Number of Processes
    parser.add_argument('--number_processes', type=int, default=1, help='Number of Process for the Multi-thread')

    # Attacks
    parser.add_argument("--attacks", nargs="+", default=["Random", 'Average'], help="Possible Attacks:"
                                                                                    "'BandWagon', 'PerfectKnowledge',"
                                                                                    "'Random', 'Average',"
                                                                                    "'LoveHate', 'Popular' ")

    # Number of Target Items
    parser.add_argument('--num_target_items', type=int, default=10, help='Number of Process for the Multi-thread')

    # Top-k Similar Items used in the Building of Filler Items
    parser.add_argument('--top_k_similar_items', type=float, default=0.25,
                        help='Top-k Similar Items used in the Building of Filler Items')

    # 1%, 2.5%, 5% Shilling Attacks
    parser.add_argument("--size_of_attacks", nargs="+", default=[0.01, 0.025, 0.05])

    # Station
    parser.add_argument("--station", type=str, default="not-specified", help='Type the name of the server')

    return parser.parse_args()


def parse_run_process():
    parser = argparse.ArgumentParser(description="Run Multiple Execution Server.")

    # Datasets
    parser.add_argument("--datasets", nargs="+", default=['SmallLibraryThing'], help="You can choose between: "
                                                                                     "'SmallLibraryThing', 'SmallLibraryThing2Hops',"
                                                                                     " 'yahoo_movies', 'yahoo_movies_2_hops' ")
    # Models
    parser.add_argument("--models", nargs="+", default=["NCF"], help="NCF, SVD, ItemkNN, UserkNN")

    # Metric to Evaluate
    parser.add_argument("--metrics", nargs="+", default=["HR", "PS"],
                        help="The list of metrics used to study the attack efficacy")

    # Top-K Metrics
    parser.add_argument('--top_k_metrics', type=int, default=10, help='Top-k Of Metrics')

    # Semantic Attack Types
    parser.add_argument("--semantic_attack_types", nargs="+", default=['target_similar', 'baseline'],
                        help="The type of attacks. Possible values: 'target_similar', 'baseline'")

    # Attacks
    parser.add_argument("--attacks", nargs="+", default=["Random", 'Average'], help="Possible Attacks:"
                                                                                    "'BandWagon', 'PerfectKnowledge',"
                                                                                    "'Random', 'Average',"
                                                                                    "'LoveHate', 'Popular' ")

    # Station
    parser.add_argument("--station", type=str, default="not-specified", help='Type the name of the server')

    return parser.parse_args()


def parse_run_paired_ttest():
    parser = argparse.ArgumentParser(description="Run Paired T-Test.")

    # Datasets
    parser.add_argument("--datasets", nargs="+", default=['SmallLibraryThing'], help="You can choose between: "
                                                                                     "'SmallLibraryThing', 'SmallLibraryThing2Hops',"
                                                                                     " 'yahoo_movies', 'yahoo_movies_2_hops' ")
    # Models
    parser.add_argument("--models", nargs="+", default=["NCF"], help="NCF, SVD, ItemkNN, UserkNN")

    # Metric to Evaluate
    parser.add_argument("--metrics", nargs="+", default=["HR", "PS"],
                        help="The list of metrics used to study the attack efficacy")

    # Top-K Metrics
    parser.add_argument('--top_k_metrics', type=int, default=10, help='Top-k Of Metrics')

    # Semantic Attack Types
    parser.add_argument("--semantic_attack_types", nargs="+", default=['target_similar', 'baseline'],
                        help="The type of attacks. Possible values: 'target_similar', 'baseline'")

    # Attacks
    parser.add_argument("--attacks", nargs="+", default=["Random", 'Average'], help="Possible Attacks:"
                                                                                    "'BandWagon', 'PerfectKnowledge',"
                                                                                    "'Random', 'Average',"
                                                                                    "'LoveHate', 'Popular' ")

    # Station
    parser.add_argument("--station", type=str, default="not-specified", help='Type the name of the server')

    return parser.parse_args()
