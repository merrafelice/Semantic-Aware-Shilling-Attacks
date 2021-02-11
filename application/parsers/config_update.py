def config_update(cfg, args):
    # Operation
    cfg.random_sampling = args.random_sampling
    cfg.initial_predictions = args.initial_predictions
    cfg.evaluate_similarities = args.evaluate_similarities
    cfg.generate_profiles = args.generate_profiles
    cfg.post_predictions = args.post_predictions

    # Similarity Types
    cfg.similarity_types = args.similarity_types

    # Parameters for Katz and Exclusivity
    cfg.topk = args.topk
    cfg.alpha = args.alpha

    # Models
    cfg.models = args.models

    # Similarity Types
    cfg.selection_types = args.selection_types

    # Semantic Attack Types
    cfg.semantic_attack_types = args.semantic_attack_types

    # Datasets
    cfg.datasets = args.datasets

    # Number of Attacked Items For Each Data Sample
    cfg.item_size = args.item_size

    # Number of Processes
    cfg.number_processes = args.number_processes

    # Attacks
    cfg.attacks = args.attacks

    # Top-k Similar Items used in the Building of Filler Items
    cfg.top_k_similar_items = args.top_k_similar_items

    # 1%, 2.5%, 5% Shilling Attacks
    cfg.size_of_attacks = args.size_of_attacks

    # Station
    cfg.station = args.station


def config_update_run_process(cfg, args, project_dir):
    # Project Dir
    cfg.project_dir = project_dir

    # Datasets
    cfg.datasets = args.datasets

    # Models
    cfg.models = args.models

    # Metric to Evaluate
    cfg.metrics = args.metrics

    # Top-K Metrics
    cfg.top_k_metrics = args.top_k_metrics

    # Semantic Attack Types
    cfg.semantic_attack_types = args.semantic_attack_types

    # Attacks
    cfg.attacks = args.attacks

    # Station
    cfg.station = args.station


def config_update_run_paired_ttest(cfg, args, project_dir):
    # Project Dir
    cfg.project_dir = project_dir

    # Datasets
    cfg.datasets = args.datasets

    # Models
    cfg.models = args.models

    # Metric to Evaluate
    cfg.metrics = args.metrics

    # Top-K Metrics
    cfg.top_k_metrics = args.top_k_metrics
