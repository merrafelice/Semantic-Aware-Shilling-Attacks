import os
import config as cfg


def check_dir_data_samples(root):
    """
    Create The Dataset Directory/ Or Remove and Restore for the dataset in  a specific space
    i.e., we need to store results
    This method remove the samples for the directory
    :param root:
    :return:
    """
    samples_dir = os.path.join(root, cfg.dataset)
    if os.path.exists(samples_dir):
        import shutil
        shutil.rmtree(samples_dir)
    os.makedirs(samples_dir)


def check_dir_results(root):
    """
    Create The Dataset Directory/ Or Remove and Restore for the dataset in  a specific space
    i.e., we need to store results
    This method remove the old RESULTS PREDICTIONS for the directory
    :param root:
    :return:
    """
    samples_dir = os.path.join(root, cfg.dataset)
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)


def check_dir_attack(root):
    """
    Create The Dataset Directory/ Or Remove and Restore for the dataset in  a specific space
    i.e., we need to store results
    This method remove the old reuslts for the directory
    :param root:
    :return:
    """
    # Create Dataset Directory if it does not exist
    samples_dir = os.path.join(root, cfg.dataset)
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)

    # Create/ReWrite Attack Directory
    samples_dir = os.path.join(root, cfg.dataset, "{0}_{1}".format(cfg.attack_type, 'push' if cfg.push else 'nuke'))
    if os.path.exists(samples_dir):
        import shutil
        shutil.rmtree(samples_dir)
    os.makedirs(samples_dir)


def check_dir_semantic_attack(root):
    """
    Create The Dataset Directory/ Or Remove and Restore for the dataset in  a specific space
    i.e., we need to store results
    This method remove the old reuslts for the directory
    :param root:
    :return:
    """
    # Create Dataset Directory if it does not exist
    samples_dir = os.path.join(root, cfg.dataset)
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)

    sim_file_name = ""
    if cfg.semantic:
        if cfg.similarity_type == cfg.cosine:
            sim_file_name = 'cosine'
        elif cfg.similarity_type == cfg.katz:
            sim_file_name = 'katz-a{0}-top{1}'.format(cfg.alpha, cfg.topk)
        elif cfg.similarity_type == cfg.exclusivity:
            sim_file_name = 'exclusivity-a{0}-top{1}'.format(cfg.alpha, cfg.topk)

    # Create/ReWrite Attack Directory
    samples_dir = os.path.join(root, cfg.dataset, "{0}_{1}_{2}_{3}_{4}".format(cfg.attack_type,
                                                                               'push' if cfg.push else 'nuke',
                                                                               sim_file_name,
                                                                               cfg.semantic_attack_type,
                                                                               cfg.selection_type)
                               )
    if os.path.exists(samples_dir):
        import shutil
        shutil.rmtree(samples_dir)
    os.makedirs(samples_dir)


def check_dir_similarities(root):
    """
    Create The Dataset Directory/ Or Remove and Restore for the dataset in  a specific space
    i.e., we need to store results
    This method remove the old RESULTS PREDICTIONS for the directory
    :param root:
    :return:
    """
    samples_dir = os.path.join(root, cfg.dataset)
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)


def check_dir_metrics_and_plots(root):
    """
    Create The Dataset Directory to store HR and PS
    :param root:
    :return:
    """
    # Create Dataset Directory if it does not exist
    samples_dir = os.path.join(root, cfg.dataset)

    # If we want only check the existence
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)