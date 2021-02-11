import sys
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print(os.getcwd())

from application.parsers.run_mul import parse_run_multiple, print_args
from application.parsers.config_update import config_update
from application.utils.random_sampler import move_samples
from application.utils.initial_predictor import initial_predictor
from application.utils.profile_generator import generate_shilling_profiles
from application.utils.post_attack_predictor import generate_post_prediction
from application.utils.similarities_evaluator import evaluate_similarities
from application.utils.timer import timer
from application.utils.sendmail import sendmail
import config as cfg

date = time.asctime(time.localtime(time.time())).split(' ')[1:3]


def format_error(error):
    ret = ""
    for line in error:
        ret += line
    return ret


def run():
    args = parse_run_multiple()
    config_update(cfg, args)
    print_args(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    try:
        start_experiment = time.time()

        for dataset in cfg.datasets:
            print("******* Dataset: {0} *******".format(dataset))

            cfg.dataset = dataset
            cfg.r_max = max(cfg.rating_scale[cfg.dataset])
            cfg.r_min = min(cfg.rating_scale[cfg.dataset])

            if 'hop' in dataset.lower():
                print("\t\t\t I do not need baseline because it is the same")
                cfg.semantic_attack_types = [cfg.attack_target_similar]

            start_experiment_dataset = time.time()
            sendmail('{0} - Experiment Journal - {1} {2}'.format(cfg.station, date[1], date[0]),
                     'START {0}'.format(cfg.dataset))

            #  STEP 1.1: Random Sampling
            if cfg.random_sampling:
                print("Generate Random Samples (MOVE THE ENTIRE DATASET INTO DATASAMPLE)")
                move_samples()  # It is Necessary
                sendmail('SERVER - Experiment Journal - {0} {1}'.format(date[1], date[0]),
                         'END Data Move to Data_Samples in {0}'.format(timer(start_experiment, time.time())))
                print("END\t-\tGenerate Random Samples")

            # We don't have Generation of Random Samples but We should move to data_samples the original data
            #  STEP 2.2: Evaluate Similarities
            if cfg.evaluate_similarities:
                print("Evaluate Cosine and [Asymmetric] Cosine Similarities")
                evaluate_similarities()  # It is Necessary
                sendmail('{0} - Experiment Journal - {1} {2}'.format(cfg.station, date[1], date[0]),
                         'END Evaluate Similarities {0}'.format(timer(start_experiment, time.time())))
                print("END\t-\tEvaluate Cosine and [Asymmetric] Cosine Similarities")

            #   STEP 2: Generate Initial Predictions For Each Model and Metrics
            for model in cfg.models:
                start = time.time()
                sendmail('{0} - Experiment Journal - {1} {2}'.format(cfg.station, date[1], date[0]),
                         'START MODEL for {0} - {1} at {2}'.format(cfg.dataset, model,
                                                                   time.strftime("%m/%d/%Y, %H:%M:%S",
                                                                                 time.localtime())))
                cfg.model = model

                print("**************** START MODEL {0} ****************".format(model))

                if cfg.initial_predictions:
                    print("  Generate Initial Predictions")
                    initial_predictor()
                    print("  END\t-\tGenerate Initial Predictions")

                for attack_target in [1]:  # 1 PUSH, 0 NUKE
                    cfg.push = attack_target

                    for attack in cfg.attacks:  # Attack Category bandwagon, average, random, love_hate

                        cfg.attack_type = attack

                        for semantic_attack in cfg.semantic_attack_types:

                            if semantic_attack == cfg.attack_baseline:
                                cfg.semantic = 0

                                print("\t**************** START BASELINE ATTACK {0} {1} ****************".format(attack,
                                                                                                                 'push' if cfg.push else 'nuke'))
                                #   STEP 3: Generate Shilling Profiles
                                if cfg.generate_profiles:
                                    print("\t\tGenerate Shilling Profile")
                                    generate_shilling_profiles(cfg.r_max, cfg.r_min)
                                    print("\t\tEND\t-\tGenerate Shilling Profile")

                                #   STEP 4: Generate Predictions Adding Shilling Profiles
                                if cfg.post_predictions:
                                    for attack_size in cfg.size_of_attacks:
                                        cfg.attackSizePercentage = attack_size
                                        print("\t\tPost Attack Predictions on Percentage of Attack: {0}%".format(
                                            attack_size * 100))
                                        generate_post_prediction()
                                        print(
                                            "\t\tEND\t-\tPost Attack Predictions on Percentage of Attack: {0}%".format(
                                                attack_size * 100))
                                print("\t**************** END BASELINE ATTACK {0}{1} ****************".format(attack,
                                                                                                              'push' if cfg.push else 'nuke'))
                            else:
                                cfg.semantic = 1
                                cfg.semantic_attack_type = semantic_attack

                                for similarity_type in cfg.similarity_types:
                                    cfg.similarity_type = similarity_type

                                    for selection_type in cfg.selection_types:
                                        cfg.selection_type = selection_type
                                        print(
                                            "\t**************** START SEMANTIC ATTACK {0} {1} {2} {3} {4}****************".format(
                                                attack,
                                                'push' if cfg.push else 'nuke', cfg.semantic_attack_type,
                                                cfg.selection_type, cfg.similarity_type))
                                        #   STEP 3: Generate Shilling Profiles
                                        if cfg.generate_profiles:
                                            print("\t\tGenerate Shilling Profile")
                                            generate_shilling_profiles(cfg.r_max, cfg.r_min)
                                            print("\t\tEND\t-\tGenerate Shilling Profile")

                                        #   STEP 4: Generate Predictions Adding Shilling Profiles
                                        if cfg.post_predictions:
                                            for attack_size in cfg.size_of_attacks:
                                                cfg.attackSizePercentage = attack_size
                                                print(
                                                    "\t\tPost Attack Predictions on Percentage of Attack: {0} {1} {2}%".format(
                                                        attack_size * 100, cfg.semantic_attack_type,
                                                        cfg.selection_type))
                                                generate_post_prediction()
                                                print(
                                                    "\t\tEND\t-\tPost Attack Predictions on Percentage of Attack: {0} {1} {2}%".format(
                                                        attack_size * 100, cfg.semantic_attack_type,
                                                        cfg.selection_type))
                                        print(
                                            "\t**************** END SEMANTIC ATTACK {0}{1} {2} {3} ****************".format(
                                                attack,
                                                'push' if cfg.push else 'nuke', cfg.semantic_attack_type,
                                                cfg.selection_type))

                print("**************** END MODEL {0} ****************\n"
                      "**************** {1} ****************".format(model, timer(start, time.time())))
                # Send Email At The End of The MODEL
                sendmail('{0} - Experiment Journal - {1} {2}'.format(cfg.station, date[1], date[0]),
                         'END MODEL for {0} - {1} in {2}'.format(cfg.dataset, model, timer(start, time.time())))

            sendmail('{0} - Experiment Journal - {1} {2}'.format(cfg.station, date[1], date[0]),
                     'END EXPERIMENT on {0} in {1}'.format(cfg.dataset, timer(start_experiment_dataset, time.time())))

        sendmail('{0} - Experiment Journal - {1} {2}'.format(cfg.station, date[1], date[0]),
                 'END ALL EXPERIMENT in {0}'.format(timer(start_experiment, time.time())))

    except Exception as e:
        import traceback
        import sys

        sendmail('{0} - Experiment Journal - {1} {2}'.format(cfg.station, date[1], date[0]),
                 "{0} - {1} - {2}".format(cfg.dataset, e, format_error(traceback.format_exception(*sys.exc_info()))))
        raise  # re-raises the exception


if __name__ == '__main__':
    run()
