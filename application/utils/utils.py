import pickle


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def save_results_csv(obj, name):
    with open(name + '.csv', 'w') as f:
        f.write('Attack,SimilarityType,FeatureType,0,1,2.5,5\n')
        for name_of_exp in obj.keys():
            for type_of_sim in obj[name_of_exp].keys():
                if type_of_sim != 'Baseline':
                    for type_of_features in obj[name_of_exp][type_of_sim].keys():
                        val = ''
                        for perc in obj[name_of_exp][type_of_sim][type_of_features]:
                            val += str(perc) + ','
                        f.write('{},{},{},{}\n'.format(
                            name_of_exp,
                            type_of_sim,
                            type_of_features,
                            val[:-1]
                        ))  # -1 remove the last comma
                else:
                    val = ''
                    for perc in obj[name_of_exp][type_of_sim]:
                        val += str(perc) + ','
                    f.write('{},{},{},{}\n'.format(
                        name_of_exp,
                        type_of_sim,
                        'XXX',
                        val[:-1]
                    ))  # -1 remove the last comma


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
