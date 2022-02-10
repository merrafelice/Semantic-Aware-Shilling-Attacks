# Configuration File

# Chose Steps
random_sampling = 0
initial_predictions = 1
evaluate_similarities = 0
generate_profiles = 0
post_predictions = 0

# Similarity Types
cosine = 'cosine'
asymmetric_cosine = 'asymmetric_cosine'
katz = 'katz'
exclusivity = 'exclusivity'
similarity_type = None
similarity_types = [katz]

# Parameters for Katz and Exclusivity
alpha = 0.25
topk = 10

# Semantic
semantic = 0  # Set During Execution

# Computer name
station = 'Server'

# Directories
data = 'data'
data_samples = 'data_samples'
results = 'results'
shilling_profiles = 'shilling_profiles'
similarities = 'similarities'

# Models and Directory
userknn = 'UserkNN'
itemknn = 'ItemkNN'
svd = 'SVD'
ncf = 'NCF'
models = [ncf]  # , svd, userknn, itemknn]

# Similarities Types
selection_types = ['categorical', 'ontological', 'factual']
selection_type = None

# Datasets
facebook_book = 'facebook_book'
facebook_movies = 'facebook_movies'
facebook_music = 'facebook_music'
last_fm = 'last_fm'
library_thing = 'LibraryThing'
amazon_book = 'amazon_book'
yahoo_movies = 'yahoo_movies'
yahoo_movies_2_hops = 'yahoo_movies_2_hops'
small_library_thing = 'SmallLibraryThing'
small_library_thing_2_hops = 'SmallLibraryThing2Hops'

# datasets = [yahoo_movies, yahoo_movies_2_hops]
datasets = [small_library_thing]

rating_scale = {
    library_thing: (1.0, 10.0),
    small_library_thing: (1.0, 10.0),
    small_library_thing_2_hops: (1.0, 10.0),
    yahoo_movies: (1.0, 5.0),
    yahoo_movies_2_hops: (1.0, 5.0),
    facebook_book: (1.0, 1.0)
}

r_min = 1
r_max = 5

# Files
training_file = 'ratings.csv'
data_samples_metrics = 'data_samples_metrics.csv'
user_data_samples_metrics = 'user_data_samples_metrics.csv'
item_data_samples_metrics = 'item_data_samples_metrics.csv'
final_results = 'final_results.csv'
final_results_user_item = 'final_results_user_item.csv'
map_file = 'map.tsv'
similarities_file = '{0}_{1}_{2}_similarities.txt'
target_items = 'target_items.csv'
selected_features = 'selected_features.csv'

initial_prediction = 'initial_prediction'
post_prediction = 'post_prediction'

results_name = 'results_mean_at_{}'
results_std_name = 'results_stddev_at_{}'

# Old Metrics
overall_prediction_shift = 'overall_prediction_shift'
overall_hit_ration_at_k = 'overall_hit_ration_at_{0}'

# Metrics
metrics = ['HR', 'PS']
metric = None

# Number of Attacked Items For Each Data Sample
item_size = 0.05  # Equal to 5%

# MultiThread
# number_processes = mp.cpu_count() - 1  # -1 Useful To Avoid Lock
number_processes = 1

# Attacks
bandwagon = 'BandWagon'  # Implemented
perfect_knowledge = 'PerfectKnowledge'
random = 'Random'  # Implemented
average = 'Average'  # Implemented
love_hate = 'LoveHate'
popular = 'Popular'

attacks = [bandwagon, random, average]

# Semantic Attacks
attack_target_similar = 'target_similar'
attack_popular_similar = 'popular_similar'
attack_baseline = 'baseline'
semantic_attack_type = None
semantic_attack_types = [attack_target_similar]
# semantic_attack_types = [attack_baseline]

# Pre Conf
bandwagonAttackType = 1  # 1 = Average, 0 = Random

#################################################################
#########            SERVER CONFIGURATION               #########
#################################################################
# Core RS Model
model = svd

# Num of Data Samples for Each Dataset
num_data_samples = 1

top_k_similar_items = 0.25  # 25%

# This configurations is for Quartile-based analysis
target_quartile_sensitive = 0
num_quartiles = 2
num_quartiles_target_items = 10

# num_target_items = 50   # Set to division 4 because it is used also for quartile
num_target_items = 1  # Set to division 4 because it is used also for quartile

# Attack Configuration
attack_type = bandwagon  # SELECT ATTACK TYPE
push = 1  # Push=1, Nuke=0
size_of_attacks = [0.01, 0.025, 0.05]  # 1%, 2.5%, 5%
attackSizePercentage = max(size_of_attacks)  # MAX Percentage of Users
targetRating = r_max if push else r_min
fillerRating = int(r_max - r_min)
fillerSize = 0
selectedSize = 0

# SAVE FULL REC
save_full_rec_list = 0

# PRINT INFORMATION
print("Configuration\n")
print("******* N Processes: {0} *******".format(number_processes))
print("******* YOU HAVE TO EVALUATE TOP {0} BASED ON THE DATASET *******".format(top_k_similar_items))
print("******* List of Attacks: {0} *******".format(str(attacks)))
print("******* Target QUARTILE SENSITIVE: {0} *******\n".format('YES' if target_quartile_sensitive else 'NO'))
if target_quartile_sensitive:
    print("******* N Attacked Items for Quartile: {0} *******".format(num_quartiles_target_items))
else:
    print("******* N Attacked Items: {0} *******".format(num_target_items))
print("******* SAVE Full Rec Lists: {0} *******\n".format('YES' if save_full_rec_list else 'NO'))
