# Towards Semantic-Aware Shilling Attacks against Collaborative Recommender Systems
This file presents the reproducibility details of the paper. **Towards Semantic-Aware Shilling Attacks against Collaborative Recommender Systems** submitted at Semantic Web Journal 2021

**Table of Contents:**
- [Requirements](#requirements)
- [Datasets](#datasets)
- [Reproducibility Details](#reproducibility-details)
  - [1. Identify the Target Items](#1-identify-the-target-items)
  - [2. Evaluate the Topological Similarities](#2-evaluate-the-topological-similarities)
  - [3. Perform the Shilling Attacks](#3-perform-the-shilling-attacks)
  - [4. Evaluation](#4-evaluation)

## Requirements

To run the experiments, it is necessary to install the following requirements. 

* Python 3.6.9
* CUDA 10.1
* cuDNN 7.6.4

After having clone this repository with 
```
git clone repo-name
```
we suggest creating e virtual environment install the required Python dependencies with the following commands
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
## Datasets
The tested datasets across the paper experiments are reported in the following table.

|       Dataset      |   # Users   | # Items   |  # Ratings   | Sparsity | # Features 1HOP | # Features 1HOP |
| ------------------ |:-----------:| ------------| ------------- | --------| --------------------- | --------------------- | 
|     Library Thing     |    4,816        | 2,256        |  76,421       | 99.30% | 56,019                | 4,259,728                |
|     Yahoo! Movies     |    4,000        | 2,526        |  64,079       | 99.37% | 105,733                | 6,697,986                |

The training dataset is available at ```./data/<dataset_name>/ratings.csv``` with the format
```
user-Id,tem-Id,rating
```

The file ```./data/<dataset_name>/df_map.csv``` contains the connection between items and features in the format
```
feature,item,item_index,value
```
while, the file ```./data/<dataset_name>/features.tsv``` contains the predicate and object URIs related to each indexed feature, e.g., ```0[TAB]<http://dbpedia.org/ontology/publisher><http://dbpedia.org/resource/Pocket_Books>```,

Additionally, the file ```./data/<dataset_name>/selected_features.csv``` contains the features filtered with the process define din the article. It is possible to put here the list of features saved following the next template:
```
features,type
"[1619, 2133, 5092, 10048, 39949, 3235, 94, 33182]",categorical
"[1, 5, 98, 7, 465, 1025]",factual
"[0, 8999, 4672]",ontological
```

The datasets used in the experiments ara available in the current repo by downloading and extracting this [archive](https://drive.google.com/file/d/1iKxaYhd_33yH0LtcZuO7Nf0yFcHFQXmI/view?usp=sharing).

## Reproducibility Details
To execute the following script it is necessary to execute the following command in the shell
```
cd application
```

### 1. Identify the Target Items
The first step is to generate the target items executing the following command
```
python run_generate_target_items.py 
  --datasets <data-name-1> <data-name-2> 
  --num_target_items <num_target_items> 
```
The script will generate the file ```./data/<dataset_name>/target_items.csv```.
We have performed our experiments setting ```num_target_items = 50```.

### 2. Evaluate the Topological Similarities
To evaluate the relatedness measures to build the semantic-aware profiles it is necessary to run
```
python run_generate_target_items.py 
	 --datasets	<data-name-1> <data-name-2> 
	 --topk	10 
	 --topn	100
	 --top_k_similar_items	0.25
	 --alpha	0.25
	 --is_exclusivity	1
	 --is_katz	1
```
The results will be stored in the directory ```./data/<dataset_name>/similarities/```.
The cosine similarity evaluation is performed within the following command.

### 3. Perform the Shilling Attacks
After having executed the previous commands we can start the flow of attacking the recommender models by running the following command.
```
python run_multiple_execution_server.py 
         --gpu  -1
         --random_sampling      1
         --initial_predictions  0
         --evaluate_similarities        0
         --generate_profiles    0
         --post_predictions     0
         --similarity_types     ['katz', 'exclusivity', 'cosine']
         --semantic_attack_types        ['target_similar']
         --topk 10
         --alpha        0.25
         --models       ['NCF', 'SVD', 'ItemkNN', 'UserkNN']
         --selection_types      ['categorical', 'ontological', 'factual']
         --datasets     ['SmallLibraryThing']
         --item_size    0.05
         --number_processes     1
         --attacks      ['Random', 'Average']
         --num_target_items     10
         --top_k_similar_items  0.25
         --size_of_attacks      [0.01, 0.025, 0.05]
         --station      <server-name>

```
Note that we have provided an example of execution. It is possible to change the command parameters to reproduce/execute any type of attack.
At the end of the execution all the crafted shilling profiles will be store in ```./<model-name>/shilling_profiles/<dataset_name>/``` while the positions and scores of the Target Items will be saved in
```./<model-name>/results/<dataset_name>/```.

### 4. Evaluation
To measure all the results that we report in tha paper, we can execute the following command
```
python run_process_attack_result.py 
         --datasets     ['SmallLibraryThing']
         --models       ['NCF']
         --metrics      ['HR', 'PS']
         --top_k_metrics        10
         --semantic_attack_types        ['target_similar', 'baseline']
         --attacks      ['Random', 'Average']
         --station      not-specified

```
At the end of this command, the results files will be saved in ```./<model-name>/HR/<dataset_name>/``` and ```./<model-name>/PS/<dataset_name>/``` directories. Note that, the results are evaluated within a significance statistical test by adding asterisks at the end of each PS/HR values.