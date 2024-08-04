"""
Given a problem input file, this script traverses 
the tagging decision tree based on models' outputs at 
each layer of the tree. 
"""
import argparse
import os
import pathlib
from mathfish.datasets import TreeTaggerDataset
from mathfish.modelWrappers import TaggingWrapper
from mathfish.evaluators import TaggingEvaluator
from tqdm import tqdm
import json
import time
import datetime
from collections import defaultdict

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        help="path to data to convert into dataset. Should be a jsonl file.")

    parser.add_argument(
        "--output_path",
        type=str,
        help="path to store datasets. Dataset will be in jsonl format"
    )

    parser.add_argument(
        "--standards_path",
        type=str,
        help="path to standards.jsonl"
    )

    parser.add_argument(
        "--domain_groups_path",
        type=str,
        help="path to domain_groups.json"
    )

    parser.add_argument(
        "--response_output_path",
        type=str,
        help="path to store model responses, in jsonl format"
    )

    parser.add_argument(
        "--prompt_name",
        type=str,
        help="Name of prompt to use, e.g. \"Promptv1\"."
    )

    parser.add_argument(
        "--shuffle",
        type=bool,
        default=True,
        help="Whether to shuffle the options shown."
    )

    parser.add_argument(
        "--no_positives",
        action='store_true',
        help="Whether the input file has positive labels to consider."
    )

    parser.add_argument(
        "--table_style",
        type=str,
        choices=['special_token', 'html', 'json', 'rst', 'markdown'], 
        default='special_token',
        help='Table formatting.'
    )

    parser.add_argument(
        "--n_shots",
        type=int, 
        choices=[0, 1, 3],
        default=0,
        help='Number of few shot examples to show.'
    )

    parser.add_argument(
        "--few_shot_file",
        type=str,
        default=None,
        help='Path to file containing few shot exemplars.'
    )

    args = parser.parse_args()

    return args

def run_datafile(data_file, level_examples, args): 
    model_name = 'gpt-4-turbo'
    retries = 3
    
    output_file_name = "{data_file}_model-{model_name}_truncate-True_samples-all_retries_{retries}_temp-default_response.jsonl".format(model_name=model_name, 
                                                                                                                        data_file=data_file,
                                                                                                                        retries=retries)
    output_file = os.path.join(args.response_output_path, output_file_name)
    print("Responses:", output_file)

    if os.path.exists(output_file):
        with open(output_file, 'r') as infile:
            lines = infile.readlines()
            if len(lines) == len(level_examples):
                return
            
    wrapper = TaggingWrapper(openai_client=True, model_name_or_path=model_name, tokenizer_name_or_path=model_name, truncate=True, max_length=128000, is_multi_turn=False)

    i = 0
    total_calls = 0
    with open(output_file, 'w') as outfile:
        for d in tqdm(level_examples, desc="Processing this level's dataset"): 
            response = wrapper.call_model_api(d, retry=True, num_retries=3, temp=None)
            total_calls += response['num_calls']
            i += 1

            # save the token_lengths and formatted messages
            json.dump(response, outfile)
            outfile.write('\n')
            if i % 20 == 0:
                ts = time.time()
                print(i, total_calls, datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'), response['prompt-length'], response['completion-length']) 

def make_and_run_domain_level(dataset, example_positives, output_path, input_prefix, args): 
    for instance in tqdm(dataset.instances, desc="Making domain dataset"): 
        if instance['id'] not in example_positives: continue
        pos_domain_groups = example_positives[instance['id']]['pos_domain_groups']
        dataset.make_domain_level(instance, pos_domain_groups)

    output_path = dataset.output_dataset(output_path, input_prefix, precursor='domain')
    print("Dataset:", output_path)
    domain_examples = dataset.examples['domain'] 
    data_file = os.path.basename(output_path)
    run_datafile(data_file, domain_examples, args)

def get_previous_layer(dataset, output_path, input_prefix, args, prev_layer): 
    model_name = 'gpt-4-turbo'
    retries = 3 
    data_file = 'data_treetagger-{input_prefix}_precursor-{prev_layer}_prompt-{prompt}_table-{table_style}_turns-single_shots-{n_shots}.jsonl'.format(prompt=args.prompt_name, 
                                                                                                                                        input_prefix=input_prefix,
                                                                                                                                        table_style=args.table_style,
                                                                                                                                        prev_layer=prev_layer,
                                                                                                                                        n_shots=args.n_shots)
    data_file_path = os.path.join(output_path, data_file)
    evaluator = TaggingEvaluator(data_file_path)
    input_file_name = "{data_file}_model-{model_name}_truncate-True_samples-all_retries_{retries}_temp-default_response.jsonl".format(model_name=model_name, 
                                                                                                                        data_file=data_file,
                                                                                                                        retries=retries)
    input_file = os.path.join(args.response_output_path, input_file_name) 
    idx_to_option_list = {} # {instance ID: option list}
    with open(data_file_path, 'r') as infile: 
        for line in infile: 
            d = json.loads(line)
            idx_to_option_list[d['id']] = d['messages'][-2]['options']
    
    problems_to_pred = {}
    with open(input_file, 'r') as infile: 
        for line in infile: 
            d = json.loads(line)
            model_response = evaluator.clean_output(d['model-response'])
            if type(model_response) != list: continue
            response_idx = set([evaluator.get_option_letter_to_idx(letter) for letter in model_response])
            prev_layer_pred = []
            for idx in response_idx: 
                if idx > len(idx_to_option_list[d['id']])-1: 
                    # hallucinated option
                    continue
                branch = idx_to_option_list[d['id']][idx]
                pointer = dataset.retriever.get_pointer_to_next_branch(branch, prev_layer)
                if type(pointer) == str:
                    prev_layer_pred.append(pointer)
                else: 
                    prev_layer_pred.extend(pointer)
            problem_id = evaluator.get_problem_id(d['id'])
            problems_to_pred[problem_id] = prev_layer_pred
    return problems_to_pred

def make_and_run_cluster_level(dataset, example_positives, output_path, input_prefix, args): 
    problems_to_pred = get_previous_layer(dataset, output_path, input_prefix, args, 'domain') 
            
    for instance in tqdm(dataset.instances, desc="Making cluster dataset"): 
        if instance['id'] not in example_positives: continue
        prev_layer_pred = problems_to_pred[instance['id']]
        pos_clusters = example_positives[instance['id']]['pos_clusters']
        dataset.make_cluster_level(instance, prev_layer_pred, pos_clusters) 

    output_path = dataset.output_dataset(output_path, input_prefix, precursor='cluster')
    print("Dataset:", output_path)
    cluster_examples = dataset.examples['cluster'] 
    print(len(cluster_examples))
    data_file = os.path.basename(output_path)
    run_datafile(data_file, cluster_examples, args)

def make_and_run_standard_level(dataset, example_positives, output_path, input_prefix, args):
    problems_to_pred_domain = get_previous_layer(dataset, output_path, input_prefix, args, 'domain')  
    problems_to_pred_cluster = get_previous_layer(dataset, output_path, input_prefix, args, 'cluster') 

    for instance in tqdm(dataset.instances, desc="Making standard dataset"): 
        if instance['id'] not in example_positives: continue
        prev_prev_layer_pred = problems_to_pred_domain[instance['id']]
        if not prev_prev_layer_pred or prev_prev_layer_pred == ['Modeling']: 
            # no cluster layer was created
            continue
        prev_layer_pred = problems_to_pred_cluster[instance['id']]
        pos_standards = example_positives[instance['id']]['pos_standards'] 
        dataset.make_standard_level(instance, prev_prev_layer_pred, prev_layer_pred, pos_standards)

    output_path = dataset.output_dataset(output_path, input_prefix, precursor='standard')
    print("Dataset:", output_path)
    standard_examples = dataset.examples['standard'] 
    print(len(standard_examples))
    data_file = os.path.basename(output_path)
    run_datafile(data_file, standard_examples, args)

if __name__=="__main__": 
    args = parse_args()

    standards_path = args.standards_path
    domain_groups_path = args.domain_groups_path
    prompt_file = pathlib.Path(__file__).parent.parent / "mathfish" / "datasets" / "prompts.json"
    input_prefix = os.path.basename(args.input_file).replace('.jsonl', '')

    if args.n_shots > 0: 
        assert args.few_shot_file
    dataset = TreeTaggerDataset(standards_path, args.prompt_name, prompt_file, domain_groups_path, False, 
                                few_shot_file=args.few_shot_file, 
                                shuffle_options=args.shuffle, table_style=args.table_style, n_shots=args.n_shots)
    dataset.load_instances(args.input_file)

    if args.no_positives:
        example_positives = defaultdict(dict) 
        for instance in dataset.instances: 
            example_positives[instance['id']]['pos_domain_cats'] = set()
            example_positives[instance['id']]['pos_domain_groups'] = set()
            example_positives[instance['id']]['pos_clusters'] = set()
            example_positives[instance['id']]['pos_standards'] = set()
    else:
        example_positives = dataset.get_example_positives()
        
    make_and_run_domain_level(dataset, example_positives, args.output_path, input_prefix, args)

    make_and_run_cluster_level(dataset, example_positives, args.output_path, input_prefix, args)

    make_and_run_standard_level(dataset, example_positives, args.output_path, input_prefix, args)