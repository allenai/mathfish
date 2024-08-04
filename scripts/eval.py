"""
Eval script for predictions 

Static just-calculate-some-performance-metrics mode: 
    python scripts/eval.py \
    --predictions_file data_verification-dev_nsample-5_grade-different_domain-different_prompt-Promptv1_model-Mixtral-8x7B-Instruct-v0.1_truncate-False_output_model_response.jsonl \
    --standards_file achieve-the-core/standards.jsonl \
    --problems_file dev.jsonl \
    --data_file data_verification-dev_nsample-5_grade-different_domain-different_prompt-Promptv1.jsonl \
    --output_path results/eval/ \
    --task_format verification

Interactive "show-me-examples-of-mistakes" mode, without metrics (u can do metrics too by removing the --no_compute_metrics flag): 
    python scripts/eval.py \
    --predictions_file /net/nfs.cirrascale/s2-research/tala/edreports/output/data_files/model_responses/lucy_outputs/few_shot_verification/data_verification-dev_nsample-1_neg-strat-all-negative-types_prompt-Promptv7_table-markdown_shots-3_model-gpt-4-turbo_truncate-True_samples-all_retries_5_temp-default_response.jsonl \
    --data_file /net/nfs.cirrascale/s2-research/lucyl/edreports/dataset_files/data_verification-dev_nsample-1_neg-strat-all-negative-types_prompt-Promptv7_table-markdown_shots-3.jsonl \
    --interactive --no_compute_metrics \
    --task_format verification 

Interactive works only for the * verification * prompting format. 

Author: Tal August (@tala), Lucy Li
Email:  tala@allenai.org, lucyl@allenai.org
"""

import numpy as np 
import json
import random
from tqdm import tqdm
import re
import pprint
from sklearn.metrics import f1_score, accuracy_score
import argparse
from collections import defaultdict
import os
from mathfish.utils import *
from mathfish.evaluators import VerificationEvaluator, TaggingEvaluator

def parse_args():
    '''
    predictions_file: jsonl includes the following keys 
    - dataset (string)
    - id (string), e.g. im_center_000001_neg_2
    - messages (list) containing jsons representing roles (e.g. user or assistant) and content
    - output (string), model response

    problems_file: jsonl includes keys specified in data/README.md or tests/test_json.py
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions_file",
        type=str,
        help="file for predictions",
        required=True
        )
    
    parser.add_argument(
        "--problems_file", 
        type=str,
        help="original file containing problems and metadata, which includes standardized standards",
    )

    parser.add_argument(
        "--standards_file", 
        type=str,
        help="file containing one standard per line",
    )

    parser.add_argument(
        "--data_file", 
        type=str,
        required=True,
        help="file containing input dataset",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        help="file for labels"
    )

    parser.add_argument(
        "--task_format", 
        type=str,
        required=True,
        choices=['verification', 'tagging'],
        help="task format",
    )
    
    parser.add_argument(
        "--interactive", 
        default=False,
        action='store_true',
        help="whether to show examples of different types of mistakes"
    )

    parser.add_argument(
        "--num_examples",
        type=int,
        default=10,
        help="used only in interactive mode, the number of examples to show"
    )

    parser.add_argument(
        "--no_compute_metrics",
        default=False,
        action='store_true',
        help="Whether to calculate eval metrics"
    )

    args = parser.parse_args()

    if not args.no_compute_metrics:
        if not args.problems_file or not args.output_path or not args.standards_file:
            parser.error('The default is computing metrics, which requires --problems_file, --standards_file, and --output_path arguments')

    return args

def load_predictions_and_labels(predictions_file, evaluator):
    results = defaultdict(dict)
    with open(predictions_file, 'r') as infile: 
        for line in infile: 
            d = json.loads(line)
            example_id = d['id']
            output = d['model-response']
            label = evaluator.get_true_label(example_id) 
            assert example_id not in results
            results[example_id]['output'] = output
            results[example_id]['label'] = label
            results[example_id]['prediction'] = evaluator.clean_output(output)
            results[example_id]['dataset'] = d['dataset']
            results[example_id]['messages'] = d['message_content']
    return results 

def get_problem_metadata(problems_file): 
    problem_meta = defaultdict(dict)
    with open(problems_file, 'r') as infile: 
        for line in infile: 
            d = json.loads(line)
            problem_id = d['id']
            problem_meta[problem_id]['problem_activity_type'] = d['metadata']['problem_activity_type']
            problem_meta[problem_id]['standards'] = d['standards'] 
            problem_meta[problem_id]['elements'] = list(d['elements'].keys())
            problem_meta[problem_id]['source'] = d['source']
    return problem_meta

def get_standard_desc_dict(standards_file): 
    '''
    Get a mapping from descriptions to standards
    '''
    standards_dict = {}
    with open(standards_file, 'r') as infile: 
        for line in infile: 
            d = json.loads(line)
            if d['level'] != 'Standard': continue
            standard_id = d['id']
            standard_desc = d['description']
            assert standard_desc not in standards_dict
            standards_dict[standard_desc] = standard_id
    return standards_dict

if __name__ == "__main__":
    args = parse_args()
    pred_filename = os.path.basename(args.predictions_file)
    data_filename = os.path.basename(args.data_file).replace('.jsonl', '') 
    assert pred_filename.startswith(data_filename)
    if args.task_format == 'verification': 
        evaluator = VerificationEvaluator(args.data_file)
    elif args.task_format == 'tagging': 
        evaluator = TaggingEvaluator(args.data_file) 

    # load predictions
    results = load_predictions_and_labels(args.predictions_file, evaluator)

    if not args.no_compute_metrics:
        problem_meta = get_problem_metadata(args.problems_file)
        standards_dict = get_standard_desc_dict(args.standards_file)

        # check if output directory exists
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)

        print("Computing metrics...")
        metric_dict = {}

        metrics = evaluator.calculate_overall_stats(results)
        for m in metrics: 
            metric_dict[m] = metrics[m]
        subsetted_data = evaluator.subset_data(results, problem_meta, standards_dict) 
        metric_dict['subsets'] = evaluator.calculate_subsetted_stats(subsetted_data)

        output_file = os.path.join(args.output_path, 'results_'+os.path.basename(args.predictions_file)) 
        with open(output_file, "w") as f:
            f.write(json.dumps(metric_dict))

    if args.interactive:
        if args.task_format == 'tagging': 
            raise NotImplementedError("Sorry!")
        option_pools = evaluator.get_correct_incorrect(results)
        options = sorted(list(option_pools.keys()))
        while True: 
            user_choice = input("What types of model responses do you want to view? Type one of the following: " + ', '.join(options) + ".\n- ")
            if user_choice in options: 
                pool = option_pools[user_choice]
                if args.num_examples < len(pool):
                    sample = random.sample(pool, args.num_examples)
                else:
                    sample = pool
                for example in sample: 
                   print(example)
                   pprint.pprint(results[example])
                   print('-------------------')
                   print()
            else: 
                print("Please type one of " + ', '.join(options) + ".") 
