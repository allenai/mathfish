"""
Author: Lucy (lucy3)
Email: lucyl@allenai.org 
"""

import os
import json
from collections import defaultdict, Counter
from tqdm import tqdm
import csv
from utils import *
import argparse
from typing import List
import re
from mathfish.datasets import RandomTaggerDataset, TreeTaggerDataset
import pathlib

'''
Outputs a jsonl file that looks like the following.
    {
        dataset: dataset_name, 
        id: id, 
        messages: [ 
            {       # message 1
                role: 'user', 
                prompt_template: '',
                options: [], # these are options to include in prompt
                problem_activity: '',
            },
            {       # message 2
                role: 'assistant', 
                response_template: '',
                correct_option_index: int, # index in previous message's options
            }, # the last message is the "correct" final answer for a particular model run
        ]
    }


example commands:

python scripts/output_dataset_tagging.py \
--input_file dev.jsonl  \
--output_path /output_folder/ \
--standards_path standards.jsonl \
--domain_groups_path domain_groups.json \
--prompt_name Promptv1 \
--dataset_type single_tree
'''

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        help="path to data to convert into dataset. Should be a jsonl file.")

    parser.add_argument(
        "--output_path",
        type=str,
        help="path to store dataset. Dataset will be in jsonl format"
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
        "--num_options",
        type=int,
        default=5,
        help="Number of options to show for random standard tagging"
    )

    parser.add_argument(
        "--prompt_name",
        type=str,
        help="Name of prompt to use, e.g. \"Promptv1\"."
    )

    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=['random', 'single_tree', 'multi_tree'],
        help="What type of dataset to output. Multi_tree is a multi-turn setup where previous layers of " + \
                "the tree are prepended to prompts, while single_tree is the setup used in our paper. " + \
                "Random is a case where options random standards, and doesn't involve the standards hierarchy."
    )

    parser.add_argument(
        "--shuffle",
        type=bool,
        default=True,
        help="Whether to shuffle the options shown."
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

        
if __name__=="__main__": 
    args = parse_args()


    standards_path = args.standards_path
    domain_groups_path = args.domain_groups_path
    prompt_file = pathlib.Path(__file__).parent.parent / "mathfish" / "datasets" / "prompts.json"
    input_prefix = os.path.basename(args.input_file).replace('.jsonl', '')

    if args.dataset_type == 'random': 
        if args.n_shots > 0: 
            raise NotImplementedError("No few-shot examples yet for random tagger, sorry")
        dataset = RandomTaggerDataset(standards_path, args.prompt_name, prompt_file, domain_groups_path, 
                                    num_options=args.num_options, shuffle_options=args.shuffle, table_style=args.table_style)

        dataset.load_instances(args.input_file)
        dataset.make_all_examples()
        print('Created total of {} examples.'.format(dataset.get_example_count()))
    elif args.dataset_type == 'single_tree': 
        if args.n_shots > 0: 
            assert args.few_shot_file
        dataset = TreeTaggerDataset(standards_path, args.prompt_name, prompt_file, domain_groups_path, False, 
                                    few_shot_file=args.few_shot_file,
                                    shuffle_options=args.shuffle, table_style=args.table_style, n_shots=args.n_shots)
        dataset.load_instances(args.input_file)
        dataset.make_all_examples()
        print("Created a total of {} domain examples, {} cluster examples, and {} standard examples".format(dataset.get_domain_example_count(),
                                                                                                            dataset.get_cluster_example_count(),
                                                                                                            dataset.get_standard_example_count()))
    elif args.dataset_type == 'multi_tree':
        if args.n_shots > 0: 
            raise NotImplementedError("No few-shot examples yet for multi tree, sorry") 
        dataset = TreeTaggerDataset(standards_path, args.prompt_name, prompt_file, domain_groups_path, True,
                                    shuffle_options=args.shuffle, table_style=args.table_style)
        dataset.load_instances(args.input_file)
        dataset.make_all_examples()
        print("Created a total of {} domain examples, {} cluster examples, and {} standard examples".format(dataset.get_domain_example_count(),
                                                                                                            dataset.get_cluster_example_count(),
                                                                                                            dataset.get_standard_example_count()))

    # check if output directory exists
    if args.output_path is not None:
        output_dir = os.path.dirname(args.output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    print(args.output_path)

    dataset.output_dataset(args.output_path, input_prefix)