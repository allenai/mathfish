"""
Author: Tal August (@tala), Lucy Li
Email:  tala@allenai.org, lucyl@allenai.org
"""

from mathfish.preprocessors import StandardStandardizer, DataExpander
from mathfish.datasets import VerificationDataset
import os
import json
from collections import defaultdict, Counter
from tqdm import tqdm
import csv
from utils import *
import argparse
from typing import List
import re
import pathlib

'''

Outputs a dataset in tulu-chat format for instruction tuning and/or eval. 
    Specifically a jsonl file where each line looks like: 
    
    {
    'dataset':<name of dataset>
    'id':<id for the unique instance>
    'messages': [
            {'role':'user', 'content':<prompt>,
            'role':'assistant', 'content:<correct_response>
        ]
    }


example script:

python scripts/output_dataset_verification.py \
--input_file dev.jsonl  \
--output_path /output_folder/ \
--prompt_name Promptv1 \
--n_sample 1 \
--neg_strat all-negative-types \
--standards_path standards.jsonl \
--table_style markdown 
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
        help="path to achieve-the-core standards.jsonl"
    )

    parser.add_argument(
        "--is_eval",
        type=bool,
        default=True,
        help=" whether to include correct responses \
        (i.e., if the dataset is being used for eval or for instruction tuning)"
    )

    parser.add_argument(
        "--get_negatives",
        type=bool,
        default=True,
        help="whether to include negative examples in the dataset"
    )

    parser.add_argument(
        "--n_sample",
        type=int,
        default=1,
        help="number of samples to get from possible negative pool"
    )

    parser.add_argument(
        "--neg_strat",
        type=str,
        choices=['same-domain-same-grade', 'same-domain-different-grade', 
                 'different-domain-different-grade', 'different-domain-same-grade', 
                 'neighbors', 'all-negative-types', 'none'], 
        default='different-domain-different-grade',
        help='Strategy for sampling negative examples.'
    )

    parser.add_argument(
        "--table_style",
        type=str,
        choices=['special_token', 'html', 'json', 'rst', 'markdown'], 
        default='special_token',
        help='Table formatting.'
    )

    parser.add_argument(
        "--prompt_name",
        type=str,
        help="Name of prompt to use, e.g. \"Promptv1\"."
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
    # get_negatives should be true if n_sample, grade, or domain are included
    if (not args.get_negatives):
        assert (args.n_sample is None) and (args.grade is None) and (args.domain is None), "get_negatives should be set to true if n_sample, grade, and domain are included"

    return args

        
if __name__=="__main__": 
    args = parse_args()


    if args.n_shots > 0: 
        assert args.few_shot_file
        
    standards_path = args.standards_path
    prompt_file = pathlib.Path(__file__).parent.parent / "mathfish" / "datasets" / "prompts.json"
    expander = DataExpander(standards_path=standards_path)
    standardizer = StandardStandardizer(standards_path=standards_path)
    input_prefix = os.path.basename(args.input_file).replace('.jsonl', '')

    dataset = VerificationDataset(standards_path=standards_path, prompt_name=args.prompt_name, prompt_file=prompt_file, 
                                few_shot_file=args.few_shot_file, n_shots=args.n_shots,
                                n_sample=args.n_sample, neg_strat=args.neg_strat, table_style=args.table_style)

    dataset.load_instances(args.input_file)
    dataset.make_positive_examples()
    if args.neg_strat != 'none': 
        dataset.make_negative_examples()

    print('created total of {} positive and {} negative examples.'.format(dataset.get_pos_count(), dataset.get_neg_count()))

    # check if output directory exists
    if args.output_path is not None:
        output_dir = os.path.dirname(args.output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)


    dataset.output_dataset(args.output_path, input_prefix)