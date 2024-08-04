"""
Author: Tal August (@tala)
Email:  tala@allenai.org
"""


'''

Takes a dataset in jsonl format and runs a specified model with it.
Specifically the dataset jsonl file should have lines that look similar to this:

    {
    'dataset':<name of dataset>,
    'id':<id for the unique instance>,
    'messages': [
            {
                role: 'user',
                prompt_template: 'You are a math expert reviewing K-12 curricular materials. ... \n\nProblem/activity:\n{problem_activity}\n\nDescription:\n{standard_description}',
                description: '...',
                problem_activity: '...',
            }
            'role':'assistant', 'content':<correct_response>
        ]
    }

Depending on the task format (e.g., verification vs. tagging vs. multi-turn tagging) the format might look slightly different
The script will output a jsonl file where each line looks like:

    {
    'dataset':<name of dataset>,
    'id':<id for the unique instance>,
    'messages': [
            {'role':'user', 'content':<prompt, formatted based on the prompt template and provided strings/options>},
        ],
    'model_response':<model response>,
    'logs':[
        <log in jsonl format of model tries and truncation, if necessary>
    ]
}

example script:

python scripts/run_wrapper.py \
        --dataset_file prefix_to_a_set_of_files \
        --task_name verification \
        --use_openai_client \
        --model_name_or_path meta-llama/Llama-2-70b-chat-hf \
        --tokenizer_name_or_path meta-llama/Llama-2-70b-chat-hf \
        --output_path results/ \
        --truncate \
        --retry \
        --max_retries 5 \
        --n_samples -1 \
        --all_prompts \
        --shots 3
'''

from mathfish.modelWrappers import VerificationWrapper, TaggingWrapper
import os
from pathlib import Path
from mathfish.utils import *
import argparse
import json
from tqdm import tqdm
import random
import datetime
import time

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_file",
        type=str,
        help="Path to dataset prefix, which should be followed by zero, one, and three shot versions of the prompt.")

    parser.add_argument(
        "--task_name",
        type=str,
        choices=["verification", "tagger"],
        help="What task format to follow, defines which wrapper to use"
    )
    parser.add_argument(
        "--all_prompts",
        default=False,
        action='store_true',
        help="Whether to run all the best prompts for the model"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="model name or path to model"
    )
    parser.add_argument(
        "--use_openai_client",
        action='store_true',
        default=False,
        help="Whether to use OpenAI client instead of loading a Huggingface model"
    )

    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        help="tokenizer name or path to tokenizer"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="path to store model outputs"
    )
    parser.add_argument(
        "--retry",
        default=False,
        action='store_true',
        help="Whether to retry if first model response is not parseable"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=500,
        help="How many samples to run, defaults to 500. Put -1 if you want to run all samples in the dataset."
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=0,
        help="number of retries for model"
    )
    parser.add_argument(
        "--truncate",
        action='store_true',
        default=False,
        help="Whether or not to use custom truncatation logic, defaults to false"
    )
    parser.add_argument(
        "--temp",
        type=float,
        help="temperature for the model. None means default temperature."
    )

    parser.add_argument(
        "--multi_turn",
        action='store_true',
        default=False,
        help="Whether this is single or multi-turn"
    )

    parser.add_argument(
        "--shots",
        type=int,
        default=0,
        help="Whether to also run few-shot datafiles"
    )

    args = parser.parse_args()

    assert not (args.retry and args.max_retries <= 0), "Number of retries must be greater than 0 if you want to enable retries"
    assert not (args.multi_turn and args.task_name =='verification'), "No multi-turn for verification task."

    print('Running Arguments:', args)

    return args



def run_datafile(args, data_file, max_length):

    print(f"Running model on {data_file}")

    # check if input file exists
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"File {data_file} not found.")

    # check if output directory exists
    if args.output_path is not None:
        output_dir = os.path.dirname(args.output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # make output file
    file_name = "{data_file}_model-{model_name}_truncate-{truncate}_samples-{n_samples}_retries_{retries}_temp-{temp}_response.jsonl"
    output_file_name = file_name.format(data_file=Path(data_file).stem,
                                 model_name=model_name,
                                 truncate=args.truncate,
                                 n_samples=args.n_samples if args.n_samples != -1 else "all",
                                 retries=args.max_retries,
                                 temp=args.temp if args.temp is not None else "default")
    output_file = os.path.join(args.output_path, output_file_name)

    # check if file exists and is the same number of lines that we plan to read
    # if so, skip
    if os.path.exists(output_file):
        with open(output_file, 'r') as infile:
            lines = infile.readlines()
            if args.n_samples != -1 and len(lines) == args.n_samples:
                print(f"File {output_file} already exists and has the same number of lines as the dataset. Skipping.")
                return
            elif args.n_samples == -1 and len(lines) == len(open(data_file, 'r').readlines()):
                print(f"File {output_file} already exists and has the same number of lines as the dataset. Skipping.")
                return

    # clear output file
    open(output_file, 'w').close()

    # get our wrapper
    if args.task_name == "verification":
        wrapper = VerificationWrapper(openai_client=args.use_openai_client, model_name_or_path=args.model_name_or_path, tokenizer_name_or_path=args.tokenizer_name_or_path, truncate=args.truncate, max_length=max_length)
    elif args.task_name == "tagger":
        wrapper = TaggingWrapper(openai_client=args.use_openai_client, model_name_or_path=args.model_name_or_path, tokenizer_name_or_path=args.tokenizer_name_or_path, truncate=args.truncate, max_length=max_length, is_multi_turn=args.multi_turn)
    else:
        raise ValueError(f"Task name {args.task_name} not recognized. Please choose from ['verification', 'tagging']")


    # read in the dataset
    with open(data_file, 'r') as infile:
        lines = infile.readlines()
        if args.n_samples != -1:
            # lines = lines[:args.n_samples]
            lines = lines[46:48]

        i = 0
        total_calls = 0
        for line in tqdm(lines, desc="Processing dataset", total=len(lines)):
            d = json.loads(line)


            response = wrapper.call_model_api(d, retry=args.retry, num_retries=args.max_retries, temp=args.temp)
            total_calls += response['num_calls']
            i += 1

            # save the token_lengths and formatted messages
            with open(output_file, 'a', 10) as outfile:
                json.dump(response, outfile)
                outfile.write('\n')
            if i % 20 == 0:
                ts = time.time()
                print(i, total_calls, datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'), response['prompt-length'], response['completion-length'])

if __name__=="__main__":
    args = parse_args()

    # simple dict to define the right table format for each model
    if args.task_name == 'verification': 
        models_favorites = {'gpt-4-turbo':{'table':'markdown', 'prompts':['1', '4', '7'], 'length':'128000'}, 
                            'Llama-2-70b-chat-hf':{'table':'markdown', 'prompts':['4', '10', '15'], 'length':'4096'}, 
                            'Mixtral-8x7B-Instruct-v0.1':{'table':'json', 'prompts':['1', '5', '15'], 'length':'32768'}}
    elif args.task_name == 'tagger': 
        models_favorites = {'gpt-4-turbo':{'table':'markdown', 'prompts':['1', '4', '10'], 'length':'128000'},
                        'Llama-2-70b-chat-hf':{'table':'markdown', 'prompts':['3', '11', '12'], 'length':'4096'},
                        'Mixtral-8x7B-Instruct-v0.1':{'table':'json', 'prompts':['5', '6', '12'], 'length':'32768'}}

    # models_favorites = {'gpt-4-turbo':{'table':'markdown', 'prompts':['2', '3', '5', '6', '8', '9', '10', '11', '12', '13', '14', '15']},
    #                     'Llama-2-70b-chat-hf':{'table':'markdown', 'prompts':['1', '2', '3', '5', '6', '7', '8', '9', '11', '12', '13', '14', '15']},
    #                     'Mixtral-8x7B-Instruct-v0.1':{'table':'json', 'prompts':['2', '3', '4', '6', '7', '8', '9', '10', '11', '12', '13', '14']}}



    model_name = args.model_name_or_path.split('/')[1] if len(args.model_name_or_path.split('/')) > 1 else args.model_name_or_path
    assert model_name in models_favorites.keys(), f"Model {model_name} not recognized. Please choose from {models_favorites.keys()}"
    fav_table = models_favorites[model_name]['table']
    fav_prompts = models_favorites[model_name]['prompts']
    max_length = int(models_favorites[model_name]['length'])


    is_multi = 'multi' if args.multi_turn else 'single'

    if args.all_prompts:
        prompts = fav_prompts
    else:
        prompts = random.sample(fav_prompts, 1)

    # make the full dataset file path given the favorite table and prompt
    # e.g., data_verification-dev_nsample-5_grade-different_domain-different_prompt-Promptv10_table-json.jsonl
    dataset_path = Path(args.dataset_file)

    if args.task_name == 'verification': 
        for prompt in prompts: 
            if args.shots == 0: 
                data_file = f"{dataset_path.parent}/{dataset_path.stem}_prompt-Promptv{prompt}_table-{fav_table}.jsonl"
                run_datafile(args, data_file, max_length) 
            if args.shots == 1: 
                data_file = f"{dataset_path.parent}/{dataset_path.stem}_prompt-Promptv{prompt}_table-{fav_table}_shots-1.jsonl"
                run_datafile(args, data_file, max_length)
            if args.shots == 3: 
                data_file = f"{dataset_path.parent}/{dataset_path.stem}_prompt-Promptv{prompt}_table-{fav_table}_shots-3.jsonl"
                run_datafile(args, data_file, max_length) 

    if args.task_name == 'tagger': 
        for prompt in prompts:
            if args.shots == 0: 
                data_file = f"{dataset_path.parent}/{dataset_path.stem}_prompt-Promptv{prompt}_table-{fav_table}_turns-{is_multi}.jsonl"
                run_datafile(args, data_file, max_length)
            if args.shots == 1:
                data_file = f"{dataset_path.parent}/{dataset_path.stem}_prompt-Promptv{prompt}_table-{fav_table}_turns-{is_multi}_shots-1.jsonl"
                run_datafile(args, data_file, max_length)
            if args.shots == 3:
                data_file = f"{dataset_path.parent}/{dataset_path.stem}_prompt-Promptv{prompt}_table-{fav_table}_turns-{is_multi}_shots-3.jsonl"
                run_datafile(args, data_file, max_length)



