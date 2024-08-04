# Evaluating Language Model Math Reasoning via Grounding in Educational Curricula

This guide walks through how to run models on the tasks introduced in our paper, **TBD link**. 

## Installation

`pip install -e .`

## Obtaining data

On Huggingface, we release the following datasets: 
- [mathfish](https://huggingface.co/datasets/allenai/mathfish): preprocessed math problems from open educational resources (Illustrative Math and Fishtank Learning) labeled with standards. This includes train, dev, and test splits. In our paper, we only use the dev split for evaluating models' performance on assessing math standards alignment. 
- [mathfish-tasks](https://huggingface.co/datasets/allenai/mathfish-tasks): The dev subset of MathFish transformed into tasks for language models, formatted with prompts. These could be used to replicate our results. 
- mathfish-images: In MathFish, there are images interweaved with problems which we don't use in the paper, but may facilitate multimodal work for others. **TBD????** 
- [achieve-the-core](https://huggingface.co/datasets/allenai/achieve-the-core): Math standards, their descriptions, and metadata obtained from Achieve the Core. This includes `domain_groups.json` and `standards.jsonl`. 

On Github, we include the following additional data and output files: 
- `data/generated_problems_teacher_annotations.csv`: anonymized teachers' annotations of generated problems (study 1 in our paper)
- `data/gsm8k_test.jsonl`: GSM8k's test set, reformatted to match our problem input format. 

## Turning data into tasks

Two scripts transform data into two different task formats: verification and tagging. 

Usage example for creating verification inputs: 

```
python scripts/output_dataset_verification.py \
    --input_file dev.jsonl  \
    --output_path /output_folder/ \
    --prompt_name Promptv1 \
    --n_sample 1 \
    --neg_strat all-negative-types' \
    --standards_path standards.jsonl \
    --n_shots 1 \
    --few_shot_file mathfish/datasets/few_shot_verification_exp.csv \
    --table_style markdown 
```

In the above, `input_file` is a MathFish dataset file containing standards-labeled problems. The above `n_sample` and `neg_strat` parameters signify that we want to sample one example with each negative sampling strategy (different domain + different grade, different domain + same grade, same domain + different grade, same domain + same grade, and ATC neighbors). All prompts (15 possibilities total per task format) for `prompt_name` can be found in `mathfish/dataset/prompts.json`. 

Usage example for creating *assisted traversal* tagging inputs: 

```
python scripts/output_dataset_tagging.py \
    --input_file dev.jsonl  \
    --output_path /output_folder/ \
    --standards_path standards.jsonl \
    --domain_groups_path domain_groups.json \
    --prompt_name Promptv1 \
    --n_shots 1 \
    --few_shot_file mathfish/datasets/few_shot_tagging_exp.csv \
    --dataset_type single_tree
```

You can also turn non-MathFish problems into input files for running models on verification/tagging. For example, if you had a set of generated problems or problems from another dataset, format those problems into a `jsonl` with the following keys: `id`, `standards`, `text`, and `elements`. See `data/generated_problems_as_input_data.jsonl` for an example of this using the generated problems from our paper's study 1. 

## Generate model outputs

To run models on our tasks' prompts, we use an API wrapper that handles input truncation and retries (e.g. API connection errors or ill-formed responses). Our wrapper is implemented for three models: Llama 2 70B, GPT-4 turbo, and Mixtral 7x8B. This wrapper runs each model on tasks with optimized prompting preferences as described in our paper (e.g. `Promptv*` wording and table formatting). The `dataset_file` flag points to a collection of files outputted by "Turning data into tasks" above. 

Usage example for verification with Llama 2: 

```
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
```

Usage example for tagging with GPT-4: 

```
python scripts/run_wrapper.py \
          --dataset_file prefix_to_a_set_of_files \
          --task_name tagger \
          --use_openai_client \
          --model_name_or_path gpt-4-turbo \
          --tokenizer_name_or_path gpt-4-turbo \
          --output_path results/ \
          --truncate \
          --n_samples -1 \
          --all_prompts \
          --retry \
          --max_retries 3 \
          --shots 3
```

Each model output file name is prefixed by the name of their corresponding input prompt file, to make it easier to identify which outputs are generated by which inputs. 

Finally, here are two examples of running *self-guided* tagging: the first on MathFish's dev set and the second on GSM8k's test set. Note that this script handles both transforming problem data into the tagging task input, and calling models' API wrappers. 

```
python scripts/tree_traverser.py \
    --input_file dev.jsonl  \
    --output_path ../ \
    --response_output_path results/traversal/ \
    --prompt_name Promptv1 \
    --table_style markdown \
    --standards_path standards.jsonl \
    --domain_groups_path domain_groups.json \
    --n_shots 3 \
    --few_shot_file mathfish/datasets/few_shot_tagging_exp.csv
```

```
python scripts/tree_traverser.py \
    --input_file data/gsm8k_test.jsonl  \
    --output_path ../ \
    --response_output_path results/traversal/ \
    --prompt_name Promptv1 \
    --table_style markdown \
    --standards_path standards.jsonl \
    --domain_groups_path domain_groups.json \
    --n_shots 3 \
    --no_positives \
    --few_shot_file mathfish/datasets/few_shot_tagging_exp.csv
```

## Evaluating model outputs

Our eval script evaluates tagging or verification responses from models. Usage example: 

```
python scripts/eval.py \
    --predictions_file data_turned_into_verification_task_with_model_response.jsonl \
    --standards_file achieve-the-core/standards.jsonl \
    --problems_file dev.jsonl \
    --data_file data_turned_into_verification_task.jsonl \
    --output_path results/eval/ \
    --task_format verification
```

You can toggle `task_format` between `tagging` and `verification` as appropriate. The `predictions_file` should be the output of "Generate model outputs" above, and `problems_file` is a file containing standards-labeled problems. In addition, `data_file` is the output of "Turning data into tasks" above. 
