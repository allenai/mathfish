"""
In our paper, we ran GPT-4 on "self-guided traversal", where
a model follows its own tagging decisions for each level of the tagging tree. 

This script takes the models' responses and compares the final output
against gold labels. 
"""
import os
from mathfish.preprocessors import DataExpander
from mathfish.evaluators import TaggingEvaluator
from mathfish.tree_retriever import TreeRetriever, ATCMap
import json
from mathfish.utils import get_options, get_domain_cat, get_grade
from collections import defaultdict, Counter
import numpy as np
import networkx as nx

def get_correct_standards(data_file):
    with open(data_file, "r") as f:
        instances = [json.loads(x) for x in f.readlines()]

    standards_path = '/net/nfs.cirrascale/s2-research/lucyl/edreports/Achieve_the_Core/standards.jsonl'
    expander = DataExpander(standards_path=standards_path)
    correct_answers = {}
    for i, instance in enumerate(instances):
        idx = instance['id']
        new_standards = expander.inherit_to_standard_level(instance['standards'])
        pos_standards = set([tup[1] for tup in new_standards if tup[0] in ['Alignment', 'Addressing']])
        if len(pos_standards) < 1:
            # No standards listed, will not use this instance
            continue
        correct_answers[idx] = pos_standards 

    return correct_answers

def example_options(model_input_file): 
    examples_to_options = {} # example ID to options
    with open(model_input_file, 'r') as infile: 
        for line in infile: 
            d = json.loads(line)
            assert d['id'] not in examples_to_options
            this_options = d['messages'][0]['options']
            examples_to_options[d['id']] = this_options

    return examples_to_options

def get_predicted_standards(response_file, evaluator, examples_to_options, options, retriever): 
    choices = defaultdict(set) # problem ID to chosen standard
    with open(response_file, 'r') as infile: 
        for line in infile: 
            d = json.loads(line)
            this_options = examples_to_options[d['id']]
            problem_id = '_'.join(d['id'].split('_')[:-2])
            response = evaluator.clean_output(d['model-response'])
            if type(response) != list: 
                # response not good format
                continue
            for letter in response: 
                if letter not in options: 
                    # hallucinated response option
                    print(response)
                    continue
                idx = options.index(letter)
                if idx > len(this_options) -1: 
                    # hallucinated response option
                    continue
                chosen = this_options[idx]
                standard = retriever.standards_descriptions_rev[chosen]
                assert standard
                choices[problem_id].add(standard)
    return choices

def calculate_metrics(correct_answers, predicted_answers, evaluator, retriever, atc_map):
    '''
    This calculates the following: 
    - exact and weak accuracy of standards per problem
    - average # of true and predicted standards labels per problem
    - exact accuracy across domains
    - domain, grade, and ATC map similarity of predicted and true standard labels
    '''
    results = defaultdict(dict)
    correct_len = []
    predicted_len = []
    in_true_dc = 0
    in_true_gr = 0
    total_s = 0
    min_dists = []
    no_path_count = 0
    dc_labels = defaultdict(Counter)
    for problem_id in correct_answers: 
        results[problem_id]['label'] = sorted(correct_answers[problem_id])
        results[problem_id]['prediction'] = sorted(predicted_answers[problem_id])
        true_dc = set([retriever.domain_cat_to_domain_group[get_domain_cat(s)] for s in correct_answers[problem_id]])
        true_gr = set([get_grade(s) for s in correct_answers[problem_id]])
        for dc in true_dc: 
            if results[problem_id]['label'] == results[problem_id]['prediction']:  
                dc_labels[dc]['correct'] += 1
            dc_labels[dc]['total'] += 1 
        for s in predicted_answers[problem_id]: 
            total_s += 1
            dc = retriever.domain_cat_to_domain_group[get_domain_cat(s)]
            if dc in true_dc: 
                in_true_dc += 1
            gr = get_grade(s)
            if gr in true_gr: 
                in_true_gr += 1
            min_dist = float("inf")
            for label in correct_answers[problem_id]: 
                try:
                    dist = atc_map.get_distance(s, label)
                    min_dist = min(dist, min_dist)
                except nx.NetworkXNoPath:
                    continue
            if min_dist != float("inf"): 
                min_dists.append(min_dist)
            else: 
                no_path_count += 1
        
        predicted_len.append(len(predicted_answers[problem_id]))
        correct_len.append(len(correct_answers[problem_id]))

    res = evaluator.calculate_overall_stats(results)
    print(res)
    print(np.mean(predicted_len), np.mean(correct_len))
    print("% of the time in same domain as true standards:", in_true_dc / total_s)
    print("% of the time in same grade as true standards:", in_true_gr / total_s)
    print("Minimum distance on graph:", np.mean(min_dists))
    print("No path count:", no_path_count / total_s)

    print("Average min distance on graph:")
    subgraphs = nx.connected_components(atc_map.undir_graph)
    for sb in subgraphs:
        g = atc_map.undir_graph.subgraph(sb)
        if len(g) > 2:
            print('\t', len(g), nx.average_shortest_path_length(g))

    print('\nAccuracy by domain group:')
    for dc in dc_labels: 
        print(dc, dc_labels[dc]['correct'] / dc_labels[dc]['total'])

def calculate_none(evaluator): 
    '''
    How many cases where the correct answer is "none" does the model actually output none?
    '''
    response_folder = '/net/nfs.cirrascale/s2-research/tala/edreports/output/data_files/model_responses/lucy_outputs/precursor'
    for level in ['standard', 'cluster']:
        print('\n' + level)
        dead_end_predicted = 0
        dead_end_total = 0 
        response_file = f'data_treetagger-dev_precursor-{level}_prompt-Promptv1_table-markdown_turns-single_shots-3.jsonl_model-gpt-4-turbo_truncate-True_samples-all_retries_3_temp-default_response.jsonl' 
        response_file = os.path.join(response_folder, response_file)
        responses = {}
        with open(response_file, 'r') as infile: 
            for line in infile: 
                d = json.loads(line)
                responses[d['id']] = evaluator.clean_output(d['model-response'])

        model_input_file = f'/home/lucyl/edreports/precursor_datafiles/data_treetagger-dev_precursor-{level}_prompt-Promptv1_table-markdown_turns-single_shots-3.jsonl' 
        with open(model_input_file, 'r') as infile: 
            for line in infile: 
                d = json.loads(line)
                correct_options = d['messages'][-1]['correct_option_index'] 
                if not correct_options: 
                    # correct answer should be none
                    dead_end_total += 1
                    if responses[d['id']] == []: 
                        dead_end_predicted += 1
        print("Dead ends predicted:", dead_end_predicted)
        print("Dead ends total:", dead_end_total)
        print("Fraction of correct dead ends:", dead_end_predicted/dead_end_total)

if __name__=="__main__":
    data_file = '/net/nfs.cirrascale/s2-research/tala/edreports/output/dev.jsonl'
    response_folder = '/net/nfs.cirrascale/s2-research/tala/edreports/output/data_files/model_responses/lucy_outputs/precursor'
    response_file = 'data_treetagger-dev_precursor-standard_prompt-Promptv1_table-markdown_turns-single_shots-3.jsonl_model-gpt-4-turbo_truncate-True_samples-all_retries_3_temp-default_response.jsonl'
    response_file = os.path.join(response_folder, response_file)
    model_input_file = '/home/lucyl/edreports/precursor_datafiles/data_treetagger-dev_precursor-standard_prompt-Promptv1_table-markdown_turns-single_shots-3.jsonl'

    standards_path = '/net/nfs.cirrascale/s2-research/lucyl/edreports/Achieve_the_Core/standards.jsonl'
    domain_groups_path = '/net/nfs.cirrascale/s2-research/lucyl/edreports/Achieve_the_Core/domain_groups.json'
    retriever = TreeRetriever(standards_path, domain_groups_path)
    atc_map = ATCMap(standards_path)
    atc_map.create_undirected_graph()

    evaluator = TaggingEvaluator(model_input_file)
    examples_to_options = example_options(model_input_file) 
    options = get_options()

    correct_answers = get_correct_standards(data_file)
    predicted_answers = get_predicted_standards(response_file, evaluator, examples_to_options, options, retriever)

    assert len(set(predicted_answers.keys()) - set(correct_answers.keys())) == 0

    calculate_metrics(correct_answers, predicted_answers, evaluator, retriever, atc_map)
    calculate_none(evaluator)
    
