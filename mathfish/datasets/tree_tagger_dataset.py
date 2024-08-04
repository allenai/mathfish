"""
Create datasets for tagging task format.

Author: Lucy Li
Email: lucyl@allenai.org
"""

from mathfish.utils import get_domain_cat
from mathfish.tree_retriever import TreeRetriever
import json
import os
import random
from mathfish.datasets import BaseDataset
import re
import string
from collections import defaultdict
import csv
import numpy as np
import ast

class TreeTaggerDataset(BaseDataset): 
    # In the future we can consider having this inherit BaseDataset
    def __init__(self, standards_path: str, prompt_name: str, prompt_file: str, domain_groups_path: str, multi_turn: bool,
                 table_style='special_token', image_style='special_token', 
                 random_seed=0, n_shots=0, few_shot_file=None,
                 domain_descriptions=True, shuffle_options=True):
        '''
        @inputs: 
        - standards_path: path to standards.jsonl
        - prompt_name: name of prompts
        - prompt_file: the file from which we load prompts
        - domain_groups_path: a path containing information about domain groups
        - multi_turn: whether to make the outputs multi turn (True) or single turn (False)
        - domain_descriptions: whether to include domain descriptions. It's recommended to keep this True.
        - shuffle_options: whether to shuffle options. It's recommended to keep this True. 
        '''
        super().__init__(standards_path, prompt_name, prompt_file, table_style=table_style, image_style=image_style)
        assert n_shots in set([0, 1, 3])

        if prompt_name not in self.tagging_prompts: 
            raise BaseException("no prompt, ", prompt_name)
        
        if n_shots > 0: 
            assert few_shot_file
            self.few_shot_exemplars = {}
            self.random_state = np.random.RandomState(random_seed)
            with open(few_shot_file, 'r') as infile: 
                reader = csv.DictReader(infile)
                for row in reader: 
                    assert row['grade'] not in self.few_shot_exemplars # no repeats
                    self.few_shot_exemplars[row['grade']] = row
        
        self.retriever = TreeRetriever(standards_path, domain_groups_path, random_seed=random_seed)

        prompt_elements = {
            'standard': {
                'level': "standards",
                'Level': "Standard",
                'relation': "aligns with",
            },
            'cluster': {
                'level': "mathematical concepts/skills",
                'Level': "Mathematical concept/skill",
                'relation': "teaches",
            },
            'domain': {
                'level': "topics",
                'Level': "Topic",
                'relation': "teaches",
            }
        }

        self.prompt_templates = {}
        self.few_shot_templates = {}
        self.examples = {}
        self.multi_turn = multi_turn
        self.n_shots = n_shots
        for level in prompt_elements: 
            prompt_dict = self.tagging_prompts[self.prompt_name]['promptTemplate']
            body = prompt_dict['body']
            relation_definition = prompt_dict['relation_definition'][level]
            body = body.format(level=prompt_elements[level]['level'], 
                            relation=prompt_elements[level]['relation'], 
                            relation_definition=relation_definition)
            if n_shots > 0: 
                # few-shot prompting
                body += '\n\n{few_shots}'
            if self.multi_turn and level in set(['cluster', 'standard']): 
                prompt = prompt_dict['multi_wrap'].format(level=prompt_elements[level]['level'], 
                                                    options='{options}',
                                                    relation=prompt_elements[level]['relation'],
                                                    body=body,
                                                    Level=prompt_elements[level]['Level'])
                assert set(re.findall("{(.+?)}", prompt)).issubset(set(['options', 'few_shots']))
            else: 
                prompt = prompt_dict['single_wrap'].format(level=prompt_elements[level]['level'], 
                                                    problem_activity='{problem_activity}',
                                                    options='{options}',
                                                    relation=prompt_elements[level]['relation'],
                                                    body=body,
                                                    Level=prompt_elements[level]['Level'])
                assert set(re.findall("{(.+?)}", prompt)).issubset(set(['options', 'problem_activity', 'few_shots']))
            if self.multi_turn: 
                if level == 'cluster': 
                    pre_prompt = prompt_dict['mid_wrap'].format(
                        level=prompt_elements[level]['level'], 
                        Level=prompt_elements[level]['Level'],
                        options='{options}',
                        relation=prompt_elements[level]['relation'], 
                    ) 
                    self.prompt_templates['pre_' + level] = pre_prompt 
                elif level == 'domain': 
                    pre_prompt = prompt_dict['top_wrap'].format(
                        level=prompt_elements[level]['level'],
                        problem_activity='{problem_activity}', 
                        Level=prompt_elements[level]['Level'],
                        options='{options}',
                        relation=prompt_elements[level]['relation'], 
                    ) 
                    self.prompt_templates['pre_' + level] = pre_prompt 
            few_shot_temps = prompt_dict['few_shot'].replace('{Level}', prompt_elements[level]['Level']) # to avoid escaping format later
            few_shot_conclusion = prompt_dict['few_shot_conclusion'].format(level=prompt_elements[level]['level'], 
                                                                            relation=prompt_elements[level]['relation'])  
            self.prompt_templates[level] = prompt
            self.few_shot_templates[level] = (few_shot_temps, few_shot_conclusion)
            self.examples[level] = []

        self.response_template = self.tagging_prompts[prompt_name]['responseTemplate']
        self.domain_descriptions = domain_descriptions
        self.shuffle_options = shuffle_options
        self.response_format = self.tagging_prompts[self.prompt_name]['responseFormat']

    def _make_exemplar(self, temp, choice, level, i):
        '''
        @inputs: 
        - temp: template for inserting exemplar 
        - choice: the chosen exemplar. this contains exemplar problem/activity text, elements, options, answers
        - level: level (domain, cluster, standard) of the exemplar 
        - i: example #
        '''
        example_problem_activity = self.clean_problem_text(choice['text'], json.loads(choice['elements'])) 
        example_options = self.format_options(ast.literal_eval(choice[level + '_options'])) 
        example_response = self.format_indices_as_comma_list(ast.literal_eval(choice[level + '_answers'])) 
        example_thought = choice[level + '_explanation'].strip()
        if i: 
            this_example = temp.format(example_problem_activity=example_problem_activity,
                                                          example_options=example_options,
                                                          example_response=example_response,
                                                          example_thought=example_thought,
                                                          i=i)
        else: 
            this_example = temp.format(example_problem_activity=example_problem_activity,
                                                        example_options=example_options,
                                                        example_response=example_response,
                                                        example_thought=example_thought)
        return this_example 

    def make_shots(self, level):
        example_text = ''
        temp = self.few_shot_templates[level][0] 
        example_meta = []
        if self.n_shots == 1: 
            keys = ['K', '1', '2', '3', '4', '5', '6', '7', '8', 'HS_0']
            choice = self.random_state.choice(keys)
            choice = self.few_shot_exemplars[choice]
            temp = temp.replace('Example {i}:', 'Example:')
            this_example = self._make_exemplar(temp, choice, level, None)
            example_meta.append([choice['id'], choice['grade']])
            example_text += this_example + '\n\n'
        elif self.n_shots == 3: 
            # sample {K, 3, 5}, {6, 7, 8}, or {HS, HS, HS}
            possibilities = [('K', '7', 'HS_0'), ('K', '3', '5'), ('6', '7', '8'), ('HS_0', 'HS_1', 'HS_2')]
            choice_idx = self.random_state.choice(range(len(possibilities)))
            choices = possibilities[choice_idx]
            these_shots = [self.few_shot_exemplars[choice] for choice in choices]
            for i, choice in enumerate(these_shots): 
                this_example = self._make_exemplar(temp, choice, level, i+1) 
                example_meta.append([choice['id'], choice['grade']])
                example_text += this_example + '\n\n'
        example_text += self.few_shot_templates[level][1]
        return example_text, example_meta
    
    def _make_example(self, instance, option_list, level, correct_set, idx): 
        '''
        @inputs
        - instance: a problem/activity instance
        - option_list: list of options provided by TreeRetriever
        - level: "domain", "cluster", or "standard"
        - correct_set: the correct domain groups / clusters / standards
        - idx: to create unique IDs for each task example, e.g. 
        without this we can't account for cases where we "fork" in the tree
        because a problem is associated with multiple domain groups or multiple
        clusters. 
        '''
        e = {}
        e['id'] = instance['id'] + '_' + level + '_' + str(idx)
        e['dataset'] = 'treetagger-' + level
        user_message = {}
        user_message['role'] = 'user'
        prompt_template = self.prompt_templates[level]
        if self.n_shots > 0: 
            few_shots, shot_info = self.make_shots(level)
            user_message['few_shots'] = few_shots
            user_message['shot_info'] = shot_info
        user_message['prompt_template'] = prompt_template
        user_message['options'] = option_list
        if not self.multi_turn or (self.multi_turn and level == 'domain'):
            user_message['problem_activity'] = self.clean_problem_text(instance['text'], instance['elements'])
        assist_message = {}
        assist_message['role'] = 'assistant'
        assist_message['response_template'] = self.response_template
        assist_message['response_format'] = self.response_format
        indices = []
        for i, descript in enumerate(option_list): 
            branch = self.retriever.get_pointer_to_next_branch(descript, level)
            if type(branch) == list and set(branch) & correct_set:
                indices.append(i)
            elif type(branch) == str and branch in correct_set: 
                indices.append(i)
        assist_message['correct_option_index'] = indices
        e['messages'] = [user_message, assist_message] 
        return e

    def make_domain_level(self, instance, pos_domain_groups): 
        '''
        Show the problem/activity and an option list involving a fixed set of domain options
        '''
        option_list = self.retriever.get_list_of_domains(give_description=self.domain_descriptions, 
                                                                        shuffle_options=self.shuffle_options)
        e = self._make_example(instance, option_list, 'domain', pos_domain_groups, 0)
        self.examples['domain'].append(e)

    def format_options(self, option_list): 
        options = list(string.ascii_uppercase) + ['A' + i for i in string.ascii_uppercase]
        option_string = ''
        for i in range(len(option_list)): 
            option_string += options[i] + '. ' + option_list[i] + '\n'
        return option_string
    
    def format_indices_as_comma_list(self, indices): 
        options = list(string.ascii_uppercase) + ['A' + i for i in string.ascii_uppercase]
        answer = ', '.join([options[i] for i in indices])
        return answer 

    def format_indices(self, indices):
        '''
        Example: 
            indices = [0, 2]
            answer = "A, C" or {{ "answer": "A, C" }}
        We need double brackets for json so it doesn't get replaced when calling str.format()
        '''
        options = list(string.ascii_uppercase) + ['A' + i for i in string.ascii_uppercase]
        answer = ', '.join([options[i] for i in indices])
        if self.response_format == 'json': 
            return "{\n  \"answer\": \""+ answer +"\"\n}"
        else: 
            return answer

    def _make_previous_domain_level(self, instance, correct_set): 
        '''
        Show the problem/activity and an option list involving a fixed set of domain options
        Used in the multi-turn setup where the domain level is part of a series of "previous" messages 
        '''
        option_list = self.retriever.get_list_of_domains(give_description=self.domain_descriptions, 
                                                                        shuffle_options=self.shuffle_options)
        user_message = {}
        user_message['role'] = 'user'
        user_message['prompt_template'] = self.prompt_templates['pre_domain']
        user_message['options'] = option_list
        user_message['problem_activity'] = self.clean_problem_text(instance['text'], instance['elements'])
        assist_message = {}
        assist_message['role'] = 'assistant'
        indices = []
        for i, descript in enumerate(option_list): 
            branch = self.retriever.get_pointer_to_next_branch(descript, 'domain')
            if type(branch) == list and set(branch) & correct_set:
                indices.append(i)
            elif type(branch) == str and branch in correct_set: 
                indices.append(i)
        assert len(indices) > 0
        assist_message['content'] = self.format_indices(indices)
        return [user_message, assist_message] 

    def make_cluster_level(self, instance, pos_domain_groups, pos_clusters):
        '''
        For each domain_group in pos_domain_groups: 
            Show the problem/activity and an option list involving a set of cluster options
        ''' 
        if self.multi_turn: 
            earlier_turns = []
            domain_turn = self._make_previous_domain_level(instance, pos_domain_groups)
            earlier_turns.extend(domain_turn)
        else: 
            earlier_turns = []
        for idx, domain_group in enumerate(pos_domain_groups):
            if domain_group == 'Modeling': 
                continue
            option_list = self.retriever.get_possible_clusters(domain_group, shuffle_options=self.shuffle_options) 
            e = self._make_example(instance, option_list, 'cluster', pos_clusters, idx)
            e['messages'] = earlier_turns + e['messages']
            self.examples['cluster'].append(e)

    def _make_previous_cluster_level(self, domain_group, correct_set):
        ''''
        Show the problem/activity and an option list of clusters
        Used in the multi-turn setup where the cluster level is part of a series of "previous" messages 
        ''' 
        option_list = self.retriever.get_possible_clusters(domain_group, shuffle_options=self.shuffle_options)
        user_message = {}
        user_message['role'] = 'user'
        user_message['prompt_template'] = self.prompt_templates['pre_cluster']
        user_message['options'] = option_list
        assist_message = {}
        assist_message['role'] = 'assistant'
        indices = []
        for i, descript in enumerate(option_list): 
            branch = self.retriever.get_pointer_to_next_branch(descript, 'cluster')
            if type(branch) == list and set(branch) & correct_set:
                indices.append(i)
            elif type(branch) == str and branch in correct_set: 
                indices.append(i)
        assert len(indices) > 0
        assist_message['content'] = self.format_indices(indices)
        return [user_message, assist_message] 

    def make_standard_level(self, instance, pos_domain_groups, pos_clusters, pos_standards):
        '''
        For each cluster in pos_clusters: 
            Show the problem/activity and an option list involving a set of standards
        ''' 
        if self.multi_turn: 
            domain_turn = self._make_previous_domain_level(instance, pos_domain_groups)
        else: 
            domain_turn = []
            earlier_turns = []
        for idx, cluster in enumerate(pos_clusters):
            if self.multi_turn:
                this_domain_cat = get_domain_cat(cluster) 
                this_domain_group = self.retriever.domain_cat_to_domain_group[this_domain_cat]
                cluster_turn = self._make_previous_cluster_level(this_domain_group, pos_clusters)
                earlier_turns = domain_turn + cluster_turn
            option_list = self.retriever.get_possible_standards(cluster, shuffle_options=self.shuffle_options)
            e = self._make_example(instance, option_list, 'standard', pos_standards, idx)
            e['messages'] = earlier_turns + e['messages']
            self.examples['standard'].append(e)

    def get_example_positives(self): 
        modeling_standards = self.retriever.get_modeling_standards()
        ret = defaultdict(dict)
        for i, instance in enumerate(self.instances):
            idx = instance['id']
            new_standards = self.expander.inherit_to_standard_level(instance['standards'])
            pos_standards = set([tup[1] for tup in new_standards if tup[0] in ['Alignment', 'Addressing']])
            if len(pos_standards) < 1:
                # No standards listed, will not use this instance
                continue
            
            pos_domain_cats = set([get_domain_cat(s) for s in pos_standards])
            if pos_standards & modeling_standards: 
                pos_domain_cats.add('M')
            pos_domain_groups = set([self.retriever.domain_cat_to_domain_group[dc] for dc in pos_domain_cats])
            pos_clusters = set(['.'.join(s.split('.')[:-1]) for s in pos_standards])

            ret[idx]['pos_domain_cats'] = pos_domain_cats
            ret[idx]['pos_domain_groups'] = pos_domain_groups
            ret[idx]['pos_clusters'] = pos_clusters
            ret[idx]['pos_standards'] = pos_standards
        return ret

    def make_all_examples(self): 
        modeling_standards = self.retriever.get_modeling_standards()
        for i, instance in enumerate(self.instances):
            new_standards = self.expander.inherit_to_standard_level(instance['standards'])
            pos_standards = set([tup[1] for tup in new_standards if tup[0] in ['Alignment', 'Addressing']])
            if len(pos_standards) < 1:
                # No standards listed, will not use this instance
                continue
            
            pos_domain_cats = set([get_domain_cat(s) for s in pos_standards])
            if pos_standards & modeling_standards: 
                pos_domain_cats.add('M')
            pos_domain_groups = set([self.retriever.domain_cat_to_domain_group[dc] for dc in pos_domain_cats])
            pos_clusters = set(['.'.join(s.split('.')[:-1]) for s in pos_standards])

            self.make_domain_level(instance, pos_domain_groups)
            self.make_cluster_level(instance, pos_domain_groups, pos_clusters)
            self.make_standard_level(instance, pos_domain_groups, pos_clusters, pos_standards)

    def get_example_count(self):
        return self.get_domain_example_count() + self.get_cluster_example_count() + self.get_standard_example_count()
    
    def get_domain_example_count(self): 
        return len(self.examples['domain'])
    
    def get_cluster_example_count(self): 
        return len(self.examples['cluster'])
    
    def get_standard_example_count(self): 
        return len(self.examples['standard'])

    def output_dataset(self, output_path: str, input_prefix: str, precursor='none'):
        '''
        Outputs are currently in the format of
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
        '''
        if self.multi_turn: 
            turn_label = 'multi'
        else: 
            turn_label = "single"

        if self.n_shots > 0: 
            output_file = os.path.join(output_path, 'data_treetagger-{input_prefix}_precursor-{precursor}_prompt-{prompt}_table-{table_style}_turns-{turn_label}_shots-{n_shots}.jsonl'.format(prompt=self.prompt_name, 
                                                                                                                                        input_prefix=input_prefix,
                                                                                                                                        precursor=precursor,
                                                                                                                                        table_style=self.table_style,
                                                                                                                                        turn_label=turn_label,
                                                                                                                                        n_shots=self.n_shots)) 
        else: 
            # this if statement is mostly here so that if we ever want to rerun previous experiemnts, we match the naming convention of those experiments 
            output_file = os.path.join(output_path, 'data_treetagger-{input_prefix}_precursor-{precursor}_prompt-{prompt}_table-{table_style}_turns-{turn_label}.jsonl'.format(prompt=self.prompt_name, 
                                                                                                                                        input_prefix=input_prefix,
                                                                                                                                        precursor=precursor,
                                                                                                                                        table_style=self.table_style,
                                                                                                                                        turn_label=turn_label)) 
        examples = self.examples['standard'] + self.examples['cluster'] + self.examples['domain']
        random.shuffle(examples)

        with open(output_file, mode='wt', encoding='utf-8') as outfile:
            for e in examples:
                outfile.write(json.dumps(e) + "\n")

        return output_file
        
