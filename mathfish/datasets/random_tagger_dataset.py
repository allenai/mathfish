"""
Create datasets for tagging task format
for random standards only (e.g. not conditioned
standards' domains or clusters)

Author: Lucy Li
Email: lucyl@allenai.org
"""

from mathfish.tree_retriever import TreeRetriever
import json
import os
import random
from mathfish.datasets import BaseDataset
import re

class RandomTaggerDataset(BaseDataset): 
    # In the future we can consider having this inherit BaseDataset
    def __init__(self, standards_path: str, prompt_name: str, prompt_file: str, domain_groups_path: str, 
                 num_options: int, random_seed=0, table_style='special_token', image_style='special_token', 
                 shuffle_options=True):
        '''
        @inputs: 
        - standards_path: path to standards.jsonl
        - prompt_name: name of prompt.
        - prompt_file
        - domain_groups_path: path to domain groups file (not really needed for this dataset, but needed to run TreeRetriever)
        - num_options: number of options to provide to model per example
        - random_seed: random seed for random state of TreeRetriever
        - shuffle_options: whether to random shuffle the options provided
        '''
        super().__init__(standards_path, prompt_name, prompt_file, table_style=table_style, image_style=image_style)

        if self.prompt_name not in self.tagging_prompts: 
            raise BaseException("no prompt, ", self.prompt_name)
        
        self.retriever = TreeRetriever(standards_path, domain_groups_path, random_seed=random_seed)

        prompt_elements = {
            'level': "standards",
            'Level': "Standard",
            'relation': "aligns with",
        }

        prompt_dict = self.tagging_prompts[self.prompt_name]['promptTemplate']
        body = prompt_dict['body']
        relation_definition = prompt_dict['relation_definition']['standard']
        body = body.format(level=prompt_elements['level'], 
                           relation=prompt_elements['relation'], 
                           relation_definition=relation_definition)
        prompt = prompt_dict['single_wrap'].format(level=prompt_elements['level'], 
                                                   problem_activity='{problem_activity}',
                                                   options='{options}',
                                                   relation=prompt_elements['relation'],
                                                   body=body,
                                                   Level=prompt_elements['Level'])
        assert set(re.findall("{(.+?)}", prompt)) == set(['options', 'problem_activity'])

        self.prompt_template = prompt
        self.response_template = self.tagging_prompts[self.prompt_name]['responseTemplate']
        self.standard_examples = []
        self.shuffle_options = shuffle_options
        self.num_options = num_options
        self.response_format = self.tagging_prompts[self.prompt_name]['responseFormat']
    
    def get_example_count(self):
        return len(self.standard_examples)

    def _make_standard_level(self): 
        for i, instance in enumerate(self.instances):
            new_standards = self.expander.inherit_to_standard_level(instance['standards'])
            pos_standards = [tup[1] for tup in new_standards if tup[0] in ['Alignment', 'Addressing']]
            if len(pos_standards) < 1:
                # No standards listed, will not use this instance
                continue
            option_list = self.retriever.get_random_standards(pos_standards, self.num_options)
            e = {}
            e['id'] = instance['id'] + '_standard_' + str(i)
            e['dataset'] = 'randomtagger-standard'
            user_message = {}
            user_message['role'] = 'user'
            user_message['prompt_template'] = self.prompt_template
            user_message['options'] = option_list
            user_message['problem_activity'] = self.clean_problem_text(instance['text'], instance['elements'])
            assist_message = {}
            assist_message['role'] = 'assistant'
            assist_message['response_template'] = self.response_template
            assist_message['response_format'] = self.response_format
            indices = []
            for i, s_descript in enumerate(option_list): 
                s = self.retriever.get_pointer_to_next_branch(s_descript, 'standard')
                if s in pos_standards: 
                    indices.append(i)
            if len(pos_standards) < self.num_options: 
                assert len(indices) > 0
                assert len(pos_standards) == len(indices)
            else: 
                assert len(indices) == self.num_options
            assist_message['correct_option_index'] = indices
            e['messages'] = [user_message, assist_message]
            self.standard_examples.append(e)

    def make_all_examples(self): 
        self._make_standard_level()

    def output_dataset(self, output_path: str, input_prefix: str):
        '''
        @output_path: path to output

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
        output_file = os.path.join(output_path, 'data_randomtagger-{input_prefix}_options-{num_options}_prompt-{prompt}_table-{table_style}.jsonl'.format(num_options=self.num_options, 
                                                                                                               input_prefix=input_prefix,
                                                                                                               prompt=self.prompt_name,
                                                                                                               table_style=self.table_style)) 
        examples = self.standard_examples
        random.shuffle(examples)

        with open(output_file, mode='wt', encoding='utf-8') as outfile:
            for e in examples:
                outfile.write(json.dumps(e) + "\n")
        