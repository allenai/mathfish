'''
Class for creating verification dataset

Author: Tal August, Lucy Li
Email:  tala@allenai.org, lucyl@allenai.org
'''
from mathfish.preprocessors import StandardStandardizer
import json 
import random
import os
from mathfish.datasets import BaseDataset
from collections import defaultdict
import numpy as np
import csv

class VerificationDataset(BaseDataset):
    def __init__(self, standards_path: str, prompt_name: str, prompt_file: str, n_sample=5,
                 n_shots=0, few_shot_file=None, random_seed=0,
                 neg_strat='all-negative-types', table_style='special_token', image_style='special_token'):
        '''
        @inputs: 
        - standards_path: path to standards.jsonl
        - prompt_name: name of prompt.
        - prompt_file
        - n_sample: number of negative samples
        - grade, domain, neighbors: negative sampling strategies. If neighbor=True then 
        we use Achieve the Core's neighbors on their graph of standard connections, 
        otherwise we sample from same/different grades and domains. 
        '''
        super().__init__(standards_path, prompt_name, prompt_file, table_style=table_style, image_style=image_style)
        assert n_shots in set([0, 1, 3])

        assert neg_strat in ['same-domain-same-grade', 'same-domain-different-grade', 
                 'different-domain-different-grade', 'different-domain-same-grade', 
                 'neighbors', 'all-negative-types', 'none']

        if self.prompt_name not in self.verification_prompts: 
            raise BaseException("no prompt, ", self.prompt_name)
        p = self.verification_prompts[self.prompt_name]

        self.n_shots = n_shots
        prompt_template = p['promptTemplate']
        self.response_template = p['responseTemplate'] 
        self.few_shot_templates = None
        if self.n_shots > 0:
            self.few_shot_templates = (p['few_shot'], p['few_shot_conclusion']) 
        else: 
            prompt_template = prompt_template.replace('{few_shots}', '')
        self.prompt_template = prompt_template

        if n_shots > 0: 
            assert few_shot_file
            self.few_shot_exemplars = defaultdict(list)
            self.random_state = np.random.RandomState(random_seed)
            with open(few_shot_file, 'r') as infile: 
                reader = csv.DictReader(infile)
                for row in reader: 
                    self.few_shot_exemplars[row['grade']].append(row)
            
        self.standardizer = StandardStandardizer(standards_path=standards_path)
        self.n_sample = n_sample
        self.neg_strat = neg_strat
        self.positive_examples = []
        self.negative_examples = []
    
    def get_example_count(self): 
        return len(self.positive_examples) + len(self.negative_examples) 

    def get_pos_count(self):
        return len(self.positive_examples)

    def get_neg_count(self):
        return len(self.negative_examples)
    
    def _make_exemplar(self, temp, choice, i):
        '''
        @inputs: 
        - temp: template for inserting problem
        - choice: the chosen exemplar. this contains exemplar problem/activity text, elements, options, answers
        - i: example #
        '''
        example_problem_activity = self.clean_problem_text(choice['text'], json.loads(choice['elements'])) 
        example_response = choice['aligns'].title().strip() + '.'
        example_thought = choice['explanation'].strip()
        descript = choice['standard_description']
        if i: 
            this_example = temp.format(example_problem_activity=example_problem_activity,
                                                          example_standard_description=descript,
                                                          example_answer=example_response,
                                                          example_thought=example_thought,
                                                          i=i)
        else: 
            this_example = temp.format(example_problem_activity=example_problem_activity,
                                                        example_standard_description=descript,
                                                        example_answer=example_response,
                                                        example_thought=example_thought)
        return this_example 

    def make_shots(self):
        example_text = ''
        temp = self.few_shot_templates[0] 
        example_meta = []
        if self.n_shots == 1: 
            keys = ['K', '1', '2', '3', '4', '5', '6', '7', '8', 'HS_0']
            choice = self.random_state.choice(keys)
            # choose neg or pos exemplar for that grade
            choice = self.random_state.choice(self.few_shot_exemplars[choice]) 
            temp = temp.replace('Example {i}:', 'Example:')
            this_example = self._make_exemplar(temp, choice, None)
            example_meta.append([choice['id'], choice['grade']])
            example_text += this_example + '\n\n'
        elif self.n_shots == 3: 
            # sample {K, 7, HS}, {K, 3, 5}, {6, 7, 8}, or {HS, HS, HS}
            possibilities = [('K', '7', 'HS_0'), ('K', '3', '5'), ('6', '7', '8'), ('HS_0', 'HS_1', 'HS_2')]
            choices = self.random_state.choice(range(len(possibilities)))
            choices = possibilities[choices]
            # choose neg or pos exemplar for each grade level in triple
            these_shots = [self.random_state.choice(self.few_shot_exemplars[choice]) for choice in choices]
            for i, choice in enumerate(these_shots): 
                this_example = self._make_exemplar(temp, choice, i+1) 
                example_meta.append([choice['id'], choice['grade']])
                example_text += this_example + '\n\n'
        example_text += self.few_shot_templates[1]
        return example_text, example_meta
            
    def make_positive_examples(self, overwrite=True):
        if len(self.positive_examples) > 0 and not overwrite:
            print('Overwite set to False, but positive examples are not empty, appending new examples...')
        
        if overwrite:
            self.positive_examples = []
        
        for instance in self.instances:
            new_standards = self.expander.inherit_to_standard_level(instance['standards'])
            pos_standards = [tup[1] for tup in new_standards if tup[0] in ['Alignment', 'Addressing']]
            for i, s in enumerate(pos_standards):
                e = {}
                e['id'] = instance['id']+'_pos_{}'.format(i)
                e['dataset'] = 'verification-standard'
                user_message = {}
                user_message['role'] = 'user'
                if self.n_shots > 0: 
                    few_shots, shot_info = self.make_shots()
                    user_message['few_shots'] = '\n' + few_shots + '\n'
                    user_message['shot_info'] = shot_info
                user_message['prompt_template'] = self.prompt_template
                user_message['standard_description'] = self.standardizer.get_standard_description(s)
                user_message['problem_activity'] = self.clean_problem_text(instance['text'], instance['elements'])
                assist_message = {}
                assist_message['role'] = 'assistant'
                assist_message['response_template'] = self.response_template
                assist_message['aligns'] = 'yes'
                e['messages'] = [user_message, assist_message]

                self.positive_examples.append(e)

    def make_negative_examples(self, overwrite=True):
        if self.neg_strat == 'none': 
            raise NotImplementedError("Please specify an actual negative strategy if you want to make negative examples.")

        if len(self.negative_examples) > 0 and not overwrite:
            print('Overwite set to False, but negative examples are not empty, appending new examples...')
        
        if overwrite:
            self.negative_examples = []

        for instance in self.instances:
            new_standards = self.expander.inherit_to_standard_level(instance['standards'])
            pos_standards = [tup[1] for tup in new_standards if tup[0] in ['Alignment', 'Addressing']]

            if len(pos_standards) < 1:
                continue

            negative_standards, neg_strats = self.expander.get_negative_examples_with_strat(pos_standards, self.neg_strat, n_sample=self.n_sample)
        
            for i, neg_s in enumerate(negative_standards):
                e = {}
                e['id'] = instance['id']+'_{strat}_{i}'.format(i=i, strat=neg_strats[i])
                e['dataset'] = 'verification-standard'
                user_message = {}
                user_message['role'] = 'user'
                if self.n_shots > 0: 
                    few_shots, shot_info = self.make_shots()
                    user_message['few_shots'] = '\n' + few_shots + '\n'
                    user_message['shot_info'] = shot_info
                user_message['prompt_template'] = self.prompt_template
                user_message['standard_description'] = self.standardizer.get_standard_description(neg_s)
                user_message['problem_activity'] = self.clean_problem_text(instance['text'], instance['elements'])
                assist_message = {}
                assist_message['role'] = 'assistant'
                assist_message['response_template'] = self.response_template
                assist_message['aligns'] = 'no'
                e['messages'] = [user_message, assist_message]

                self.negative_examples.append(e)

    def output_dataset(self, output_path: str, input_prefix: str):
        random.seed(0)
        if self.n_shots > 0: 
            output_file = os.path.join(output_path, 'data_verification-{input_prefix}_nsample-{neg_sample}_neg-strat-{neg_strat}_prompt-{prompt}_table-{table_style}_shots-{n_shots}.jsonl'.format(
                input_prefix=input_prefix, prompt=self.prompt_name, table_style=self.table_style, neg_sample=self.n_sample, neg_strat=self.neg_strat, n_shots=self.n_shots))
        else: 
            output_file = os.path.join(output_path, 'data_verification-{input_prefix}_nsample-{neg_sample}_neg-strat-{neg_strat}_prompt-{prompt}_table-{table_style}.jsonl'.format(
                input_prefix=input_prefix, prompt=self.prompt_name, table_style=self.table_style, neg_sample=self.n_sample, neg_strat=self.neg_strat))
        
        examples = self.positive_examples + self.negative_examples
        random.shuffle(examples)

        with open(output_file, mode='wt', encoding='utf-8') as outfile:
            for e in examples:
                outfile.write(json.dumps(e) + "\n")