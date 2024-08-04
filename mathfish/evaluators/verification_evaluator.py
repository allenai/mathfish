'''
Evaluation of a yes/no verification task format, where
problems/activities are paired with positive and
negative standard labels' descriptions in prompts.

Author: Lucy Li
Email:  lucyl@allenai.org
'''
from sklearn.metrics import f1_score, accuracy_score
from mathfish.utils import *
from collections import defaultdict, Counter
from tqdm import tqdm
from mathfish.evaluators import BaseEvaluator
import re
import json

class VerificationEvaluator(BaseEvaluator):
    def __init__(self, data_file): 
        self.correct_responses = {}
        self.standard_descriptions = {}
        self.shot_info = {} # information about shot choices per example
        with open(data_file, 'r') as infile: 
            for line in infile: 
                d = json.loads(line)
                last_message = d['messages'][-1]
                assert last_message['role'] == 'assistant'
                answer = last_message['aligns']
                self.correct_responses[d['id']] = answer
                first_message = d['messages'][0]
                assert first_message['role'] == 'user'
                sd = first_message['standard_description']
                self.standard_descriptions[d['id']] = sd
                if 'shot_info' in first_message: 
                    self.shot_info[d['id']] = first_message['shot_info'] 

    def clean_output(self, res: str):
        '''
        Takes in output text from a model and
        extracts "yes" or "no". 
        '''
        if not res: # output is null due to failed retries
            return ''
        split_res = re.split('. |\s |\n', res) 
        for r in split_res:
            cleaned_r = re.sub('\.|,|\s', '', r.lower())
            if cleaned_r in ['yes', 'no']:
                return cleaned_r
        return res
    
    def get_true_label(self, example_id: str): 
        '''
        Get the true label
        '''
        return self.correct_responses[example_id]
    
    def calculate_overall_stats(self, results): 
        '''
        @output: 
        - accuracy and micro F1 over all dataset examples
        '''
        trues = defaultdict(list)
        preds = defaultdict(list)
        malformed_count = 0
        for example_id in results:
            trues['all'].append(results[example_id]['label'])
            preds['all'].append(results[example_id]['prediction'])
            if results[example_id]['prediction'] != 'yes' and results[example_id]['prediction'] != 'no': 
                malformed_count = 1
            example_type = example_id.split('_')[-2]
            trues[example_type].append(results[example_id]['label'])
            preds[example_type].append(results[example_id]['prediction'])
        ret = {}
        for example_type in trues: 
            acc = accuracy_score(trues[example_type], preds[example_type])
            if example_type == 'all': 
                ret['acc'] = round(acc, 3)
            else: 
                ret['acc_' + example_type] = round(acc, 3)
        ret['ok_format'] = 1 - malformed_count / len(trues)
        ret['f1'] = round(f1_score(trues['all'], preds['all'], labels=['yes', 'no'], average='micro'), 3)
        return ret
    
    def get_problem_id(self, example_id): 
        return '_'.join(example_id.split('_')[:-2])
    
    def calculate_subsetted_stats(self, subset_list): 
        '''
        @inputs: 
        - subset_list = [({subset : {pred: [], true: []}}, subset_name)]
        @output: 
        - {subset_name: {subset1: {metric: score}, subset2: {metric: score}}}, 
        or accuracy & micro f1 for each subset.

        For example, performance metrics could subsetted by grade level, domain, or 
        some other characteristic of the data. 
        '''
        subset_dict = defaultdict(dict) 
        for tup in subset_list: 
            s_dict = tup[0]
            subset_name = tup[1]
            print("Calculating scores for...", subset_name)
            for s_key in tqdm(s_dict): 
                ret = {}
                trues = s_dict[s_key]['true']
                preds = s_dict[s_key]['pred']
                example_ids = s_dict[s_key]['example_ids']
                assert len(trues) == len(preds)
                assert len(trues) == len(example_ids)
                if len(preds) == 0: continue

                # get score by example type (e.g. pos / neg)
                trues_by_example = defaultdict(list)
                preds_by_example = defaultdict(list)
                for i, e_id in enumerate(example_ids): 
                    example_type = e_id.split('_')[-2]
                    trues_by_example[example_type].append(trues[i])
                    preds_by_example[example_type].append(preds[i]) 
                for example_type in trues_by_example: 
                    acc = accuracy_score(trues_by_example[example_type], preds_by_example[example_type])
                    ret['acc_' + example_type] = round(acc, 3)

                # get some overall scores
                ret['acc'] = accuracy_score(trues, preds)
                ret['f1'] = f1_score(trues, preds, labels=['yes', 'no'], average='micro')  
                subset_dict[subset_name][s_key] = ret
        return subset_dict
    
    def get_negative_standard(self, example_id, standards_dict): 
        '''
        Given a prompt inputted into a model,
        map back to the negative standard used to generate that prompt.
        '''
        standard_desc = self.standard_descriptions[example_id]
        neg_standard = standards_dict[standard_desc]
        return neg_standard
    
    def subset_data(self, results, problem_meta, standards_dict): 
        '''
        @inputs: 
        - {example_id: {pred: yes/no, label: yes/no}}
        @output: 
        - {subset_name: {subset1: {metric: score}, subset2: {metric: score}}}

        We build upon the BaseEvaluator's subsets and include an additional
        subset in which we investigate how evaluation performance varies based
        on the grade level distance between negative and positive labels. 
        '''
        base_subsetted_data = super().subset_data(results, problem_meta)
        grade_level_dist_dict = defaultdict(dict) # {diff between neg and pos: {pred: [], true: []}}
        exemplar_dict = defaultdict(dict) # {problem grade - shot grade/s: {pred: [], true: []}}

        for example_id in tqdm(results): 
            problem_id = '_'.join(example_id.split('_')[:-2]) 
            r = results[example_id]

            pos_standards = []
            for tup in problem_meta[problem_id]['standards']: 
                if tup[0] not in ['Alignment', 'Addressing']: continue
                pos_standards.append(tup[1])

            if '_pos_' not in example_id: 
                # calculate this only for negative examples
                neg_standard = self.get_negative_standard(example_id, standards_dict)
                _, neg_grade, pos_grade = get_grade_level_distance(pos_standards, neg_standard)
                key = str(neg_grade) + '_' + str(pos_grade)
                grade_level_dist_dict = self._update_dict(grade_level_dist_dict, key, r['prediction'], r['label'], example_id)

            if self.shot_info: 
                pos_standards = []
                for tup in problem_meta[problem_id]['standards']: 
                    if tup[0] not in ['Alignment', 'Addressing']: continue
                    pos_standards.append(tup[1])

                problem_grade = map_number_to_grade(get_max_grade(pos_standards))
                this_shots = self.shot_info[example_id] 
                shot_grades = '_'.join([str(s[1]) for s in this_shots])
                key = problem_grade + '-' + shot_grades
                exemplar_dict = self._update_dict(exemplar_dict, key, r['prediction'], r['label'], example_id)
                
        subsetted_data = base_subsetted_data + [(grade_level_dist_dict, 'grade_level_distance'), (exemplar_dict, 'few_shot')]
        return subsetted_data

    def get_correct_incorrect(self, results): 
        '''
        Returns incorrect/correct options and examples split by them. 
        Used for eval.py interactive mode
        '''
        option_pools = defaultdict(set)
        for example_id in results: 
            if results[example_id]['prediction'] == 'yes' and results[example_id]['prediction'] == results[example_id]['label']: 
                option_pools['TP'].add(example_id)
            elif results[example_id]['prediction'] == 'no' and results[example_id]['prediction'] != results[example_id]['label']: 
                option_pools['FN'].add(example_id)
            elif results[example_id]['prediction'] == 'yes' and results[example_id]['prediction'] != results[example_id]['label']: 
                option_pools['FP'].add(example_id)
            elif results[example_id]['prediction'] == 'no' and results[example_id]['prediction'] == results[example_id]['label']: 
                option_pools['TN'].add(example_id)
            else: 
                option_pools['malformed'].add(example_id) 
        return option_pools 