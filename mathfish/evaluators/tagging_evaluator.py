"""
Evaluation of tagging task format.

Author: Lucy Li
Email:  lucyl@allenai.org
"""
from sklearn.metrics import accuracy_score
from mathfish.utils import *
from collections import defaultdict, Counter
from tqdm import tqdm
from mathfish.evaluators import BaseEvaluator
import re
import json
import string

class TaggingEvaluator(BaseEvaluator):
    def __init__(self, data_file):
        self.correct_responses = {}
        self.response_format = None
        self.shot_info = {} # information about shot choices per example
        self.num_options = {} # example id to number of options
        with open(data_file, 'r') as infile: 
            for line in infile: 
                d = json.loads(line)
                last_message = d['messages'][-1]
                assert last_message['role'] == 'assistant'
                answer = last_message['correct_option_index']
                self.correct_responses[d['id']] = answer
                if not self.response_format:  
                    self.response_format = last_message['response_format']
                else: 
                    assert self.response_format == last_message['response_format']
                user_message = d['messages'][-2]
                assert user_message['role'] == 'user'
                if 'shot_info' in user_message: 
                    self.shot_info[d['id']] = user_message['shot_info']
                self.num_options[d['id']] = len(user_message['options'])

        self.options = list(string.ascii_uppercase)
        more_options = []
        for letter1 in self.options: 
            for letter2 in self.options: 
                more_options.append(letter1 + letter2)
        self.options += more_options

    def clean_output(self, res: str):
        if not res: return ''
        res = res.strip()
        if self.response_format == 'comma_list': 
            if res.startswith('none') or res.startswith('None'): 
                return []
            match = re.match(r'^[A-Z,\s]+\b', res)
            if not match: 
                return res
            options = sorted([i.strip() for i in match.group().split(', ') if i.strip() in self.options])
        elif self.response_format == 'comma_list_last': 
            lines = res.split('\n')
            if not lines[-1].startswith('Answer:'): 
                return res
            answer = lines[-1].replace('Answer:', '').strip()
            if answer.lower() == 'none': 
                return []
            match = re.match(r'^[A-Z,\s]+\b', answer)
            if not match: 
                return res
            options = sorted([i.strip() for i in match.group().split(', ') if i.strip() in self.options])
        elif self.response_format == 'json': 
            try:
                response_json = json.loads(res)
            except ValueError as e:
                return res
            if 'answer' not in response_json: return res
            if response_json['answer'].lower() == 'none': 
                return []
            match = re.match(r'^[A-Z,\s]+\b', response_json['answer'])
            if not match: 
                return res
            options = sorted([i.strip() for i in match.group().split(', ') if i.strip() in self.options])
        else: 
            return res
        return options
    
    def _calculate_problem_level_stats(self, results, ret): 
        '''
        TODO
        '''
        return ret
    
    def _calculate_option_accuracy(self, example_id, label, pred): 
        '''
        If pred = B, E (01001) and true = B (01000), 
        accuracy = 4/5. 
        '''
        num_o = self.num_options[example_id]
        num_correct = 0
        for i in range(num_o): 
            o = self.options[i]
            if o in label and o in pred: 
                num_correct += 1
            elif o not in label and o not in pred: 
                num_correct += 1
        return num_correct / num_o

    def calculate_overall_stats(self, results): 
        correct_count = 0
        weak_correct_count = 0
        label_subset_pred = 0 # pred overpredicts labels
        pred_subset_label = 0 # pred misses some true labels 
        malformed_count = 0
        tp = 0 # no option fits, and model predicted no options
        fp = 0 # at least one option fits, but model predicted no options
        fn = 0 # no option fits, but model predicted an option
        option_acc = []
        for example_id in results: 
            label = results[example_id]['label'] # e.g. ['A']
            pred = results[example_id]['prediction'] # e.g. ['A', 'B']
            if type(pred) != list: 
                malformed_count += 1
            else: 
                option_acc.append(self._calculate_option_accuracy(example_id, label, pred))
            if label == pred: 
                # model got every standard
                correct_count += 1
            if type(pred) == list and set(label) & set(pred):
                # model got at least one correct standard
                weak_correct_count += 1
                if label != pred and set(label).issubset(set(pred)): 
                    label_subset_pred += 1
                if label != pred and set(pred).issubset(set(label)): 
                    pred_subset_label += 1
            if len(label) == 0 and len(pred) == 0: 
                tp += 1
            elif len(label) == 0 and len(pred) != 0: 
                fn += 1
            elif len(label) != 0 and len(pred) == 0: 
                fp += 1
        acc = correct_count / len(results)
        ok_format = (len(results) - malformed_count) / len(results)
        weak_acc = weak_correct_count / len(results)
        if weak_correct_count > 0: 
            weak_pred_subset_label = round(pred_subset_label / weak_correct_count, 3)
            weak_label_subset_pred = round(label_subset_pred / weak_correct_count, 3)
        else: 
            weak_pred_subset_label = 0
            weak_label_subset_pred = 0 
        if (tp + fp) > 0: 
            precision_none = tp / (tp + fp)
        else: 
            precision_none = None
        if (tp + fn) > 0: 
            recall_none = tp / (tp + fn)
        else: 
            recall_none = None
        ret = {'acc': acc, 
                'ok_format': ok_format, 
                'weak_acc': weak_acc,
                'weak_label_subset_pred': weak_label_subset_pred, 
                'weak_pred_subset_label': weak_pred_subset_label,
                'precision_none': precision_none,
                'recall_none': recall_none,
                'option_acc': sum(option_acc) / len(option_acc),
                }
        ret = self._calculate_problem_level_stats(results, ret) 
        return ret
    
    def get_problem_id(self, example_id):
        problem_id = '_'.join(example_id.split('_')[:-2])
        return problem_id
    
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
                trues = s_dict[s_key]['true']
                preds = s_dict[s_key]['pred']
                correct_count = 0
                weak_correct_count = 0
                bad_format = 0
                label_subset_pred = 0
                pred_subset_label = 0
                option_acc = []
                f1s = []
                for i in range(len(trues)): 
                    if type(preds[i]) == list: # not malformed response
                        option_acc.append(self._calculate_option_accuracy(s_dict[s_key]['example_ids'][i], trues[i], preds[i]))
                        if len(set(preds[i])) and len(set(trues[i])): 
                            precision = len(set(trues[i]) & set(preds[i])) / len(set(preds[i]))
                            recall = len(set(trues[i]) & set(preds[i])) / len(set(trues[i]))
                            if precision == 0 and recall == 0: 
                                f1s.append(0)
                            else: 
                                f1s.append(2*precision*recall / (precision + recall))
                        else: 
                            f1s.append(0)
                    if trues[i] == preds[i]: 
                        correct_count += 1
                    if type(preds[i]) == list and set(trues[i]) & set(preds[i]): 
                        weak_correct_count += 1
                        if trues[i] != preds[i] and set(trues[i]).issubset(set(preds[i])): 
                            label_subset_pred += 1
                        if trues[i] != preds[i] and set(preds[i]).issubset(set(trues[i])): 
                            pred_subset_label += 1
                assert len(trues) == len(preds)
                if len(preds) == 0: continue
                acc = correct_count / len(trues)
                weak_acc = weak_correct_count / len(trues)
                if weak_correct_count > 0: 
                    weak_pred_subset_label = round(pred_subset_label / weak_correct_count, 3)
                    weak_label_subset_pred = round(label_subset_pred / weak_correct_count, 3)
                else: 
                    weak_pred_subset_label = 0
                    weak_label_subset_pred = 0 
                subset_dict[subset_name][s_key] = {'acc': acc, 
                                                   'weak_acc': weak_acc,
                                                   'weak_pred_subset_label': weak_pred_subset_label,
                                                   'weak_label_subset_pred': weak_label_subset_pred,
                                                   'total': len(trues),
                                                   'option_acc': sum(option_acc) / len(option_acc),
                                                   'f1': sum(f1s) / len(f1s)
                                                   }
        return subset_dict
    
    def subset_data(self, results, problem_meta, standards_dict): 
        '''
        @inputs: 
        - {example_id: {pred: yes/no, label: yes/no}}
        @output: 
        - {subset_name: {subset1: {metric: score}, subset2: {metric: score}}}

        We build upon the BaseEvaluator's subsets and include an additional
        subset in which we investigate how evaluation performance varies based
        on the level of the tree we are operating at
        '''
        base_subsetted_data = super().subset_data(results, problem_meta)
        level_dict = defaultdict(dict) # {level of tree: {pred: [], true: []}}
        option_dict = defaultdict(dict) # {number of options: {pred: [], true: []}}
        exemplar_dict = defaultdict(dict) # {problem grade - shot grade/s: {pred: [], true: []}}

        for example_id in tqdm(results): 
            r = results[example_id]
            problem_id = self.get_problem_id(example_id)

            level = r['dataset'].split('-')[1]
            level_dict = self._update_dict(level_dict, level, r['prediction'], r['label'], example_id)

            num_o = self.num_options[example_id]
            option_dict = self._update_dict(option_dict, num_o, r['prediction'], r['label'], example_id)

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
                
        subsetted_data = base_subsetted_data + [(level_dict, 'level'), (exemplar_dict, 'few_shot'), (option_dict, 'num_options')]
        return subsetted_data

    def get_true_label(self, example_id: str): 
        answers = []
        for idx in self.correct_responses[example_id]: 
            answers.append(self.options[idx])
        answers = sorted(answers)
        return answers
    
    def get_option_letter_to_idx(self, letter): 
        '''
        A -> 0, B -> 1, C -> 2, etc
        '''
        return self.options.index(letter)
    
    def get_correct_incorrect(self, results): 
        '''
        Returns incorrect/correct options and examples split by them. 
        Used for eval.py interactive mode
        '''
        raise NotImplementedError("OOPS")