'''
Parent class for various classes
that evaluate different task formats

Author: Lucy Li
Email:  lucyl@allenai.org
'''
from sklearn.metrics import accuracy_score
from mathfish.utils import *
from collections import defaultdict, Counter
from tqdm import tqdm

class BaseEvaluator:
    def clean_output(self, res: str):
        '''
        Dummy function
        '''
        return res
    
    def get_true_label(self, example_id: str): 
        '''
        Dummy function
        '''
        return example_id
    
    def calculate_overall_stats(self, results): 
        '''
        Returns only accuracy.
        '''
        trues = []
        preds = []
        for example_id in results: 
            trues.append(results[example_id]['label'])
            preds.append(results[example_id]['prediction'])
        acc = accuracy_score(trues, preds)
        return acc
    
    def calculate_subsetted_stats(self, subset_list): 
        '''
        @inputs: 
        - subset_list = [({subset : {pred: [], true: []}}, subset_name)]
        @output: 
        - {subset_name: {subset1: {metric: score}, subset2: {metric: score}}}, 
        or performance for each subset.

        For example, performance metrics could subsetted by grade level, domain, or 
        some other characteristic of the data. 
        Returns only accuracy.
        '''
        subset_dict = defaultdict(dict) 
        for tup in subset_list: 
            s_dict = tup[0]
            subset_name = tup[1]
            print("Calculating scores for...", subset_name)
            for s_key in tqdm(s_dict): 
                trues = s_dict[s_key]['true']
                preds = s_dict[s_key]['pred']
                assert len(trues) == len(preds)
                if len(preds) == 0: continue
                acc = accuracy_score(trues, preds)
                subset_dict[subset_name][s_key] = {'acc': acc}
        return subset_dict
    
    def _update_dict(self, d, key, pred, true, example_id): 
        if key not in d: 
            d[key] = {'pred': [], 'true': [], 'example_ids': []}
        d[key]['pred'].append(pred)
        d[key]['true'].append(true)
        d[key]['example_ids'].append(example_id)
        return d
    
    def get_problem_id(self, example_id): 
        return example_id
    
    def subset_data(self, results, problem_meta, *args, **kwargs): 
        '''
        @inputs: 
        - {example_id: {pred: yes/no, label: yes/no}}
        @output: 
        - {subset_name: {subset1: {metric: score}, subset2: {metric: score}}}

        The following subsets of our dataset include: 
        - problem_activity_type (e.g. task, lesson)
        - grade_level (e.g. K-8, HS)
        - image (boolean, problem/activity contains image)
        - table (boolean, problem/activity contains table)
        - domain_cat (e.g. Operations & Algebraic Thinking, Geometry)
        '''
        p_a_type_dict = defaultdict(dict) # {problem_activity_type : {pred: [], true: []}}, one per example
        grade_level_dict = defaultdict(dict) # {grade_level: {pred: [], true: []}}, one or more per example
        image_dict = defaultdict(dict) # {has image: {pred: [], true: []}, no image: {pred: [], true: []}}, one per example
        table_dict = defaultdict(dict) # {has table: {pred: [], true: []}, no table: {pred: [], true: []}}, one per example
        domain_cat_dict = defaultdict(dict) # {domain_cat: {pred: [], true: []}}, one or more per example

        for example_id in tqdm(results):
            problem_id =self.get_problem_id(example_id)
            r = results[example_id]
            assert problem_id in problem_meta

            problem_activity_type = problem_meta[problem_id]['source'] + ' ' + problem_meta[problem_id]['problem_activity_type']
            p_a_type_dict = self._update_dict(p_a_type_dict, problem_activity_type, r['prediction'], r['label'], example_id)

            has_image = False
            has_table = False
            for el in problem_meta[problem_id]['elements']: 
                if 'TABLE' in el: 
                    has_table = True
                if 'IMAGE' in el: 
                    has_image = True
            image_dict = self._update_dict(image_dict, has_image, r['prediction'], r['label'], example_id)
            table_dict = self._update_dict(table_dict, has_table, r['prediction'], r['label'], example_id)

            domain_cats = set()
            grade_levels = set()
            for tup in problem_meta[problem_id]['standards']: 
                if tup[0] not in ['Alignment', 'Addressing']: continue
                domain_cats.add(get_domain_cat(tup[1]))
                grade_levels.add(get_grade(tup[1]))
            for dc in domain_cats: 
                domain_cat_dict = self._update_dict(domain_cat_dict, dc, r['prediction'], r['label'], example_id)
            for gr in grade_levels: 
                grade_level_dict = self._update_dict(grade_level_dict, gr, r['prediction'], r['label'], example_id)

        return [(p_a_type_dict, 'problem_activity_type'), 
                (grade_level_dict, 'grade_level'), 
                (domain_cat_dict, 'domain_cat'), 
                (image_dict, 'has_image'), 
                (table_dict, 'has_table')]
    
    def get_correct_incorrect(self, results): 
        '''
        Returns incorrect/correct options and examples split by them. 
        Used for eval.py interactive mode
        '''
        option_pools = defaultdict(set)
        for example_id in results: 
            if results[example_id]['label'] == results[example_id]['prediction']: 
                option_pools['correct'].add(example_id)
            else: 
                option_pools['incorrect'].add(example_id)
        return option_pools
        