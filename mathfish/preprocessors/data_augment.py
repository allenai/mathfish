"""
Given a dataset or problem/activities labeled with standards,
augment it with standards-level negative examples and additional
standards-level positive examples.

Author: Lucy Li (@lucy3)
Email:  lucyl@allenai.org
"""
import json
from collections import defaultdict
from typing import List
from mathfish.utils import *
import numpy as np

class DataExpander:
    def __init__(self, standards_path: str, random_seed: int = 0):
        self.random_state = np.random.RandomState(random_seed)
        self.children = defaultdict(set)
        self.parent = {}
        self.math_standards_levels = {}
        self.standard_grades = {}
        self.standard_domains_cats = {} # k8 domains or HS cats
        self.connections = defaultdict(dict) # { origin standard : {relation : [destination standards] } 
        with open(standards_path, 'r') as infile:
            for line in infile:
                d = json.loads(line)
                self.math_standards_levels[d['id']] = d['level']
                if d["children"]:
                    self.children[d['id']] = set(d["children"])
                if d['parent']:
                    self.parent[d['id']] = d['parent']

                self.standard_domains_cats[d['id']] = get_domain_cat(d['id'])
                self.standard_grades[d['id']] = get_grade(d['id'])
                self.connections[d['id']] = d['connections']

    def inherit_to_standard_level(self, standards: List[List[str]], keep_other_levels=False):
        '''
        If a cluster or domain is listed for a problem/activity, then assume the
        problem/activity aligns with all standards (children) within that cluster or domain.

        Similarly, if a sub-standard is listed for a problem/activity, assume it aligns with
        the parent standard.

        Examples:
        - F-IF.C.7e -> F-IF.C.7
        - S-IC.B -> S-IC.B.3, S-IC.B.4, S-IC.B.5, S-IC.B.6

        @params:
        - standards: a list of [relation, standard]
        - keep_other_levels: whether to keep clusters, domains, and sub-standards
        that are part of the traversed tree and original set of standard labels. If False,
        we return *only* standard-level standards.
        '''
        new_standards = []
        for tup in standards:
            rel = tup[0]
            s = tup[1]
            if self.math_standards_levels[s] == 'Sub-standard':
                new_standards.append((rel, self.parent[s]))

            queue = []
            queue.append(s)

            while queue:
                standard = queue.pop(0)
                new_standards.append((rel, standard))
                if standard in self.children:
                    for child in self.children[standard]:
                        queue.append(child)

        new_standards = [list(tup) for tup in set(new_standards)]

        resulting_standards = []
        if not keep_other_levels:
            resulting_standards = []
            for tup in new_standards:
                if self.math_standards_levels[tup[1]] == 'Standard':
                    resulting_standards.append(tup)
        else:
            resulting_standards = new_standards
        return resulting_standards
    
    def get_negative_examples_with_strat(self, positive_standards: List[str], neg_strat: str, n_sample=1): 
        '''

        @returns
        - a list of tuples e.g. [(strat, negative standard)]
        '''
        assert neg_strat in ['same-domain-same-grade', 'same-domain-different-grade', 
                 'different-domain-different-grade', 'different-domain-same-grade', 
                 'neighbors', 'all-negative-types']
        if 'domain' in neg_strat and 'grade' in neg_strat:
            ret = self.get_negative_examples_by_grade_and_domain(positive_standards=positive_standards, n_sample=n_sample, neg_strat=neg_strat) 
            strats = [neg_strat]*len(ret)
            return ret, strats
        elif neg_strat == 'neighbors': 
            ret = self.get_negative_examples_by_connections(positive_standards=positive_standards, n_sample=n_sample)
            strats = [neg_strat]*len(ret)
            return ret, strats
        else:
            assert neg_strat == 'all-negative-types'
            # all-negative-types
            all_ret = []
            all_strats = []

            for grade in ['different', 'same']: 
                for domain in ['different', 'same']:
                    this_neg_strat = domain + '-domain-' + grade + '-grade' 
                    ret = self.get_negative_examples_by_grade_and_domain(positive_standards=positive_standards, 
                                                                  n_sample=n_sample, neg_strat=this_neg_strat) 
                    all_ret.extend(ret)
                    all_strats.extend([this_neg_strat]*len(ret))

            this_neg_strat = 'neighbors'
            ret = self.get_negative_examples_by_connections(positive_standards=positive_standards, n_sample=n_sample)
            all_ret.extend(ret)
            all_strats.extend([this_neg_strat]*len(ret))

            assert len(all_ret) <= n_sample*5
            return all_ret, all_strats

    def get_negative_examples_by_grade_and_domain(self, positive_standards: List[str], n_sample=1, neg_strat='different-domain-different-grade'):
        '''
        Assumption: problems do not align with other standards from other grade levels
        if those other grade levels are not listed.

        Example: problem/activity aligns with 8.NS.A -> problem/activity
        does not align with 7.NS.A if 7.NS.A is not listed.

        Assumption: problems do not align with other standards within the same
        grade level as standards that are listed.

        Example: problem/activity aligns with 8.NS.A -> problem/activity
        does not align with 8.EE.A if 8.EE.A is not listed

        @params:
        - n_sample: number of samples to get from possible negative pool

        - neg_strat containing info about
        grade - one of ['same', 'different']. Whether to sample from
        the same grade level/s as positive example/s or different ones. 
        domain - one of ['same', 'different']. Whether to sample from
        the same domain/s as positive example/s or different ones.

        - positive_standards: a list of standardized standards
        '''
        assert '-' in neg_strat
        parts = neg_strat.split('-')
        grade = parts[2]
        domain = parts[0]
        assert grade in set(['same', 'different'])
        assert domain in set(['same', 'different'])

        HScat_to_k8domains = {
            'N' : 'NS',
            'NS' : 'N',
            'A' : 'OA',
            'OA' : 'A',
            'S': 'SP',
            'SP' : 'S'
        }

        current_grades = set()
        current_domains = set()
        for standard in positive_standards:
            current_grades.add(self.standard_grades[standard])

            domain_cat = self.standard_domains_cats[standard]
            current_domains.add(domain_cat)
            if domain_cat in HScat_to_k8domains:
                current_domains.add(HScat_to_k8domains[domain_cat])

        possible_pool = set()

        for standard in self.math_standards_levels:
            if self.math_standards_levels[standard] != 'Standard': continue
            if standard in positive_standards: continue
            if grade == 'same' and self.standard_grades[standard] not in current_grades:
                # filter to only standards in same grade
                continue
            elif grade == 'different' and self.standard_grades[standard] in current_grades:
                # filter to only standards in different grade
                continue
            if domain == 'same' and self.standard_domains_cats[standard] not in current_domains:
                # filter to only standards in same domain
                continue
            elif domain == 'different' and self.standard_domains_cats[standard] in current_domains:
                # filter to only standards in different domain
                continue
            possible_pool.add(standard)

        if len(possible_pool) <= n_sample:
            sample = possible_pool
        else:
            sample = self.random_state.choice(sorted(possible_pool), size=n_sample, replace=False)
        return list(sample)

    def get_negative_examples_by_connections(self, positive_standards: List[str], relation_type='all', n_sample: int = -1): 
        '''
        Assumptions: problems do not align with other standards if those other standards
        are not listed. This makes negative labels *related* to positive ones particularly tricky. 
        To identify "related" negative labels, we leverage nearest neighbors in 
        Achieve the Core's graph of standards' relationships. 

        Arrows "X->Y" in the graph represent progression, or where students likely can't meet
        standard Y if they don't meet standard X first. 

        Undirected relationships "X--Y" represent content that is related without meaningful
        precendence. 

        @params: 
        - positive_standards: a list of standardized standards
        - type of relations to get nearest neighbors
        '''
        assert relation_type in set(['all', 'progress to', 'progress from', 'related'])

        ret = set()
        if relation_type == 'all' or relation_type == 'progress to': 
            for standard in positive_standards: 
                ret.update(self.connections[standard]['progress to'])
        if relation_type == 'all' or relation_type == 'progress from': 
            for standard in positive_standards: 
                ret.update(self.connections[standard]['progress from'])
        if relation_type == 'all' or relation_type == 'related': 
            for standard in positive_standards: 
                ret.update(self.connections[standard]['related'])

        ret = ret - set(positive_standards) # account for cases where positive standards may be connected to each other

        if len(ret) <= n_sample: 
            sample = ret
        elif n_sample > 0: 
            sample = self.random_state.choice(sorted(ret), size=n_sample, replace=False)
        else: 
            sample = ret

        return list(sample)
        
