"""
Retrieves relevant parts of the standards hierarchy, 
used to support the tagging task. 

Author: Lucy Li (@lucy3)
Email:  lucyl@allenai.org
"""
import json
from typing import List
import numpy as np
from collections import defaultdict
from mathfish.utils import *
import string

class TreeRetriever: 
    '''
    Providing multiple choice options for tagging task format. 

    Decision tree: 
    get_list_of_domains() -> get_possible_clusters() -> get_possible_standards() 
    Note that each -> can involve multiple branches (a.k.a. options are checkboxes, not radio buttons)

    No decision tree: 
    get_random_standards() 
    '''

    def __init__(self, standards_path: str, domain_groups_path: str, random_seed=0):
        with open(domain_groups_path, 'r') as infile: 
            self.domain_groups = json.load(infile)

        self.domain_cat_to_domain_group = {} # e.g. {'CC': 'Counting & Cardinality'}
        for dg in self.domain_groups: 
            for dc in self.domain_groups[dg]['domain_cats']: 
                self.domain_cat_to_domain_group[dc] = dg

        self.random_state = np.random.RandomState(random_seed)
        # note: some cluster descriptions map onto multiple clusters
        # {domain_cat: {cluster description: [cluster labels]}}
        self.domain_clusters = defaultdict(dict)
        # {cluster label: [standards]}
        self.cluster_standards = defaultdict(list)
        # {standard: description}
        self.standards_descriptions = defaultdict(str)
        # {description: standard}
        self.standards_descriptions_rev = defaultdict(str)
        self.modeling_standards = set()
        with open(standards_path, 'r') as infile:
            for line in infile:
                d = json.loads(line)
                if d['level'] == 'Cluster': 
                    text = d['description'].strip()
                    if '-' in d['id']: 
                        domain = d['id'].split('-')[0]
                    else: 
                        domain = d['id'].split('.')[1]
                    if text not in self.domain_clusters[domain]: 
                        self.domain_clusters[domain][text] = [] 
                    self.domain_clusters[domain][text].append(d['id'])

                if d['level'] == 'Standard': 
                    cluster = d['parent']
                    self.cluster_standards[cluster].append(d['id'])
                    self.standards_descriptions[d['id']] = d['description']
                    # all standard descriptions must be unique
                    assert d['description'] not in self.standards_descriptions_rev
                    self.standards_descriptions_rev[d['description']] = d['id']
                    if d['modeling']: 
                        self.modeling_standards.add(d['id'])

        self.cluster_descriptions = {}
        for domain in self.domain_clusters: 
            for descript in self.domain_clusters[domain]:
                # all clusters w/ a description are within the same domain 
                assert descript not in self.cluster_descriptions 
                self.cluster_descriptions[descript] = self.domain_clusters[domain][descript]

    def get_modeling_standards(self): 
        '''
        This is a special domain that isn't indicated in the name of a 
        standard (e.g. there is no standard that starts with "M.") but instead
        spans across other domains
        '''
        return self.modeling_standards

    def get_list_of_domains(self, give_description=False, shuffle_options=True): 
        '''
        @params: 
        - give_description: whether to include natural language description of
        domain or not
        - shuffle_options: whether to shuffle the options shown or not

        @returns: 
        - option_list: list of options to show in prompt
        - next_branches: next branches in tree, in the same order as entries in option_list

        Suggested use: 
            Use to create the top layer of the decision tree, and experiment
            with including and not including natural language descriptions for
            each domain group. 
        '''
        option_list = []
        for domain_group in self.domain_groups:
            domain_group_str = domain_group.strip()
            if give_description: 
                # e.g. ['Counting and Cardinality: blah blah', 'Geometry: blah blah']
                domain_group_str += ': ' + self.domain_groups[domain_group]['description']
            option_list.append(domain_group_str)
        if shuffle_options: 
            self.random_state.shuffle(option_list)
        return option_list

    def get_list_of_clusters(self, shuffle_options=True): 
        raise NotImplementedError("Currently can only get_possible_clusters")
    
    def get_list_of_standards(self, shuffle_options=True): 
        raise NotImplementedError("Currently can only get_possible_standards or see random_tagger_dataset.py for random standards")

    def get_possible_clusters(self, domain_group: str, shuffle_options=True): 
        '''
        @params: 
        - domain_group, or a string representing a domain (e.g. "Counting & Cardinality") 
        or group of k-8 domains/HS categories (e.g. "Operations, Algebra, & Algebraic Thinking")
        - shuffle_options: whether to shuffle the options shown or not

        @returns: 
        - option_list: list of cluster descriptions to show in prompt
        - next_branches: next branches in tree, in the same order as entries in option_list

        Suggested use: 
            For each domain_group ("branch") selected by the model in the previous level of the
            tree, ask it to traverse each branch to select clusters, with options
            provided by this function. 
        '''
        assert domain_group in self.domain_groups

        domain_cats = self.domain_groups[domain_group]['domain_cats']
        option_list = [] # e.g. [description1, description2]
        for dc in domain_cats: 
            cluster_descripts = self.domain_clusters[dc] 
            for descript in cluster_descripts: 
                option_list.append(descript)
        if shuffle_options: 
            self.random_state.shuffle(option_list)
        return option_list

    def get_possible_standards(self, cluster: str, shuffle_options=True): 
        '''
        @params: 
        - cluster label
        - shuffle_options: whether to shuffle the options shown or not

        @returns: 
        - option_list: list of standard descriptions to show in prompt
        - next_branches: standard labels in the same order as entries in option_list

        Suggested use: 
            For each cluster ("branch") selected by the model in the previous level of the
            tree, ask it to traverse each branch to select standards, with options
            provided by this function. 
        '''
        assert cluster in self.cluster_standards
        standard_labels = self.cluster_standards[cluster]
        if shuffle_options: 
            self.random_state.shuffle(standard_labels)
        # write out options
        option_list = []
        for i, s in enumerate(standard_labels): 
            option_list.append(self.standards_descriptions[s])
        return option_list

    def get_random_standards(self, positive_labels: List[str], num_options: int, shuffle_options=True): 
        '''
        @params: 
        - positive_labels: standard labels that are correct 
        - num_options: number of options to present to model to choose from, which
        includes the number of positive labels
        - shuffle_options: whether to shuffle the options shown or not 

        @returns: 
        - option_list: list of standard descriptions to show in prompt
        - next_branches: standard labels in the same order as entries in option_list
        
        Suggested use: 
            Have an option where instead of traversing the entire tree, we just
            present completely random standards and ask the model to choose from them, 
            with some correct answers always present as options
        '''
        # sample for options if needed
        for label in positive_labels: 
            assert label in self.standards_descriptions
        if num_options == len(positive_labels): 
            # we have exactly as many options that are allowed
            ret_list = positive_labels
        elif num_options < len(positive_labels): 
            # we have too many options, so only show the max allowed
            ret_list = self.random_state.choice(positive_labels, size=num_options, replace=False)
        else:
            # we don't have enough options, so we add some incorrect options
            num_to_sample = num_options - len(positive_labels)
            all_standards = set(self.standards_descriptions.keys())
            standards_pool = all_standards - set(positive_labels)
            if num_to_sample > len(standards_pool): 
                sample = list(standards_pool)
            else: 
                sample = list(self.random_state.choice(sorted(standards_pool), size=num_to_sample))
            ret_list = positive_labels + sample

        if shuffle_options: 
            self.random_state.shuffle(ret_list) 
        # write out options
        option_list = []
        for i, s in enumerate(ret_list):
            option_list.append(self.standards_descriptions[s])
        return option_list
    
    def get_pointer_to_next_branch(self, option_str, level): 
        assert level in set(['domain', 'cluster', 'standard'])
        if level == 'domain': 
            return option_str.split(': ')[0] # "Counting and Cardinality: blah blah" -> "Counting and Cardinality"
        elif level == 'cluster': 
            return self.cluster_descriptions[option_str.strip()] # "Extend The Counting Sequence." -> "1.NBT.A" 
        elif level == 'standard': 
            return self.standards_descriptions_rev[option_str.strip()] # "blah blah" -> "1.NBT.A.1"