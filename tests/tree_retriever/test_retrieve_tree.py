"""
Tests retrieve_tree.py with toy examples
that match what actual data looks like
in structure but has only 2-3 options
at each level of the tree. 

Author: Lucy Li (@lucy3)
Email:  lucyl@allenai.org
"""

import pathlib
import unittest
from unittest import TestCase

from mathfish.tree_retriever import TreeRetriever

class TestverificationEvaluate(TestCase):
    def setUp(self):
        standards_path = pathlib.Path(__file__).parent.parent / "fixtures" / "toy_tree.jsonl"
        domain_groups_path = pathlib.Path(__file__).parent.parent / "fixtures" / "toy_domain_groups.jsonl"
        self.option_retriever = TreeRetriever(standards_path, domain_groups_path)

    def check_options_match(self, option_list, next_branches, layer): 
        # check that option_list and next_branches match in ordering
        remake_ret_string = []
        for item in next_branches: 
            if layer == 'domain': 
                remake_ret_string.append(item)
            elif layer == 'cluster': 
                remake_ret_string.append('descript of ' + item[0].lower())
            else: 
                remake_ret_string.append('descript of ' + item.lower())
        self.assertEqual(sorted(option_list), sorted(remake_ret_string))

    def test_get_list_of_domains(self):
        # -- not shuffled -- 
        option_list = self.option_retriever.get_list_of_domains(shuffle_options=False)
        correct_string_items = ['Counting & Cardinality', 'Operations, Algebra, & Algebraic Thinking', 'Geometry']
        self.assertEqual(option_list, correct_string_items)
        next_branches = [self.option_retriever.get_pointer_to_next_branch(o, 'domain') for o in option_list]
        self.assertEqual(next_branches, correct_string_items) 

        # -- shuffled --
        option_list = self.option_retriever.get_list_of_domains()
        # all options should be present regardless of order
        correct_string_items = ['Counting & Cardinality', 
                                'Geometry', 
                                'Operations, Algebra, & Algebraic Thinking'] 
        self.assertEqual(sorted(option_list), correct_string_items)
        next_branches = [self.option_retriever.get_pointer_to_next_branch(o, 'domain') for o in option_list]
        self.assertEqual(sorted(next_branches), correct_string_items)
        self.check_options_match(option_list, next_branches, 'domain')

        # -- shuffled options + description --
        option_list = self.option_retriever.get_list_of_domains(give_description=True) 
        # all options should be present regardless of order
        correct_string_items = ['Counting & Cardinality: descript1', 
                                'Geometry: descript3', 
                                'Operations, Algebra, & Algebraic Thinking: descript2'] 
        self.assertEqual(sorted(option_list), correct_string_items)
        correct_no_descript_string_items = ['Counting & Cardinality', 
                                            'Geometry', 
                                            'Operations, Algebra, & Algebraic Thinking'] 
        next_branches = [self.option_retriever.get_pointer_to_next_branch(o, 'domain') for o in option_list]
        self.assertEqual(sorted(next_branches), correct_no_descript_string_items) 

    def test_get_possible_clusters(self):
        domain_group = 'Counting & Cardinality' # -> children are K.CC.A, K.CC.B
        # -- not shuffled --
        option_list = self.option_retriever.get_possible_clusters(domain_group, shuffle_options=False) 
        correct_string_items = ['descript of k.cc.a', 'descript of k.cc.b']
        self.assertEqual(option_list, correct_string_items)
        correct_option_values = [['K.CC.A'], ['K.CC.B']] 
        next_branches = [self.option_retriever.get_pointer_to_next_branch(o, 'cluster') for o in option_list]
        self.assertEqual(next_branches, correct_option_values)

        # -- shuffled --
        option_list = self.option_retriever.get_possible_clusters(domain_group)
        self.assertEqual(sorted(option_list), correct_string_items)
        next_branches = [self.option_retriever.get_pointer_to_next_branch(o, 'cluster') for o in option_list]
        self.assertEqual(sorted(next_branches), correct_option_values) 
        self.check_options_match(option_list, next_branches, 'cluster')

    def test_get_possible_standards(self):
        cluster = 'K.CC.A' # -> children are K.CC.A.1, K.CC.A.2, K.CC.A.3
        # -- not shuffled --
        option_list = self.option_retriever.get_possible_standards(cluster, shuffle_options=False) 
        correct_string_items = ['descript of k.cc.a.1', 
                                'descript of k.cc.a.2', 
                                'descript of k.cc.a.3']
        self.assertEqual(option_list, correct_string_items)
        correct_option_values = ['K.CC.A.1', 'K.CC.A.2', 'K.CC.A.3'] 
        next_branches = [self.option_retriever.get_pointer_to_next_branch(o, 'standard') for o in option_list]
        self.assertEqual(next_branches, correct_option_values)

        # -- shuffled --
        option_list = self.option_retriever.get_possible_standards(cluster)
        self.assertEqual(sorted(option_list), correct_string_items)
        next_branches = [self.option_retriever.get_pointer_to_next_branch(o, 'standard') for o in option_list]
        self.assertEqual(sorted(next_branches), correct_option_values) 
        self.check_options_match(option_list, next_branches, 'standard')

    def test_tricky_one_to_many_paths(self):
        '''
        Our tree is built to be able to handle the following: 
        - cross-grade domains, e.g. "Geometry" maps to clusters that span multiple grade levels
        - cluster descriptions can also repeat across grades, e.g. "Reason with shapes and their attributes"
        is in grades 1, 2, and 3 in CCSS. 

        This function tests these scenarios.
        ''' 
        domain_group = 'Geometry'
        option_list = self.option_retriever.get_possible_clusters(domain_group, shuffle_options=False) 
        correct_string_items = ['circles', 'squares']
        self.assertEqual(option_list, correct_string_items) 
        correct_option_values = [['1.G.A', '2.G.A'], ['1.G.B']] # two grades talk about circles
        next_branches = [self.option_retriever.get_pointer_to_next_branch(o, 'cluster') for o in option_list]
        self.assertEqual(next_branches, correct_option_values) 

    def test_get_random_standards(self):  
        positive_labels = ['K.CC.A.1', 'K.CC.B.2']
        # test the case where we ask for exactly as many options as positive_labels already has
        option_list = self.option_retriever.get_random_standards(positive_labels, len(positive_labels))
        correct_string_items = ['descript of k.cc.a.1', 'descript of k.cc.b.2']
        self.assertEqual(sorted(option_list), correct_string_items)
        next_branches = [self.option_retriever.get_pointer_to_next_branch(o, 'standard') for o in option_list]
        self.check_options_match(option_list, next_branches, 'standard')

        # test the case where we ask for fewer options than len(positive_labels)
        option_list = self.option_retriever.get_random_standards(positive_labels, len(positive_labels) - 1)
        correct_string_items = ['descript of k.cc.a.1', 'descript of k.cc.b.2']
        difference = set(correct_string_items) - set(option_list)
        self.assertEqual(len(difference), 1)
        next_branches = [self.option_retriever.get_pointer_to_next_branch(o, 'standard') for o in option_list]
        self.check_options_match(option_list, next_branches, 'standard')

        # test the case where we ask for more options than len(positive_labels)
        option_list = self.option_retriever.get_random_standards(positive_labels, len(positive_labels) + 1)
        pos_string_items = ['descript of k.cc.a.1', 'descript of k.cc.b.2']
        neg_string_items = ['descript of k.cc.a.2', 
                            'descript of k.cc.a.3',
                            'descript of k.cc.b.1']
        difference = (set(pos_string_items) | set(neg_string_items)) - set(option_list)
        self.assertEqual(len(difference), 2)
        next_branches = [self.option_retriever.get_pointer_to_next_branch(o, 'standard') for o in option_list]
        self.check_options_match(option_list, next_branches, 'standard')

        # test the case where we ask for wayyy more options than possible
        option_list = self.option_retriever.get_random_standards(positive_labels, 100)
        difference = (set(pos_string_items) | set(neg_string_items)) - set(option_list)
        self.assertEqual(len(difference), 0)
        next_branches = [self.option_retriever.get_pointer_to_next_branch(o, 'standard') for o in option_list]
        self.check_options_match(option_list, next_branches, 'standard') 


if __name__ == "__main__":
    unittest.main()