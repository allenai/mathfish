"""
Tests data_augment.py in preprocessors

Author: Lucy Li (@lucy3), Kyle Lo (@kyleclo)
Email:  lucyl@allenai.org, kylel@allenai.org
"""

import pathlib
import unittest
from unittest import TestCase

from mathfish.preprocessors.data_augment import DataExpander


class TestDataAugment(TestCase):

    def setUp(self):
        '''
        Note that all of the cases in this file are
        toy examples and not reflective of the full list of standards
        '''
        self.standards_path = pathlib.Path(__file__).parent.parent / "fixtures" / "toy_family_standards.jsonl"
        self.expander = DataExpander(self.standards_path)

    def test_inherit_to_standard_level(self):
        # example 1: convert all labels to standards level
        standards = [["Alignment", "S-IC.B"]]
        inherited_standards = self.expander.inherit_to_standard_level(standards)
        correct_standards_only = [
            ["Alignment", "S-IC.B.3"],
            ["Alignment", "S-IC.B.4"],
            ["Alignment", "S-IC.B.5"],
            ["Alignment", "S-IC.B.6"],
        ]
        self.assertTrue(len(correct_standards_only) > len(standards))
        self.assertEqual(sorted(correct_standards_only), sorted(inherited_standards))

        # example 2: convert all labels to standards level
        standards = [["Alignment", "F-IF.C.7e"], ["Alignment", "F-IF.C.7d"]]
        inherited_standards = self.expander.inherit_to_standard_level(standards)
        correct_standards_only = [["Alignment", "F-IF.C.7"]]
        self.assertTrue(len(correct_standards_only) < len(standards))
        self.assertEqual(sorted(correct_standards_only), sorted(inherited_standards))

        # example 2 extended: convert all labels to standards level and keep the originals too
        standards = [["Alignment", "F-IF.C.7e"]]
        inherited_standards = self.expander.inherit_to_standard_level(standards, keep_other_levels=True)
        correct_all_labels = [["Alignment", "F-IF.C.7"], ["Alignment", "F-IF.C.7e"]]
        self.assertTrue(len(correct_all_labels) > len(standards))
        self.assertEqual(sorted(correct_all_labels), sorted(inherited_standards))

    def test_get_negative_examples_by_grade_and_domain(self):
        standards = ["S-IC.B.5"]
        # sample *up to* 100 examples from same grade same domain
        neg_examples = self.expander.get_negative_examples_by_grade_and_domain(
            standards, n_sample=100, neg_strat='same-domain-same-grade'
        )
        correct_answer = ["S-IC.B.3", "S-IC.B.4", "S-IC.B.6", "S-ID.C.9"]
        self.assertTrue(len(correct_answer) <= 100) 
        self.assertEqual(sorted(correct_answer), sorted(neg_examples))
        neg_examples, strats = self.expander.get_negative_examples_with_strat(
            standards, n_sample=100, neg_strat='same-domain-same-grade'
        )
        self.assertEqual(set(strats), set(['same-domain-same-grade']))
        self.assertEqual(sorted(correct_answer), sorted(neg_examples))

        # sample 1 example from same grade but different domain
        neg_examples = self.expander.get_negative_examples_by_grade_and_domain(
            standards, n_sample=1, neg_strat='different-domain-same-grade'
        )
        correct_answer = ["F-IF.C.7"]
        self.assertTrue(len(correct_answer) <= 1) 
        self.assertEqual(sorted(correct_answer), sorted(neg_examples))
        neg_examples, strats = self.expander.get_negative_examples_with_strat(
            standards, n_sample=100, neg_strat='different-domain-same-grade'
        )
        self.assertEqual(set(strats), set(['different-domain-same-grade']))
        self.assertEqual(sorted(correct_answer), sorted(neg_examples))

        # sample 1 example from different grade but same domain
        neg_examples = self.expander.get_negative_examples_by_grade_and_domain(
            standards, n_sample=1, neg_strat='same-domain-different-grade'
        )
        correct_answer = ["6.SP.A.2"]
        self.assertTrue(len(correct_answer) <= 1) 
        self.assertEqual(sorted(correct_answer), sorted(neg_examples))
        neg_examples, strats = self.expander.get_negative_examples_with_strat(
            standards, n_sample=100, neg_strat='same-domain-different-grade'
        )
        self.assertEqual(set(strats), set(['same-domain-different-grade']))
        self.assertEqual(sorted(correct_answer), sorted(neg_examples)) 

    def test_get_negative_examples_by_connections(self):
        # these two standards -> one standard
        neg_examples = self.expander.get_negative_examples_by_connections(["S-IC.B.3", "S-IC.B.4"], relation_type='progress to')
        self.assertEqual(neg_examples, ["S-IC.B.6"])
        # sampling 1 should yield same result
        neg_examples, strats = self.expander.get_negative_examples_with_strat(["S-IC.B.3", "S-IC.B.4"], n_sample=1, neg_strat='neighbors')
        self.assertEqual(set(strats), set(['neighbors']))
        self.assertEqual(neg_examples, ["S-IC.B.6"])

        # this one standard <- three other standards
        neg_examples = self.expander.get_negative_examples_by_connections(["S-IC.B.6"], relation_type='progress from')
        self.assertEqual(sorted(neg_examples), ["S-IC.B.3", "S-IC.B.4", "S-IC.B.5"])

        # this one standard <- three other standards, but we sample only one
        neg_examples = self.expander.get_negative_examples_by_connections(["S-IC.B.6"], relation_type='progress from', n_sample=1)
        self.assertTrue(len(neg_examples) == 1)
        self.assertTrue(neg_examples[0] in ["S-IC.B.3", "S-IC.B.4", "S-IC.B.5"])

        # this standard -- one other standard 
        neg_examples = self.expander.get_negative_examples_by_connections(["S-IC.B.6"], relation_type='related')
        self.assertEqual(sorted(neg_examples), ["S-ID.C.9"])

        # "S-IC.B.3" does not have any related neighbors so the output is the same as the previous
        neg_examples = self.expander.get_negative_examples_by_connections(["S-IC.B.6", "S-IC.B.3"], relation_type='related')
        self.assertEqual(sorted(neg_examples), ["S-ID.C.9"])

    def test_get_negative_examples_all(self): 
        standards = ["S-IC.B.5"]
        neg_examples, strats = self.expander.get_negative_examples_with_strat(standards, n_sample=1, neg_strat="all-negative-types") 
        same_same_answer_pool = ["S-IC.B.3", "S-IC.B.4", "S-IC.B.6", "S-ID.C.9"]
        same_diff_answer_pool = ["F-IF.C.7"]
        diff_same_answer_pool = ["6.SP.A.2"]
        diff_diff_answer_pool = ["6.G.A.2"]
        atc_answer_pool = ["S-IC.B.6"]
        self.assertTrue(set(neg_examples) & set(same_same_answer_pool))
        self.assertTrue(set(neg_examples) & set(same_diff_answer_pool))  
        self.assertTrue(set(neg_examples) & set(diff_same_answer_pool)) 
        self.assertTrue(set(neg_examples) & set(diff_diff_answer_pool)) 
        self.assertTrue(set(neg_examples) & set(atc_answer_pool)) 
        correct_strats = ['different-domain-different-grade', 'same-domain-different-grade', 'different-domain-same-grade', 'same-domain-same-grade', 'neighbors']
        self.assertEqual(sorted(strats), sorted(correct_strats))

        

if __name__ == "__main__":
    unittest.main()
