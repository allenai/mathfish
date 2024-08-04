"""
Tests verification_evaluator in evaluators.

Author: Lucy Li (@lucy3)
Email:  lucyl@allenai.org
"""

import pathlib
import unittest
from unittest import TestCase

from mathfish.evaluators import VerificationEvaluator

class TestverificationEvaluate(TestCase):
    def setUp(self):
        data_file = pathlib.Path(__file__).parent.parent / "fixtures" / "toy_eval_data_file.jsonl"
        self.evaluator = VerificationEvaluator(data_file)

    def test_clean_output(self):
        example_no = "No."
        output = self.evaluator.clean_output(example_no)
        self.assertEqual(output, "no")

        example_yes = "Yes"
        output = self.evaluator.clean_output(example_yes)
        self.assertEqual(output, "yes")

        example_yes = "Alignment with description: Yes, the problem aligns with the description."
        output = self.evaluator.clean_output(example_yes)
        self.assertEqual(output, "yes")
        
        example_other = "This is a malformed response."
        output = self.evaluator.clean_output(example_other)
        self.assertEqual(output, "This is a malformed response.")

    def test_get_true_label(self): 
        example_id = 'im_center_000001_neg_2'
        label = self.evaluator.get_true_label(example_id)
        self.assertEqual(label, 'no')

        example_id = 'im_task_000019_pos_0'
        label = self.evaluator.get_true_label(example_id)
        self.assertEqual(label, 'yes')

    def test_calculate_overall_stats(self): 
        bad_results = {
            'fl_problem_000581': {'label': 'yes', 'prediction': 'no'},
            'fl_problem_000112': {'label': 'no', 'prediction': 'yes'}
        }
        metrics = self.evaluator.calculate_overall_stats(bad_results)
        self.assertEqual(metrics['acc'], 0.0)
        self.assertEqual(metrics['f1'], 0.0)

        good_results = {
            'fl_problem_000581': {'label': 'yes', 'prediction': 'yes'},
            'fl_problem_000112': {'label': 'no', 'prediction': 'no'},
        }
        metrics = self.evaluator.calculate_overall_stats(good_results)
        self.assertEqual(metrics['acc'], 1.0)
        self.assertEqual(metrics['f1'], 1.0)

    def test_calculate_subsetted_stats(self): 
        subsetted_dict = {
            '1': {'true': ['no', 'yes'], 'pred': ['yes', 'no'], 'example_ids': ['fl_problem_000581', 'fl_problem_000112']},
            '2': {'true': ['no', 'yes'], 'pred': ['no', 'yes'], 'example_ids': ['fl_problem_000581', 'fl_problem_000112']},
        }
        subset_name = 'Subset'
        stats = self.evaluator.calculate_subsetted_stats([(subsetted_dict, subset_name)])
        self.assertEqual(len(stats), 1)
        self.assertEqual(stats['Subset']['1']['acc'], 0.0)
        self.assertEqual(stats['Subset']['1']['f1'], 0.0)
        self.assertEqual(stats['Subset']['2']['acc'], 1.0)
        self.assertEqual(stats['Subset']['2']['f1'], 1.0)

if __name__ == "__main__":
    unittest.main()