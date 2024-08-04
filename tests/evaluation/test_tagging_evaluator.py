"""
Tests tagging_evaluator in evaluators.

Author: Lucy Li (@lucy3)
Email:  lucyl@allenai.org
"""

import pathlib
import unittest
from unittest import TestCase

from mathfish.evaluators import TaggingEvaluator

class TestTaggingEvaluate(TestCase):
    def setUp(self):
        data_file = pathlib.Path(__file__).parent.parent / "fixtures" / "toy_tag_eval_data_file.jsonl"
        self.evaluator = TaggingEvaluator(data_file)

    def test_clean_output(self):
        data_file = pathlib.Path(__file__).parent.parent / "fixtures" / "toy_tag_eval_data_file.jsonl"
        evaluator = TaggingEvaluator(data_file) 
        example = "C. Add, subtract, multiply, and divide decimals to hundredths"
        output = evaluator.clean_output(example)
        self.assertEqual(output, ['C'])

        example = " C, D (specifically, the understanding of place value and the relationship"
        output = self.evaluator.clean_output(example)
        self.assertEqual(output, ['C', 'D'])

        example = "None of them align."
        output = self.evaluator.clean_output(example)
        self.assertEqual(output, [])

        data_file = pathlib.Path(__file__).parent.parent / "fixtures" / "toy_tag_eval_data_file2.jsonl"
        evaluator = TaggingEvaluator(data_file) 

        example = "Blah blah.\n\nThought: This problem/activity teaches data analysis.\n\nAnswer: A, D, E"
        output = evaluator.clean_output(example)
        self.assertEqual(output, ['A', "D", "E"])

        example = '  Thought: This problem teaches students.\n\nAnswer: D, H, E, B'
        output = evaluator.clean_output(example)
        self.assertEqual(output, ["B", 'D', "E", "H"]) 

        example = '  Thought: This problem teaches students.\n\nAnswer: none'
        output = evaluator.clean_output(example)
        self.assertEqual(output, [])  

        data_file = pathlib.Path(__file__).parent.parent / "fixtures" / "toy_tag_eval_data_file3.jsonl"
        evaluator = TaggingEvaluator(data_file) 

        example = ' {\n  \"explanation\": \"Problem 1 aligns with standard B.\",\n  \"answer\": \"B, F, G\"\n}'
        output = evaluator.clean_output(example)
        self.assertEqual(output, ['B', 'F', 'G'])

        example = ' {\n  \"explanation\": \"Problem 1 aligns with standard B.\",\n  \"answer\": \"B, F, G\"\n}'
        output = evaluator.clean_output(example)
        self.assertEqual(output, ['B', 'F', 'G']) 

        example = '{\n  \"explanation\": \"This problem involves comparing the sugar content in two different recipes.\",\n  \"answer\": \"B\"\n}'
        output = evaluator.clean_output(example)
        self.assertEqual(output, ['B'])    
        
        example = '{\n  \"explanation\": \"There is triangle congruence.\",\n  \"answer\": \"none\"\n}'
        output = evaluator.clean_output(example)
        self.assertEqual(output, [])    
        
        example = "This is a malformed response."
        output = evaluator.clean_output(example)
        self.assertEqual(output, "This is a malformed response.")

    def test_get_true_label(self): 
        example_id = 'fl_problem_000581'
        label = self.evaluator.get_true_label(example_id)
        self.assertEqual(label, ['B'])

        example_id = 'fl_problem_000112'
        label = self.evaluator.get_true_label(example_id)
        self.assertEqual(label, ['C'])

    def test_calculate_overall_stats(self): 
        bad_results = {
            'fl_problem_000581': {'label': ['A'], 'prediction': ['B']},
            'fl_problem_000112': {'label': ['B'], 'prediction': ['A']}
        }
        metrics = self.evaluator.calculate_overall_stats(bad_results)
        self.assertEqual(metrics['acc'], 0.0)

        good_results = {
            'fl_problem_000581': {'label': ['A', 'B'], 'prediction': ['A', 'B']},
            'fl_problem_000112': {'label': ['C'], 'prediction': ['C']},
        }
        metrics = self.evaluator.calculate_overall_stats(good_results)
        self.assertEqual(metrics['acc'], 1.0)

        good_results = {
            'fl_problem_000581': {'label': ['A', 'B'], 'prediction': ['A']},
            'fl_problem_000112': {'label': ['C'], 'prediction': ['C']},
        }
        metrics = self.evaluator.calculate_overall_stats(good_results)
        self.assertEqual(metrics['weak_acc'], 1.0)

    def test_calculate_subsetted_stats(self): 
        subsetted_dict = {
            '1': {'true': [['A'], ['A']], 'pred': [['B', 'A'], ['B', 'A']], 'example_ids': ['fl_problem_000581', 'fl_problem_000112']},
            '2': {'true': [['A'], ['B']], 'pred': [['A'], ['B']], 'example_ids': ['fl_problem_000581', 'fl_problem_000112']} 
        }
        subset_name = 'Subset'
        stats = self.evaluator.calculate_subsetted_stats([(subsetted_dict, subset_name)])
        self.assertEqual(len(stats), 1)
        self.assertEqual(stats['Subset']['1']['acc'], 0.0)
        self.assertEqual(stats['Subset']['1']['weak_acc'], 1.0)
        self.assertEqual(stats['Subset']['2']['acc'], 1.0)
        self.assertEqual(stats['Subset']['2']['weak_acc'], 1.0)

if __name__ == "__main__":
    unittest.main()