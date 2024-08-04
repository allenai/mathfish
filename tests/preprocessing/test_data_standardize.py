"""
Tests data_standardize.py in preprocessors

Author: Lucy Li (@lucy3), Kyle Lo (@kyleclo)
Email:  lucyl@allenai.org, kylel@allenai.org
"""

import json
import pathlib
import unittest
from unittest import TestCase

from mathfish.preprocessors.data_standardize import StandardStandardizer


class TestStandardStandardizing(TestCase):

    def setUp(self):
        self.standards_path = pathlib.Path(__file__).parent.parent / "fixtures" / "toy_standards.jsonl"
        self.standardizer = StandardStandardizer(self.standards_path)
        self.nonstandardized_standards_path = (
            pathlib.Path(__file__).parent.parent / "fixtures" / "toy_nonstandardized_standards.jsonl"
        )
        self.standardized_standards_path = (
            pathlib.Path(__file__).parent.parent / "fixtures" / "toy_standardized_standards.jsonl"
        )

    def test_standardize_single_standard(self):
        self.assertEqual(self.standardizer.standardize_single_standard("HSS-IC.B.5"), "S-IC.B.5")
        self.assertEqual(self.standardizer.standardize_single_standard("S-IC.5"), "S-IC.B.5")
        self.assertEqual(self.standardizer.standardize_single_standard("HSF-IF.C.7.e"), "F-IF.C.7e")
        self.assertEqual(self.standardizer.standardize_single_standard("3.MD.C.7.c"), "3.MD.C.7c")
        self.assertEqual(self.standardizer.standardize_single_standard("6.RP.A.3.b"), "6.RP.A.3b")
        self.assertEqual(self.standardizer.standardize_single_standard("F.IF.C.7.A"), "F-IF.C.7a") 

    def test_standardize_standards(self):

        self.standardizer.standardize_standards(
            in_path=self.nonstandardized_standards_path, out_path=self.standardized_standards_path
        )

        with open(self.standardized_standards_path, "r") as infile:
            d = json.load(infile)
            self.assertEqual(
                sorted(d["standards"]),
                sorted(
                    [
                        ["Alignment", "S-IC.B.5"],
                        ["Alignment", "F-IF.C.7e"],
                        ["Alignment", "3.MD.C.7c"],
                        ["Alignment", "6.RP.A.3b"],
                        ["Addressing", "F-IF.C.7a"],
                    ]
                ),
            )

        self.assertTrue(self.standardizer.data_is_standardized(data_path=self.standardized_standards_path))
        self.assertFalse(self.standardizer.data_is_standardized(data_path=self.nonstandardized_standards_path))

    def test_standard_description(self):
        self.assertEqual(self.standardizer.get_standard_description("HSS-IC.B.5"), "Use data from a randomized experiment to compare two treatments; use simulations to decide if differences between parameters are significant.")
        self.assertEqual(self.standardizer.get_standard_description("S-IC.5"), "Use data from a randomized experiment to compare two treatments; use simulations to decide if differences between parameters are significant.")
        self.assertEqual(self.standardizer.get_standard_description("HSF-IF.C.7.e"), "Graph exponential and logarithmic functions, showing intercepts and end behavior, and trigonometric functions, showing period, midline, and amplitude.")
        self.assertEqual(self.standardizer.get_standard_description("3.MD.C.7.c"), "Use tiling to show in a concrete case that the area of a rectangle with whole-number side lengths a and b + c is the sum of a \u00d7 b and a \u00d7 c. Use area models to represent the distributive property in mathematical reasoning.")
        self.assertEqual(self.standardizer.get_standard_description("6.RP.A.3.b"), "Solve unit rate problems including those involving unit pricing and constant speed. For example, if it took 7 hours to mow 4 lawns, then at that rate, how many lawns could be mowed in 35 hours? At what rate were lawns being mowed?")
        self.assertEqual(self.standardizer.get_standard_description("F.IF.C.7.A"), "Graph linear and quadratic functions and show intercepts, maxima, and minima.")

if __name__ == "__main__":
    unittest.main()
