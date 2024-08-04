"""
These functions were originally used by problem
scraping scripts.

See mathfish/preprocessors
for more extensive cleanup tools.

Author: Lucy Li (@lucy3)
Email:  lucyl@allenai.org
"""

import os
import json

ROOT = '/home/lucyl/edreports'

def get_math_standards(): 
    with open(os.path.join(ROOT, 'data/standards.js'), 'r') as infile: 
        standards_dict = json.load(infile)
    math_standards = {}
    for standard in standards_dict['math']: 
        standard = standard['id'].strip()
        standard_nopunct = standard.replace('-', '').replace('.', '')
        math_standards[standard_nopunct] = standard
    return math_standards

def standardize_standard(standard, math_standards): 
    standard_nopunct = standard.replace('-', '').replace('.', '')
    if standard_nopunct not in math_standards: 
        standard = standard
    else: 
        standard = math_standards[standard_nopunct]
    return standard