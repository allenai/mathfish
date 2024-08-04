"""
Validates all of our input data (every *.jsonl that we are 
working with currently) to match specifications in the data readme

Author: Lucy Li (@lucy3)
Email:  lucyl@allenai.org
"""

import argparse
import json
import datetime

def check_standards_format(input_path): 
    all_children = set()
    all_parents = set()
    all_standards = set()
    
    with open(input_path, 'r') as infile: 
        for line in infile: 
            d = json.loads(line)
            
            assert 'id' in d and type(d['id']) == str
            all_standards.add(d['id'])
            
            assert 'description' in d and type(d['description']) == str
            assert 'source' in d and type(d['source']) == str
            
            assert 'level' in d and type(d['level']) == str
            assert d['level'] in set(['Grade', 'Domain', 'HS Category', 'Cluster', 'Standard', 'Sub-standard'])
            
            assert 'cluster_type' in d and type(d['cluster_type']) == str
            assert d['cluster_type'] in set(['major cluster', 'widely applicable prerequisite', 
                                             'additional cluster', 'supporting cluster', 'none'])
            
            assert 'aspects' in d and type(d['aspects']) == list
            for aspect in d['aspects']: 
                assert aspect in set(["Application", "conceptual understanding", "Procedural Skill and Fluency"])
            
            assert 'parent' in d and type(d['parent']) == str
            all_parents.add(d['parent'])
            
            assert 'children' in d and type(d['children']) == list
            all_children.update(d['children'])
            
    assert all_children - all_standards == set()
    
    # some items have no parents a.k.a parent is an empty string
    assert all_parents - all_standards == set([''])
    
def validate(date_text):
    try:
        datetime.date.fromisoformat(date_text)
    except ValueError:
        raise ValueError("Incorrect data format, should be YYYY-MM-DD")
    
def check_problem_activity_format(input_path): 
    with open(input_path, 'r') as infile: 
        for line in infile: 
            d = json.loads(line)
            
            assert 'id' in d and type(d['id']) == str
            assert 'metadata' in d and type(d['metadata']) == dict

            assert 'problem_activity_type' in d['metadata'] and type(d['metadata']['problem_activity_type']) == str
            assert 'url' in d['metadata'] 
            
            assert 'text' in d and type(d['text']) == str
            assert 'elements' in d and type(d['elements']) == dict
            assert 'standards' in d and type(d['standards']) == list
            for tup in d['standards']: 
                # relation and standard pairs
                assert len(tup) == 2 and type(tup[0]) == str and type(tup[1]) == str
            assert 'acquisition_date' in d and type(d['acquisition_date']) == str
            validate(d['acquisition_date'])
            
            assert 'source' in d and type(d['source']) == str
            

if __name__ == "__main__":
    # argparse input files and file types
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, choices=['standards', 'problems-activities'], required=True)
    parser.add_argument('--input_path', type=str, required=True)
    args = parser.parse_args()
    
    if args.data_type == 'standards': 
        check_standards_format(args.input_path)
    elif args.data_type == 'problems-activities': 
        check_problem_activity_format(args.input_path)