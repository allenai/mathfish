"""
After we obtained data from Fishtank Learning and Illustrative Math,
we cleaned up their text and standards labels. 

Author: Lucy Li (@lucy3)
Email:  lucyl@allenai.org
"""

from mathfish.preprocessors.data_standardize import StandardStandardizer
import os
import json
from collections import defaultdict, Counter
from tqdm import tqdm
import csv
from utils import *
import pprint

ROOT = '/net/nfs.cirrascale/s2-research/lucyl/edreports'
DATA = os.path.join(ROOT, 'Illustrative_Math')

def cleanup_text_helper(text): 
    text = text.replace('<br>', '')
    text = text.replace(u'\xa0', u' ')
    lines = text.split('\n')
    new_lines = []
    for l in lines: 
        new_lines.append(l.strip())
    text = '\n'.join(new_lines)
    return text

def cleanup_text():
    in_folder = os.path.join(ROOT, 'Illustrative_Math/v1')
    out_folder = os.path.join(ROOT, 'Illustrative_Math/v2')
    for f in os.listdir(in_folder): 
        if not f.endswith('.jsonl'): continue
        print(f)
        with open(os.path.join(out_folder, f), 'w') as outfile: 
            with open(os.path.join(in_folder, f), 'r') as infile: 
                for line in infile: 
                    d = json.loads(line)
                    d['text'] = cleanup_text_helper(d['text'])
                    outfile.write(json.dumps(d) + '\n')

    in_folder = os.path.join(ROOT, 'Fishtank_Learning/v1')
    out_folder = os.path.join(ROOT, 'Fishtank_Learning/v2')
    for f in os.listdir(in_folder): 
        if not f.endswith('.jsonl'): continue
        print(f)
        with open(os.path.join(out_folder, f), 'w') as outfile: 
            with open(os.path.join(in_folder, f), 'r') as infile: 
                for line in infile: 
                    d = json.loads(line)
                    d['text'] = cleanup_text_helper(d['text'])
                    outfile.write(json.dumps(d) + '\n')

def standardize_data(): 
    '''
    Makes all IM standards look like the ones in 
    standards.jsonl.   
    '''
    standards_path = os.path.join(ROOT, 'Achieve_the_Core', 'standards.jsonl')

    standardizer = StandardStandardizer(standards_path)
    in_folder = os.path.join(ROOT, 'Illustrative_Math/v2')
    out_folder = os.path.join(ROOT, 'Illustrative_Math')
    for f in os.listdir(in_folder): 
        if not f.endswith('.jsonl'): continue
        standardizer.standardize_standards(os.path.join(in_folder, f), os.path.join(out_folder, f))
        assert standardizer.data_is_standardized(os.path.join(out_folder, f))

    in_folder = os.path.join(ROOT, 'Fishtank_Learning/v2')
    out_folder = os.path.join(ROOT, 'Fishtank_Learning')
    for f in os.listdir(in_folder): 
        if not f.endswith('.jsonl'): continue
        standardizer.standardize_standards(os.path.join(in_folder, f), os.path.join(out_folder, f))
        assert standardizer.data_is_standardized(os.path.join(out_folder, f))
        
if __name__=="__main__": 
    cleanup_text()
    standardize_data()