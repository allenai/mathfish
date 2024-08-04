"""
Standardizes all standards listed in
in problem/activity jsonls to match 
a provided standards jsonl.

Author: Lucy Li (@lucy3)
Email:  lucyl@allenai.org
"""

import os
import json
from collections import defaultdict
from typing import List

class StandardStandardizer: 
    '''
    Standardizes Common Core standard labels. 
    
    Removes "HS" from high school standards, and also matches standards 
    that may use different punctuation formats. 
    
    Example: S-MD.5 == HSS-MD.B.5 == S-MD.B.5 (the last one
    is the "standardized" format)
    '''
    
    def __init__(self, standards_path: str):
        self.math_standards = set()
        self.math_standards_descriptions = dict()
        with open(standards_path, 'r') as infile: 
            for line in infile: 
                d = json.loads(line)
                self.math_standards.add(d['id'])
                self.math_standards_descriptions[d['id']] = d['description']
                    
        self.no_punct_standard = defaultdict(list)
        for s in self.math_standards: 
            # sometimes standards are written with or without '.' or '-'
            no_punct_s = s.replace('-', '').replace('.', '').lower()
            self.no_punct_standard[no_punct_s].append(s)
            
            components = s.replace('-', '.').split('.')
            if len(components) == 4: 
                # this accounts for cases such as 1.OA.A.1 == 1.OA.1 
                new_s = '.'.join([components[0], components[1], components[3]]).replace('.', '').lower()
                self.no_punct_standard[new_s].append(s)

        for no_punct_s in self.no_punct_standard: 
            # standards without punctuation are unique
            assert len(self.no_punct_standard[no_punct_s]) == 1
            
    def standardize_single_standard(self, s: str): 
        new_s = None
        if s.startswith('HS') and s[2:] in self.math_standards: 
            # high school standards can start with or without "HS"
            new_s = s[2:]
        elif s not in self.math_standards: 
            no_punct_s = s.replace('-', '').replace('.', '').lower()
            if no_punct_s not in self.no_punct_standard: 
                if s.startswith('HS'): 
                    s = s[2:]
                    no_punct_s = s.replace('-', '').replace('.', '').lower()
            if no_punct_s in self.no_punct_standard: 
                new_s = self.no_punct_standard[no_punct_s][0]
            else: 
                print("Cannot find:", s)
        else: 
            new_s = s
        return new_s
    
    def standardize_standards(self, in_path: str, out_path: str): 
        '''
        Takes in standards listed in in_path and then 
        outputs new file containing standardized standards.
        '''
        outfile = open(out_path, 'w')
        with open(in_path, 'r') as infile: 
            for line in infile: 
                d = json.loads(line)
                new_standards = []
                for tup in d['standards']: 
                    new_s = self.standardize_single_standard(tup[1])
                    if new_s: 
                        new_standards.append([tup[0], new_s])
                if len(d['standards']) != len(new_standards): 
                    print("Some standards were not standardizable!")
                d['standards'] = new_standards
                outfile.write(json.dumps(d) + '\n')
        outfile.close()
    
    def data_is_standardized(self, data_path: str): 
        '''
        Returns True if the data in the input path is standardized,
        otherwise prints out malformed standards and returns False.
        '''
        is_standardized = True
        with open(data_path, 'r') as infile: 
            for line in infile: 
                d = json.loads(line)
                for tup in d['standards']: 
                    if tup[1] not in self.math_standards: 
                        print("Not a standardized standard:", tup[1])
                        is_standardized = False
        return is_standardized

    def standard_is_standardized(self, s:str):
        return s in self.math_standards

    def get_standard_description(self, s:str):
        '''
        Returns the description of the standardized standard, if it exists
        '''
        standard = self.standardize_single_standard(s)
        descr = self.math_standards_descriptions.get(standard, None)
        return descr


