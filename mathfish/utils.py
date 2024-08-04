"""
Helpful utility functions that may be shared across
files.  

Author: Lucy Li (@lucy3)
Email:  lucyl@allenai.org
"""
from typing import List
import string

def get_domain_cat(s: str):
    '''
    Get domain of standard s
    '''
    if '-' in s:
        return s.split('-')[0]
    else:
        parts = s.split('.')
        if len(parts) == 1:
            # e.g. s is a grade level
            return 'None'
        else:
            return parts[1]

def get_grade(s: str):
    '''
    Get grade of standard s
    '''
    if '-' in s:
        return 'HS'
    else:
        return s.split('.')[0]
    
def map_grade_to_number(g): 
    assert g in ['K', '1', '2', '3', '4', '5', '6', '7', '8', 'HS']
    if g == 'HS': 
        g = 9
    elif g == 'K': 
        g = 0
    g = int(g)
    return g

def map_number_to_grade(g): 
    assert g in range(10)
    if g > 0 and g < 9: 
        g = str(g)
    elif g == 0: 
        g = 'K'
    elif g == 9: 
        g = 'HS'
    return g

def get_grade_level_distance(positive_standards: List[str], negative_standard: str): 
    '''
    Get the distance between the highest grade level in a list
    of positive standards and the grade level of a negative standard

    e.g. 
    - positive_standards = ['K.OA.A.1', 'K.CC.A.1']
    - negative_standard = '1.NBT.A.1'
    - returns: (dist, neg_grade, max_pos_grade), or (1, 1, 0), because the highest grade level 
    in the positive set is K, and first grade is +1 away from kindergarten. 
    '''
    g2 = get_grade(negative_standard)
    g2 = map_grade_to_number(g2)

    max_grade = get_max_grade(positive_standards)

    return g2 - max_grade, g2, max_grade

def get_max_grade(positive_standards: List[str]): 
    max_grade = 0
    for s1 in positive_standards: 
        g1 = get_grade(s1)
        g1 = map_grade_to_number(g1)
        max_grade = max(g1, max_grade)
    return max_grade

def get_options(): 
    options = list(string.ascii_uppercase)
    more_options = []
    for letter1 in options: 
        for letter2 in options: 
            more_options.append(letter1 + letter2)
    options += more_options
    return options