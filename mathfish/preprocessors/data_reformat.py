"""
Given dataset elements, transform them 
into different formats to stress test
models. 

Author: Lucy Li (@lucy3)
Email:  lucyl@allenai.org
"""
import json
from collections import defaultdict
from typing import List
from mathfish.utils import *
import html_to_json
import dashtable

class DataReformatter:
    def __init__(self, table_style='special_token', image_style='special_token'):
        self.table_style = table_style
        self.image_style = image_style
        assert self.table_style in set(['special_token', 'html', 'json', 'rst', 'markdown'])
        assert self.image_style in set(['special_token'])

    def get_table(self, table_html:str): 
        '''
        Input: table html
        Output: a string representing a table in a variety of formats
        including JSON, reStructuredText, Markdown
        '''
        if self.table_style == 'special_token': 
            return '[TAB]'
        if self.table_style == 'html': 
            return table_html
        if self.table_style == 'json': 
            return json.dumps(html_to_json.convert(table_html))
        if self.table_style == 'rst': 
            return dashtable.html2rst(table_html)
        if self.table_style == 'markdown': 
            return dashtable.html2md(table_html)
        
    def get_image(self, image_path: str): 
        '''
        Input: str representing image filename
        Output: something representing an image
        '''
        assert self.image_style in set(['special_token'])
        if self.image_style == 'special_token': 
            return '[IMG]'