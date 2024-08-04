'''
Parent class for various classes
that create different datasets

Author: Tal August
Email:  tala@allenai.org
'''
from mathfish.utils import *
from mathfish.preprocessors import DataReformatter, DataExpander
import json 

class BaseDataset:
    def __init__(self, standards_path: str, prompt_name: str, prompt_file: str,
                 table_style='special_token', image_style='special_token'):
        self.prompt_name = prompt_name

        with open(prompt_file, mode='rt', encoding='utf-8') as infile: 
            prompts = json.load(infile)
        self.verification_prompts = prompts['verification']
        self.tagging_prompts = prompts['tagging']

        self.expander = DataExpander(standards_path=standards_path)
        self.reformatter = DataReformatter(table_style=table_style, image_style=image_style)

        self.instances = []
        self.table_style = table_style

    def __get__(self, i: int) -> dict:
        if i >= len(self.instances): 
            return None
        return self.instances[i]
    
    def __len__(self) -> int:
        return len(self.instances)
    
    def load_instances(self, instance_path: str):
        with open(instance_path, "r") as f:
            self.instances = [json.loads(x) for x in f.readlines()]

    def get_example_count(self): 
        raise NotImplementedError("This function is implemented by children")
    

    def output_dataset(self, output_path: str, input_prefix: str): 
        raise NotImplementedError("This function is implemented by children")

    def clean_problem_text(self, text, elements):
        '''
        Replaces ###IMAGEX### and ###TABX### in passed text 
        with special tokens
        '''
        for ele in elements: 
            if 'IMAGE' in ele: 
                img = self.reformatter.get_image(elements[ele])
                # this assertion is here for now since we are currently only working with LLMs
                assert type(img) == str
                text = text.replace(ele, img)
            elif 'TABLE' in ele: 
                tbl = self.reformatter.get_table(elements[ele])
                text = text.replace(ele, tbl)
        return text
    
