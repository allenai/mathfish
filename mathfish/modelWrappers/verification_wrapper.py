'''
Wrapper for a yes/no verification task format, where
problems/activities are paired with positive and
negative standard labels' descriptions in prompts.

Author: Tal August
Email:  tala@allenai.org
'''
from mathfish.utils import *
from mathfish.modelWrappers import BaseModelWrapper
import together
import os
import openai
# from edreports.evaluators import VerificationEvaluator
import re


class VerificationWrapper(BaseModelWrapper):


    '''
    Returns messages in a yes/no verification task format with tulu chat as the message structure.
    Assumes that the user message has a description and problem_activity, and that the prompt template as string formatting 
    arguements for both the description and problem_activity.
    '''
    def format_prompt(self, message):
        if message.get('role') == 'user' and message.get('standard_description') and message.get('problem_activity'):
            if message.get('few_shots'):
                formatted_prompt = message['prompt_template'].format(problem_activity=message['problem_activity'], 
                                                                     standard_description=message['standard_description'],
                                                                     few_shots=message['few_shots']) 
            else: 
                formatted_prompt = message['prompt_template'].format(problem_activity=message['problem_activity'], standard_description=message['standard_description'])
            return {'role':'user', 'content':formatted_prompt}
        elif message.get('role') == 'assistant' and message.get('response_template'):
            formatted_response = message['response_template'].format(aligns=message['aligns'])
            return {'role':'assistant', 'content':formatted_response}
        
        raise ValueError("Message must have a role of 'user' or 'assistant' and the user message must have a standard_description and problem_activity instead of {}".format(message.keys()))   


    # def format_messages(self, messages):
    #     assert all(message["role"] in ["user", "assistant"] for message in messages), \
    #                 "Each message should have a `role` of either 'user' or 'assistant'."

    #     if self.truncate:
    #         messages = self.truncate_messages(messages)

    #     formatted_messages = [self.format_prompt(message) for message in messages]

    #     return formatted_messages




    def truncate_messages(self, messages):
        '''
        Truncate the messages to fit within the max_length
        Assumes that there is at most 2 messages, the first being the user message.
        If the prompt template, description, and problem activity are too long, the problem activity is truncated.

        The messages should look like this: 
        {
            role: 'user', 
            prompt_template: '',
            standard_description: '',
            problem_activity: '',
        } 

        returns messages in same form as passed, just with the problem_activity possibly truncated.
        '''
        assert len(messages) <= 2, "VerificationWrapper only supports one pair of messages at a time. Please provide maxmimum of two messages, the first being the user message."
        assert messages[0].get('role') == "user", "first message should be user message."

        # tokenize the prompt template, description, and problem activity seperately 
        template = self.tokenizer(messages[0].get('prompt_template'), truncation=False, return_length=True, add_special_tokens=False)
        description = self.tokenizer(messages[0].get('standard_description'), truncation=False, return_length=True, add_special_tokens=False)
        problem_activity = self.tokenizer(messages[0].get('problem_activity'), truncation=False, return_length=True, add_special_tokens=False)

        current_length = template['length'][0] + description['length'][0] + problem_activity['length'][0]

        if current_length > (self.max_length - self.buffer):
            max_problem_length = self.max_length - template['length'][0] - description['length'][0] - self.buffer

            self.log({'event': 'truncation-needed', 'length': current_length})

            if max_problem_length < 0:
                raise ValueError("The prompt template and standard_description are too long to fit in the max_length.")
            
            # truncate the problem activity
            problem_activity['input_ids'] = problem_activity['input_ids'][:max_problem_length]
            problem_activity['length'] = max_problem_length

        # need to turn it back into a string for correct string formatting
        # this might be an issue if tokenizer does not return correct string formatting 
        messages[0]['problem_activity'] = self.tokenizer.decode(problem_activity['input_ids'])

        return messages
    
    def verify_response(self, response: str, response_format: str):
        '''
        Check if the response is a yes or no response.
        '''
        assert response_format is None, "VerificationWrapper only supports one type of response, so response format should be None, not: {}.".format(response_format)
        
        model_response = response.choices[0].message.content

        split_res = re.split('. |\s |\n', model_response) 
        for r in split_res:
            cleaned_r = re.sub('\.|,|\s', '', r.lower())
            if cleaned_r in ['yes', 'no']:
                return cleaned_r, None
            
        return None, 'Could not parse response into yes or no'