'''
Wrapper for a tagging task format, where
problems/activities are paired with possible standards that a model must select between

Author: Tal August
Email:  tala@allenai.org
'''
from mathfish.utils import *
from mathfish.modelWrappers import BaseModelWrapper
import numpy as np
import string
import re
import json

class TaggingWrapper(BaseModelWrapper):
    def __init__(self, model_name_or_path, tokenizer_name_or_path, openai_client=None, max_length=None, buffer=100, truncate=True, is_multi_turn=False):
        super().__init__(model_name_or_path, tokenizer_name_or_path, openai_client, max_length, buffer, truncate)
        # A...Z, AA...AZ,
        self.options = list(string.ascii_uppercase)
        more_options = []
        for letter1 in self.options:
            for letter2 in self.options:
                more_options.append(letter1 + letter2)
        self.options += more_options
        self.is_multi_turn = is_multi_turn # TODO not currently using, but might not have to


    def format_prompt(self, message):
        option_string = ''

        if message.get('role') == 'user':
            assert message.get('options'), "all user messages must have options, not: {}".format(message.keys())

            option_list = message.get('options')
            for i in range(len(option_list)):
                option_string += self.options[i] + '. ' + option_list[i] + '\n'

            # check if it has the problem or just options (for multi turn)
            if message.get('problem_activity'):
                if message.get('few_shots'): 
                    formatted_prompt = message['prompt_template'].format(problem_activity=message['problem_activity'], 
                                                                         few_shots=message['few_shots'],
                                                                         options=option_string)
                else: 
                    formatted_prompt = message['prompt_template'].format(problem_activity=message['problem_activity'], options=option_string)
            else:
                formatted_prompt = message['prompt_template'].format(options=option_string)

            return {'role':'user', 'content':formatted_prompt}

        elif message.get('role') == 'assistant':
            if message.get('content'):
                return message
            elif message.get('response_template') and message.get('correct_option_index'):
                correct_option_index = message.get('correct_option_index')
                correct_option_string = ','.join([self.options[int(i)] for i in correct_option_index])

                formatted_response = message['response_template'].format(option=correct_option_string)
                return {'role':'assistant', 'content':formatted_response}
            elif message.get('response_template'): 
                return {'role':'assistant', 'content':''} 

        raise ValueError("Message must have a role of 'user' or 'assistant' and the user message must have options and problem_activity instead of {}".format(message.keys()))


    def get_message_length(self, message):
        '''
        determines how long a single message is, useful for figuring out how much to truncate
        Moslty used for multi-turn, also assumes that all messages just have the 'options' field to include
        '''
        if message.get('role') == 'user' and message.get('options'):
            assert not message.get('problem_activity'), "user message in multi-turn has a problem description, which it shouldn't have: {}".format(message)

            option_string = ''
            option_list = message.get('options')
            for i in range(len(option_list)):
                option_string += self.options[i] + '. ' + option_list[i] + '\n'

            formatted_prompt = message['prompt_template'].format(options=option_string)
            return self.tokenizer(formatted_prompt, truncation=False, return_length=True, add_special_tokens=False)['length'][0]

        elif message.get('role') == 'assistant' and message.get('content'):
            return self.tokenizer(message.get('content'), truncation=False, return_length=True, add_special_tokens=False)['length'][0]

        raise ValueError("Message must have a role of 'user' or 'assistant' and the user message must have options instead of {}".format(message.keys()))

    def truncate_messages(self, messages):
        '''
        Update: Right now just truncating the problem text
        first check if the tokenized prompt fits within the max_length
        If not, split the prompts into a series of prompts, each which has the full problem but only a subset of the labels
        Min subset of labels is 3.
        If with 3 labels the prompt is still too long, truncate the problem text
        Right now assuming single turn
        {
            role: 'user',
            prompt_template: '',
            options: [], # these are options to include in prompt
            problem_activity: '',
        }
        Returns a list of tokenized prompts (since there may be splitting of options across prompts)
        '''
        # if not self.is_multi_turn:
        #     assert len(messages) <= 2, "TaggingWrapper only supports one pair of messages at a time if single turn. Please provide maxmimum of two messages, the first being the user message or update is_multi_turn to True."
        # else:
        #     assert len(messages) > 2, "TaggingWrapper with multi turn enables needs more than 2 messages"

        # tokenize everything seperately to get lengths
        message_index = 0

        assert messages[message_index].get('role') == "user", "first message should be user message."

        # get lengths of all the messages we aren't truncating (ignoring last and first message - this is for handling multi-turn)
        existing_message_lengths = np.sum([self.get_message_length(m) for m in messages[message_index+1:-1]])

        # get lengths of current message peices
        template_tokens = self.tokenizer(messages[message_index].get('prompt_template'), truncation=False, return_length=True, add_special_tokens=False)
        options_tokens = self.tokenizer(messages[message_index].get('options'), truncation=False, return_length=True, add_special_tokens=False) # this will be a list
        problem_tokens = self.tokenizer(messages[message_index].get('problem_activity'), truncation=False, return_length=True, add_special_tokens=False)

        total_options_length = np.sum(options_tokens['length'])

        current_length = template_tokens['length'][0] + total_options_length + problem_tokens['length'][0] + existing_message_lengths

        # print(current_length, self.max_length, self.buffer)
        if current_length >= (self.max_length - self.buffer):
            # split labels and truncate
            # truncated_messages = self._truncate_message_tokens(problem_tokens, options_tokens['input_ids'], messages[0].get('prompt_template'), template_tokens)

            # just truncate problem text for now
            max_problem_text_length = self.max_length - self.buffer - template_tokens['length'][0] - total_options_length - existing_message_lengths
            if max_problem_text_length < 0:
                raise ValueError("The prompt template and options are too long to fit in the max_length. Length of options: {}, template: {} and existing messages: {}".format(total_options_length, template_tokens['length'][0], existing_message_lengths))

            # print(max_problem_text_length)
            # print(problem_tokens)
            # truncate the problem text


            problem_tokens['input_ids'] = problem_tokens['input_ids'][:int(max_problem_text_length)]
            problem_tokens['length'] = max_problem_text_length

            # print(problem_tokens['length'], np.sum(options_tokens['length']), template_tokens['length'][0])

        messages[message_index]['problem_activity'] = self.tokenizer.decode(problem_tokens['input_ids'])

        return messages


    # taken for lucy clean_output
    def verify_response(self, response: str, response_format: str):

        res = response.choices[0].message.content

        if not res: return None, 'No response'
        res = res.strip()
        if response_format == 'comma_list':
            if res.startswith('none') or res.startswith('None'):
                return [], None
            match = re.match(r'^[A-Z,\s]+\b', res)
            if not match:
                return  None, 'Could not parse response into options'
            options = sorted([i.strip() for i in match.group().split(', ') if i.strip() in self.options])
        elif response_format == 'comma_list_last':
            lines = res.split('\n')
            if not lines[-1].startswith('Answer:'):
                return None, 'Could not parse response into options'
            answer = lines[-1].replace('Answer:', '').strip()
            if answer.lower() == 'none':
                return [], None
            match = re.match(r'^[A-Z,\s]+\b', answer)
            if not match:
                return None, 'Could not parse response into options'
            options = sorted([i.strip() for i in match.group().split(', ') if i.strip() in self.options])
        elif response_format == 'json':
            try:
                response_json = json.loads(res)
            except ValueError as e:
                return None, 'Could not parse json response'
            if 'answer' not in response_json: return None, 'Could not parse response into options'
            if response_json['answer'].lower() == 'none':
                return [], None
            match = re.match(r'^[A-Z,\s]+\b', response_json['answer'])
            if not match:
                return None, 'Could not parse response into options'
            options = sorted([i.strip() for i in match.group().split(', ') if i.strip() in self.options])
        else:
            return None, 'Could not parse response format: {}'.format(response_format)
        return options, None






    # def _truncate_message_tokens(self, problem_tokens, options_tokens_inputs, prompt_template, template_tokens):
    #     '''
    #     TODO: Not using this funcation right now because just truncating the problem text
    #     Recursively split a list of tokenized labels until the the problem and labels fit within the
    #     context window, or truncate the problem text to max size to fit 3 labels
    #     Returns a list of new, truncated prompts, where each list is the problem + max number of labels to fit into context window
    #     Currently returns the prompts in natural language (decoded), this takes more time but will be useful for checking
    #     the output is well-formed. Takes options as input ids because will need to continue to split them
    #     returns list of same form as messages above, but with parts truncated and split as needed
    #     '''
    #     if template_tokens['length'][0] + problem_tokens['length'][0] + np.sum([len(x) for x in options_tokens_inputs]) <= (self.max_length - self.buffer):
    #         # Base case: No need to truncate!
    #         print('no need to truncate')
    #         print(self.tokenizer.batch_decode(options_tokens_inputs))
    #         print(self.tokenizer.decode(problem_tokens['input_ids']))
    #         return [{'role':'user', 'prompt_template':prompt_template, 'options':self.tokenizer.batch_decode(options_tokens_inputs), 'problem_activity': self.tokenizer.decode(problem_tokens['input_ids'])}]

    #     if len(options_tokens_inputs) <= 3:
    #         print('options less than 3')
    #         # other base case: at the min number of labels (needs to be 3 since risk of splitting to 1 otherwise)
    #         # truncate problem text -- never cut options tokens
    #         max_problem_text_length = self.max_length - self.buffer - np.sum([len(x) for x in options_tokens_inputs]) - template_tokens['length'][0]
    #         if max_problem_text_length < 0:
    #             raise ValueError("The prompt template and options are too long to fit in the max_length.")

    #         print(max_problem_text_length, problem_tokens['length'][0], np.sum([len(x) for x in options_tokens_inputs]),  template_tokens['length'][0])
    #         # because the problem text are already tokens, just truncate the tokens - this should work since adding no special tokens

    #         problem_tokens['input_ids'] = problem_tokens['input_ids'][:max_problem_text_length]
    #         problem_tokens['length'] = max_problem_text_length
    #         print(options_tokens_inputs)
    #         return [{'role':'user', 'prompt_template':prompt_template, 'options':self.tokenizer.batch_decode(options_tokens_inputs), 'problem_activity': self.tokenizer.decode(problem_tokens['input_ids'])}]

    #     print('splitting...', len(options_tokens_inputs))
    #     # split label list in half and recurse
    #     half_labels = len(options_tokens_inputs)//2


    #     return self._truncate_message_tokens(problem_tokens, options_tokens_inputs[:half_labels], prompt_template, template_tokens) + self._truncate_message_tokens(problem_tokens, options_tokens_inputs[half_labels:], prompt_template, template_tokens)








