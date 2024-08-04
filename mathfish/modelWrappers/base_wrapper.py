'''
Parent class for wrappers of models that handle 
truncation, and retries, and different models

Author: Tal August 
Email:  tala@allenai.org
'''
from mathfish.utils import *
from collections import defaultdict, Counter
from mathfish.modelWrappers.predictionUtils import load_hf_tokenizer, load_hf_lm_and_tokenizer

import sys
import os
import openai
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
) 

load_dotenv()


class BaseModelWrapper:
    def __init__(self, model_name_or_path, tokenizer_name_or_path, openai_client=False, max_length=None, buffer=10, truncate=True):

        assert model_name_or_path or openai_client, "Must provide either a model name or path or an OpenAI client name."


        if not openai_client:
            self.model, self.tokenizer = load_hf_lm_and_tokenizer(
                    model_name_or_path=model_name_or_path, 
                    tokenizer_name_or_path=tokenizer_name_or_path
                )
        else:
            # check if this is a gpt model and point to openai
            if "gpt" in model_name_or_path:
                self.openai_client = openai.OpenAI(
                    api_key=os.environ.get("OPENAI_API_KEY"),
                    base_url="https://api.openai.com/v1",
                )
                self.tokenizer = load_hf_tokenizer('stabilityai/stablelm-2-1_6b', token=os.environ.get("HF_TOKEN")) 

            else:
                self.openai_client = openai.OpenAI(
                    api_key=os.environ.get("TOGETHER_API_KEY"),
                    base_url="https://api.together.xyz/v1",
                )
                self.tokenizer = load_hf_tokenizer(tokenizer_name_or_path, token=os.environ.get("HF_TOKEN"))

            self.model = model_name_or_path # will just feed this into the client

        if max_length:
            self.max_length = max_length       
        elif self.tokenizer.model_max_length < 1e30:            
            self.max_length = self.tokenizer.model_max_length
        else:
            print("No max length provided, and tokenizer does not have a max length. Setting to 4097.")
            self.max_length = 4097

        self.truncate = truncate
        self.buffer = buffer

        self.logs = []

    def log(self, log_event, output=False):
        self.logs.append(log_event)
        if output:
            print(log_event, file=sys.stderr)

    def get_logs(self):
        return self.logs
    

    def reset_logs(self):
        self.logs = []


    def format_prompt(self, message):
        '''
        Dummy
        '''
        return message 

    def format_messages(self , messages):
        assert all(message["role"] in ["user", "assistant"] for message in messages), \
                    "Each message should have a `role` of either 'user' or 'assistant'."
        if self.truncate:
            messages = self.truncate_messages(messages)
        
        formatted_messages = [self.format_prompt(message) for message in messages]

        return formatted_messages


    def truncate_messages(self, messages):
        '''
        Dummy, just truncate content of messages by max length. Asssumes messages are of form: 
        {
            role: 'user',
            content: 'content'
        }
        '''
        assert len(messages) <= 2, "BaseWrapper only supports one pair of messages at a time. Please provide maxmimum of two messages, the first being the user message."
        assert messages[0].get('role') == "user", "first message should be user message."

        content = self.tokenizer(messages[0].get('content'), truncation=False, return_length=True, add_special_tokens=False)
        
        # truncate
        max_length = self.max_length - self.buffer
        content['input_ids'] = content['input_ids'][:max_length]
        content['length'] = max_length
        
        messages[0]['content'] = self.tokenizer.decode(content['input_ids'])

        return messages

    def format_chat_template(self, message, chat_template="tulu2"):
        '''
        Formats messages with the specified chat template, 
        assumes the message as already been formatted correctly
        '''
        template = get_template(chat_template)
        return template.format(instruction=message)

    def run_model_single(self, instance):
        truncated_messages = self.truncate_messages(instance)
        
        prompts = create_prompt_with_tulu_chat_format(self.truncate_messages, self.tokenizer)
        return generate_completions(
                model=self.model,
                tokenizer=self.tokenizer,
                prompts=[prompts],
                batch_size=1,
        )

    def run_model_batch(self, instances, batch_size=10):
        prompts = []
        for i in instances:
            formatted_messages = self.format_messages(i)
            # TODO: THIS ONLY TAKES THE FIRST MESSAGE
            prompts.append(self.format_chat_template(formatted_messages[0], self.tokenizer))
        
        results = generate_completions(
                model=self.model,
                temperature=0.2, # set based on convo with Luca
                tokenizer=self.tokenizer,
                prompts=prompts,
                batch_size=batch_size,
        )
        
        return results 
    
    def verify_response(self, response, response_format):
        '''
        Dummy
        '''
        raise NotImplementedError
        # return response.choices[0].message.content, None
        
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def call_api(self, messages, temp):
        if temp is None:
            return self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
        return self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temp,
            )
    
    def retry_instance(self, retry_num, formatted_messages, response, reason, temp):
        # retry is for parse issues, network issues handled by tenacity
        self.log({'event': 'retry', 'count':retry_num, 'reason': reason, 'messages': formatted_messages, 'initial-response': response})
        return self.call_api(formatted_messages, temp)
    
    
    def call_model_api(self, instance, retry, num_retries, temp):
        '''
        Run the model by calling the together API. Steps are:
        1) Format the messages correctly into the the prompt format getting used 
        2) call API for each instance
        3) retry (if retry=True) if we can't parse the instance (e.g. if there is no yes/no)
        '''
        assert self.openai_client is not None, "OpenAI client must be set to call the model API"
        assert not (retry and num_retries <= 0), "Number of retries must be greater than 0 if you want to enable retries"
        
        # don't need to format into chat prompt since passing to openai API
        formatted_messages = self.format_messages(instance['messages'])
        assert formatted_messages[0]['role'] == 'user', "First message should be user message."

        # get the correct response format from the instance, if it exists
        response_format = instance['messages'][-1].get('response_format', None)

        # call the model
        num_calls = 0
        try:
            chat_completion = self.call_api(formatted_messages[:-1], temp)
            num_calls += 1
        except Exception as e:
            # if tenacity times out, no formattting retries, even if enabled
            self.log({'event': 'model-api-call', 'error': str(e), 'messages': formatted_messages})
            print(e)
            return {'dataset':instance['dataset'], 'id':str(instance['id']), 'model': self.model, 
                 'message_content':formatted_messages[0]['content'], 
                 'prompt-length':None,
                 'completion-length':None,
                 'model-response':None,
                 'prelim-response':None,
                 'num_calls': num_calls}

        prelim_response = chat_completion.choices[0].message.content
        verified_response = None
        if retry:
            for i in range(num_retries):
                # prelim_response = chat_completion.choices[0].message.content
                verified_response, reason = self.verify_response(chat_completion, response_format)
                if verified_response:
                    break
                chat_completion = self.retry_instance(i, formatted_messages[:-1], prelim_response, reason, temp)
                num_calls += 1
        else:
            verified_response, reason = self.verify_response(chat_completion)

        self.log({'event': 'model-api-call', 'response': prelim_response})

        logs = self.get_logs()
        self.reset_logs()

        return  {'dataset':instance['dataset'], 'id':str(instance['id']), 'model': chat_completion.model, 
                 'message_content':formatted_messages, 
                 'prompt-length':chat_completion.usage.prompt_tokens,
                 'completion-length':chat_completion.usage.completion_tokens,
                 'model-response':prelim_response,
                 'prelim-response':verified_response,
                 'num_calls': num_calls}




#  '''
#     Recursively split a list of tokenized labels until the the problem and labels fit within the 
#     context window, or truncate the problem text to max size to fit 3 labels
#     Returns a list of new, truncated prompts, where each list is the problem + max number of labels to fit into context window
#     Currently returns the prompts in natural language (decoded), this takes more time but will be useful for checking 
#     the output is well-formed
#     '''
#     def _truncate_tokens(self, problem_tokens, labels_tokens):

#         if problem_tokens.length + np.sum([l.length for l in labels_tokens]) < self.max_length:
#             # No need to truncate!
#             return [self.tokenizer.batch_decode([problem_tokens] + labels_tokens)]

#          if len(labels_tokens) <= 3:
#             # min number of labels (risk of having one otherwise) so truncate problem text -- never cut label tokens
#             max_problem_text_length = self.max_length - np.sum([len(l) for l in labels_tokens])
#             return [self.tokenizer.batch_decode([problem_tokens[:max_problem_text_length]] + labels_tokens)]
        
#         # split label list in half and recurse
#         half_labels = len(labels_tokens)//2
        
#         return self._truncate_tokens(problem_tokens, labels_tokens[:half_labels]) + self._truncate_tokens(problem_tokens, labels_tokens[half_labels:])


#  def truncate_problem_text(self, problem_text: str, length: int):
#         '''
#         Dummy
#         '''
#         return problem_text[:length]

#     def truncate_labels(self, labels: str):
#          '''
#         Dummy
#         '''
#         return ', '.join(labels)


        # if len(tokenized_prompt.length) < self.max_length:
        #     # No need to truncate!
        #     return [tokenized_prompt]
        
        
        # problem, labels = self.seperate_prompt(prompt)

        # problem_tokens = self.tokenizer(problem, truncation=False, return_length=True)
        # labels_tokens = self.tokenizer(labels, truncation=False, return_length=True)

        # return self._truncate_tokens(problem_tokens, labels_tokens)






    

 # '''
    # seperate the prompt into problem and labels, simple for now, assumes prompt is in form 
    # Instructions \n  \n Problem/activity: {problem_activity} \n Description: {standard_description}"
    # '''
    # def seperate_prompt(self, prompt: str):

    # pattern = r"^(.*?)\s*\\n\s*Problem/activity:\s*{(.*?)}\s*\\n\s*Description:\s*{(.*?)}\"?$"
    # matches = re.match(pattern, prompt, re.DOTALL)

    # if matches:
    #     instructions = matches.group(1).strip()
    #     problem_activity = matches.group(2).strip()
    #     description = matches.group(3).strip()

    #     return problem_activity, [description]

    # else:
    #     raise ("No match found.")
