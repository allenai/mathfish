"""
Tests base_wrapper.py in wrapppers

Author: Tal August (@tala), 
Email:  tala@allenai.org
"""


import unittest
from unittest.mock import patch

from mathfish.modelWrappers.base_wrapper import BaseModelWrapper


class TestBaseModelWrapper(unittest.TestCase):

    def setUp(self):
        self.wrapper = BaseModelWrapper(model_name_or_path='allenai/tulu-2-7b', tokenizer_name_or_path='allenai/tulu-2-7b', max_length=100, buffer=10)

    def test_format_prompt(self):
        message = {"role": "user", "content": "You are a math expert reviewing K-12 curricular materials. You will be shown a problem or activity obtained from school curriculum and a description of math content. Your task is to assess whether the problem or activity aligns with the provided description. Answer 'yes' if it does align, and 'no' it does not.\n Problem/activity:    Stage 1: Nearest Ten or Hundred\n     Required Preparation\n     Materials to Gather\n       Colored pencils, crayons, or markers\n       Number cards 0\u201310\n       Paper clips\n     Materials to Copy\n       Blackline Masters\n      Tic Tac Round Stage 1 Gameboard\n      Tic Tac Round Stage 1 Spinner\n   <br>\n    Narrative\n     Students remove the cards that show 10 before they start. Then they choose three\u00a0number cards and make a three-digit number. They spin the spinner to get a place value to round to. Students write their number in any space on the board, each partner using a different color. The first player to get three\u00a0in a row wins.\n \n Description: Relate the domain of a function to its graph and, where applicable, to the quantitative relationship it describes.  For example, if the function h(n) gives the number of person-hours it takes to assemble n engines in a factory, then the positive integers would be an appropriate domain for the function."}
        formatted_prompt = self.wrapper.format_prompt(message)
        self.assertEqual(formatted_prompt, message)

    def test_format_messages(self):
        messages = [{"role": "user", "content": "You are a math expert reviewing K-12 curricular materials. You will be shown a problem or activity obtained from school curriculum and a description of math content. Your task is to assess whether the problem or activity aligns with the provided description. Answer 'yes' if it does align, and 'no' it does not.\n Problem/activity:    Stage 1: Nearest Ten or Hundred\n     Required Preparation\n     Materials to Gather\n       Colored pencils, crayons, or markers\n       Number cards 0\u201310\n       Paper clips\n     Materials to Copy\n       Blackline Masters\n      Tic Tac Round Stage 1 Gameboard\n      Tic Tac Round Stage 1 Spinner\n   <br>\n    Narrative\n     Students remove the cards that show 10 before they start. Then they choose three\u00a0number cards and make a three-digit number. They spin the spinner to get a place value to round to. Students write their number in any space on the board, each partner using a different color. The first player to get three\u00a0in a row wins.\n \n Description: Relate the domain of a function to its graph and, where applicable, to the quantitative relationship it describes.  For example, if the function h(n) gives the number of person-hours it takes to assemble n engines in a factory, then the positive integers would be an appropriate domain for the function."}]
        formatted_messages = self.wrapper.format_messages(messages)
        self.assertEqual(formatted_messages, messages)

    def test_truncate_messages(self):
        long_messages = [{"role": "user", "content":"You are a math expert reviewing K-12 curricular materials. You will be shown a problem or activity obtained from school curriculum and a description of math content. Your task is to assess whether the problem or activity aligns with the provided description. Answer 'yes' if it does align, and 'no' it does not.\n Problem/activity:    Stage 1: Nearest Ten or Hundred\n     Required Preparation\n     Materials to Gather\n       Colored pencils, crayons, or markers\n       Number cards 0\u201310\n       Paper clips\n     Materials to Copy\n       Blackline Masters\n      Tic Tac Round Stage 1 Gameboard\n      Tic Tac Round Stage 1 Spinner\n   <br>\n    Narrative\n     Students remove the cards that show 10 before they start. Then they choose three\u00a0number cards and make a three-digit number. They spin the spinner to get a place value to round to. Students write their number in any space on the board, each partner using a different color. The first player to get three\u00a0in a row wins.\n \n Description: Relate the domain of a function to its graph and, where applicable, to the quantitative relationship it describes.  For example, if the function h(n) gives the number of person-hours it takes to assemble n engines in a factory, then the positive integers would be an appropriate domain for the function. You are a math expert reviewing K-12 curricular materials. You will be shown a problem or activity obtained from school curriculum and a description of math content. Your task is to assess whether the problem or activity aligns with the provided description. Answer 'yes' if it does align, and 'no' it does not.\n Problem/activity:    Stage 1: Nearest Ten or Hundred\n     Required Preparation\n     Materials to Gather\n       Colored pencils, crayons, or markers\n       Number cards 0\u201310\n       Paper clips\n     Materials to Copy\n       Blackline Masters\n      Tic Tac Round Stage 1 Gameboard\n      Tic Tac Round Stage 1 Spinner\n   <br>\n    Narrative\n     Students remove the cards that show 10 before they start. Then they choose three\u00a0number cards and make a three-digit number. They spin the spinner to get a place value to round to. Students write their number in any space on the board, each partner using a different color. The first player to get three\u00a0in a row wins.\n \n Description: Compare two numbers between 1 and 10 presented as written numerals."}]
        truncated_messages = self.wrapper.truncate_messages(long_messages)
        tokenized_content = self.wrapper.tokenizer(truncated_messages[0]['content'], return_tensors='pt', return_length=True, add_special_tokens=False)
        self.assertEqual(tokenized_content['length'][0], 90)

    def test_tokenize_decode_message(self):
        messages = [{"role": "user", "content": "You are a math expert reviewing K-12 curricular materials. You will be shown a problem or activity obtained from school curriculum and a description of math content. Your task is to assess whether the problem or activity aligns with the provided description. Answer 'yes' if it does align, and 'no' it does not.\n Problem/activity:    Stage 1: Nearest Ten or Hundred\n     Required Preparation\n     Materials to Gather\n       Colored pencils, crayons, or markers\n       Number cards 0\u201310\n       Paper clips\n     Materials to Copy\n       Blackline Masters\n      Tic Tac Round Stage 1 Gameboard\n      Tic Tac Round Stage 1 Spinner\n   <br>\n    Narrative\n     Students remove the cards that show 10 before they start. Then they choose three\u00a0number cards and make a three-digit number. They spin the spinner to get a place value to round to. Students write their number in any space on the board, each partner using a different color. The first player to get three\u00a0in a row wins.\n \n Description: Relate the domain of a function to its graph and, where applicable, to the quantitative relationship it describes.  For example, if the function h(n) gives the number of person-hours it takes to assemble n engines in a factory, then the positive integers would be an appropriate domain for the function."}]
        tokenized_content = self.wrapper.tokenizer(messages[0]['content'], return_tensors='pt', return_length=True, add_special_tokens=False)
        decoded_content = self.wrapper.tokenizer.decode(tokenized_content['input_ids'][0])
        self.assertEqual(decoded_content, messages[0]['content'])

    @patch('edreports.modelWrappers.base_wrapper.generate_completions')
    @patch('edreports.modelWrappers.base_wrapper.create_prompt_with_tulu_chat_format')
    @patch('edreports.modelWrappers.base_wrapper.load_hf_lm_and_tokenizer')
    def test_run_model(self, mock_load_hf_lm_and_tokenizer, mock_create_prompt_with_tulu_chat_format, mock_generate_completions):
        messages = [{"role": "user", "content": "You are a math expert reviewing K-12 curricular materials. You will be shown a problem or activity obtained from school curriculum and a description of math content. Your task is to assess whether the problem or activity aligns with the provided description. Answer 'yes' if it does align, and 'no' it does not.\n Problem/activity:    Stage 1: Nearest Ten or Hundred\n     Required Preparation\n     Materials to Gather\n       Colored pencils, crayons, or markers\n       Number cards 0\u201310\n       Paper clips\n     Materials to Copy\n       Blackline Masters\n      Tic Tac Round Stage 1 Gameboard\n      Tic Tac Round Stage 1 Spinner\n   <br>\n    Narrative\n     Students remove the cards that show 10 before they start. Then they choose three\u00a0number cards and make a three-digit number. They spin the spinner to get a place value to round to. Students write their number in any space on the board, each partner using a different color. The first player to get three\u00a0in a row wins.\n \n Description: Relate the domain of a function to its graph and, where applicable, to the quantitative relationship it describes.  For example, if the function h(n) gives the number of person-hours it takes to assemble n engines in a factory, then the positive integers would be an appropriate domain for the function."}]
        result = self.wrapper.run_model_single(messages)
        print(result)

if __name__ == '__main__':
    unittest.main()


