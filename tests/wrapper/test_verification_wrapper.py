
"""
Tests verification_wrapper.py in wrapppers

Author: Tal August (@tala)
Email:  tala@allenai.org
"""
import unittest
from mathfish.modelWrappers.verification_wrapper import VerificationWrapper
import re

class TestVerificationWrapper(unittest.TestCase):
    def setUp(self):
        self.wrapper = VerificationWrapper(model_name_or_path='allenai/tulu-2-7b', tokenizer_name_or_path='allenai/tulu-2-7b', max_length=1028, buffer=10, truncate=False)

    def test_format_prompt(self):
        message = {
            'role': 'user',
            'prompt_template': "You are a math expert reviewing K-12 curricular materials. You will be shown a problem or activity obtained from school curriculum and a description of mathematical concepts and skills. Your task is to determine whether the problem or activity can enable students to learn the full intent of the concepts and skills outlined in the provided description. Answer 'yes' if it does, and 'no' if it does not. \n Problem/activity: {problem_activity} \n Description: {standard_description}",
            'standard_description': "Use data from a randomized experiment to compare two treatments; use simulations to decide if differences between parameters are significant.",
            'problem_activity': "Stage 1: Nearest Ten or Hundred\n     Required Preparation\n     Materials to Gather\n       Colored pencils, crayons, or markers\n       Number cards 0\u201310\n       Paper clips\n     Materials to Copy\n       Blackline Masters\n      Tic Tac Round Stage 1 Gameboard\n      Tic Tac Round Stage 1 Spinner\n   <br>\n    Narrative\n     Students remove the cards that show 10 before they start. Then they choose three\u00a0number cards and make a three-digit number. They spin the spinner to get a place value to round to. Students write their number in any space on the board, each partner using a different color. The first player to get three\u00a0in a row wins.\n \n Description: Write an inequality of the form x &gt; c or x &lt; c to represent a constraint or condition in a real-world or mathematical problem. Recognize that inequalities of the form x &gt; c or x &lt; c have infinitely many solutions; represent solutions of such inequalities on number line diagrams."
        }
        formatted_prompt = self.wrapper.format_prompt(message)
        expected_prompt = {'role':'user', 'content':message['prompt_template'].format(standard_description=message['standard_description'], problem_activity=message['problem_activity'])} 
        
        try: 
            self.assertEqual(formatted_prompt, expected_prompt)
        except AssertionError as e:
            print(formatted_prompt)
            print(expected_prompt)
            raise e


    # def seperate_prompt(self, prompt: str):

    #     # pattern = r"^(.*?)\s*\\n\s*Problem/activity:\s*{(.*?)}\s*\\n\s*Description:\s*{(.*?)}\"?$"
    #     pattern = r"<\|user\|>(.*?)Problem/activity: (.*?)Description: (.*?)<\|assistant\|>"
    #     matches = re.search(pattern, prompt, re.DOTALL)

    #     if matches:
    #         instructions = matches.group(1).strip()
    #         problem_activity = matches.group(2).strip()
    #         description = matches.group(3).strip()

    #         return instructions, problem_activity, description

    #     else:
    #         print(prompt)
    #         raise BaseException("No match found.")



    def test_truncate_messages(self):
        '''
        Want to check that only the problem_activity is truncated if the prompt template and description are too long
        '''
        message = {
            'role': 'user',
            'prompt_template': "You are a math expert reviewing K-12 curricular materials. You will be shown a problem or activity obtained from school curriculum and a description of mathematical concepts and skills. Your task is to determine whether the problem or activity can enable students to learn the full intent of the concepts and skills outlined in the provided description. Answer 'yes' if it does, and 'no' if it does not. \n Problem/activity: {problem_activity} \n Description: {standard_description}",
            'standard_description': "Use data from a randomized experiment to compare two treatments; use simulations to decide if differences between parameters are significant.",
            'problem_activity': "Stage 1: Nearest Ten or Hundred\n     Required Preparation\n     Materials to Gather\n       Colored pencils, crayons, or markers\n       Number cards 0\u201310\n       Paper clips\n     Materials to Copy\n       Blackline Masters\n      Tic Tac Round Stage 1 Gameboard\n      Tic Tac Round Stage 1 Spinner\n   <br>\n    Narrative\n     Students remove the cards that show 10 before they start. Then they choose three\u00a0number cards and make a three-digit number. They spin the spinner to get a place value to round to. Students write their number in any space on the board, each partner using a different color. The first player to get three\u00a0in a row wins.\n \n Description: Write an inequality of the form x &gt; c or x &lt; c to represent a constraint or condition in a real-world or mathematical problem. Recognize that inequalities of the form x &gt; c or x &lt; c have infinitely many solutions; represent solutions of such inequalities on number line diagrams."
        }
        self.wrapper = VerificationWrapper(model_name_or_path='allenai/tulu-2-7b', tokenizer_name_or_path='allenai/tulu-2-7b', max_length=256, buffer=10, truncate=False)

        truncated_messages = self.wrapper.truncate_messages([message])

        self.assertEqual(truncated_messages[0]['prompt_template'], message['prompt_template'])
        self.assertEqual(truncated_messages[0]['standard_description'], message['standard_description'])

        tokenized_content = self.wrapper.tokenizer(truncated_messages[0]['problem_activity'], return_tensors='pt', return_length=True, add_special_tokens=False)
        self.assertLess(tokenized_content['length'][0], 256) 
  

def test_truncate_short(self):
    long_message = {
            'role': 'user',
            'prompt_template': "You are a math expert reviewing K-12 curricular materials. You will be shown a problem or activity obtained from school curriculum and a description of mathematical concepts and skills. Your task is to determine whether the problem or activity can enable students to learn the full intent of the concepts and skills outlined in the provided description. Answer 'yes' if it does, and 'no' if it does not. \n Problem/activity: {problem_activity} \n Description: {standard_description}",
            'standard_description': "Use data from a randomized experiment to compare two treatments; use simulations to decide if differences between parameters are significant.",
            'problem_activity': "Stage 1: Nearest Ten or Hundred\n     Required Preparation\n     Materials to Gather\n       Colored pencils, crayons, or markers\n       Number cards 0\u201310\n       Paper clips\n     Materials to Copy\n       Blackline Masters\n      Tic Tac Round Stage 1 Gameboard\n      Tic Tac Round Stage 1 Spinner\n   <br>\n    Narrative\n     Students remove the cards that show 10 before they start. Then they choose three\u00a0number cards and make a three-digit number. They spin the spinner to get a place value to round to. Students write their number in any space on the board, each partner using a different color. The first player to get three\u00a0in a row wins.\n \n Description: Write an inequality of the form x &gt; c or x &lt; c to represent a constraint or condition in a real-world or mathematical problem. Recognize that inequalities of the form x &gt; c or x &lt; c have infinitely many solutions; represent solutions of such inequalities on number line diagrams. Stage 1: Nearest Ten or Hundred\n     Required Preparation\n     Materials to Gather\n       Colored pencils, crayons, or markers\n       Number cards 0\u201310\n       Paper clips\n     Materials to Copy\n       Blackline Masters\n      Tic Tac Round Stage 1 Gameboard\n      Tic Tac Round Stage 1 Spinner\n   <br>\n    Narrative\n     Students remove the cards that show 10 before they start. Then they choose three\u00a0number cards and make a three-digit number. They spin the spinner to get a place value to round to. Students write their number in any space on the board, each partner using a different color. The first player to get three\u00a0in a row wins.\n \n Description: Write an inequality of the form x &gt; c or x &lt; c to represent a constraint or condition in a real-world or mathematical problem. Recognize that inequalities of the form x &gt; c or x &lt; c have infinitely many solutions; represent solutions of such inequalities on number line diagrams. Stage 1: Nearest Ten or Hundred\n     Required Preparation\n     Materials to Gather\n       Colored pencils, crayons, or markers\n       Number cards 0\u201310\n       Paper clips\n     Materials to Copy\n       Blackline Masters\n      Tic Tac Round Stage 1 Gameboard\n      Tic Tac Round Stage 1 Spinner\n   <br>\n    Narrative\n     Students remove the cards that show 10 before they start. Then they choose three\u00a0number cards and make a three-digit number. They spin the spinner to get a place value to round to. Students write their number in any space on the board, each partner using a different color. The first player to get three\u00a0in a row wins.\n \n Description: Write an inequality of the form x &gt; c or x &lt; c to represent a constraint or condition in a real-world or mathematical problem. Recognize that inequalities of the form x &gt; c or x &lt; c have infinitely many solutions; represent solutions of such inequalities on number line diagrams. Stage 1: Nearest Ten or Hundred\n     Required Preparation\n     Materials to Gather\n       Colored pencils, crayons, or markers\n       Number cards 0\u201310\n       Paper clips\n     Materials to Copy\n       Blackline Masters\n      Tic Tac Round Stage 1 Gameboard\n      Tic Tac Round Stage 1 Spinner\n   <br>\n    Narrative\n     Students remove the cards that show 10 before they start. Then they choose three\u00a0number cards and make a three-digit number. They spin the spinner to get a place value to round to. Students write their number in any space on the board, each partner using a different color. The first player to get three\u00a0in a row wins.\n \n Description: Write an inequality of the form x &gt; c or x &lt; c to represent a constraint or condition in a real-world or mathematical problem. Recognize that inequalities of the form x &gt; c or x &lt; c have infinitely many solutions; represent solutions of such inequalities on number line diagrams. Stage 1: Nearest Ten or Hundred\n     Required Preparation\n     Materials to Gather\n       Colored pencils, crayons, or markers\n       Number cards 0\u201310\n       Paper clips\n     Materials to Copy\n       Blackline Masters\n      Tic Tac Round Stage 1 Gameboard\n      Tic Tac Round Stage 1 Spinner\n   <br>\n    Narrative\n     Students remove the cards that show 10 before they start. Then they choose three\u00a0number cards and make a three-digit number. They spin the spinner to get a place value to round to. Students write their number in any space on the board, each partner using a different color. The first player to get three\u00a0in a row wins.\n \n Description: Write an inequality of the form x &gt; c or x &lt; c to represent a constraint or condition in a real-world or mathematical problem. Recognize that inequalities of the form x &gt; c or x &lt; c have infinitely many solutions; represent solutions of such inequalities on number line diagrams. Stage 1: Nearest Ten or Hundred\n     Required Preparation\n     Materials to Gather\n       Colored pencils, crayons, or markers\n       Number cards 0\u201310\n       Paper clips\n     Materials to Copy\n       Blackline Masters\n      Tic Tac Round Stage 1 Gameboard\n      Tic Tac Round Stage 1 Spinner\n   <br>\n    Narrative\n     Students remove the cards that show 10 before they start. Then they choose three\u00a0number cards and make a three-digit number. They spin the spinner to get a place value to round to. Students write their number in any space on the board, each partner using a different color. The first player to get three\u00a0in a row wins.\n \n Description: Write an inequality of the form x &gt; c or x &lt; c to represent a constraint or condition in a real-world or mathematical problem. Recognize that inequalities of the form x &gt; c or x &lt; c have infinitely many solutions; represent solutions of such inequalities on number line diagrams. Stage 1: Nearest Ten or Hundred\n     Required Preparation\n     Materials to Gather\n       Colored pencils, crayons, or markers\n       Number cards 0\u201310\n       Paper clips\n     Materials to Copy\n       Blackline Masters\n      Tic Tac Round Stage 1 Gameboard\n      Tic Tac Round Stage 1 Spinner\n   <br>\n    Narrative\n     Students remove the cards that show 10 before they start. Then they choose three\u00a0number cards and make a three-digit number. They spin the spinner to get a place value to round to. Students write their number in any space on the board, each partner using a different color. The first player to get three\u00a0in a row wins.\n \n Description: Write an inequality of the form x &gt; c or x &lt; c to represent a constraint or condition in a real-world or mathematical problem. Recognize that inequalities of the form x &gt; c or x &lt; c have infinitely many solutions; represent solutions of such inequalities on number line diagrams. Stage 1: Nearest Ten or Hundred\n     Required Preparation\n     Materials to Gather\n       Colored pencils, crayons, or markers\n       Number cards 0\u201310\n       Paper clips\n     Materials to Copy\n       Blackline Masters\n      Tic Tac Round Stage 1 Gameboard\n      Tic Tac Round Stage 1 Spinner\n   <br>\n    Narrative\n     Students remove the cards that show 10 before they start. Then they choose three\u00a0number cards and make a three-digit number. They spin the spinner to get a place value to round to. Students write their number in any space on the board, each partner using a different color. The first player to get three\u00a0in a row wins.\n \n Description: Write an inequality of the form x &gt; c or x &lt; c to represent a constraint or condition in a real-world or mathematical problem. Recognize that inequalities of the form x &gt; c or x &lt; c have infinitely many solutions; represent solutions of such inequalities on number line diagrams."
        }
    wrapper = VerificationWrapper(openai_client=True, model_name_or_path="mistralai/Mixtral-8x7B-Instruct-v0.1", tokenizer_name_or_path="mistralai/Mixtral-8x7B-Instruct-v0.1", truncate=True)
    short_wrapper = VerificationWrapper(openai_client=True, model_name_or_path="mistralai/Mixtral-8x7B-Instruct-v0.1", tokenizer_name_or_path="mistralai/Mixtral-8x7B-Instruct-v0.1", truncate=True, max_length=100)

    truncated_messages = wrapper.truncate_messages([long_message])
    short_truncated_messages = short_wrapper.truncate_messages([long_message])

    self.assertEqual(truncated_messages[0]['prompt_template'], long_message['prompt_template'])
    self.assertEqual(truncated_messages[0]['standard_description'], long_message['standard_description'])

    self.assertEqual(short_truncated_messages[0]['prompt_template'], long_message['prompt_template'])
    self.assertEqual(short_truncated_messages[0]['standard_description'], long_message['standard_description'])


    tokenized_content = wrapper.tokenizer(truncated_messages[0]['problem_activity'], return_tensors='pt', return_length=True, add_special_tokens=False)
    short_tokenized_content = short_wrapper.tokenizer(short_truncated_messages[0]['problem_activity'], return_tensors='pt', return_length=True, add_special_tokens=False)

    self.assertLess(tokenized_content['length'][0], 4097) 
    self.assertLess(short_tokenized_content['length'][0], 100) 


        
if __name__ == '__main__':
    unittest.main()

