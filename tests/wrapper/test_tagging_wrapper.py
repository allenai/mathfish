
import unittest
from mathfish.modelWrappers.tagging_wrapper import TaggingWrapper
import string

class TestTaggingWrapper(unittest.TestCase):
    def setUp(self):
        self.wrapper = TaggingWrapper(model_name_or_path='allenai/tulu-2-7b', tokenizer_name_or_path='allenai/tulu-2-7b', max_length=512, buffer=20, truncate=True)

    def test_format_prompt(self):
        message = {
            "role": "user", 
            "prompt_template": "You are a math expert reviewing K-12 curricular materials. You will be shown a problem or activity obtained from school curriculum and a description of mathematical concepts and skills. A problem or activity aligns with a standard if it can enable students to learn the full intent of the concepts and skills outlined in the standard's description. Your task is to assign the following problem or activity to the standard/s it aligns with, and format your output as a comma-separated list of options e.g. A, B, C. Output \"none\" if none of the descriptions are relevant. DO NOT make up additional standards. \n\nProblem/activity:\n{problem_activity}\n\nHere are possible math standards:\n{options}",
            "options": ["Explain why addition and subtraction strategies work, using place value and the properties of operations. (Explanations may be supported by drawings or objects.)", "Solve word problems involving dollar bills, quarters, dimes, nickels, and pennies, using $ and \u00a2 symbols appropriately. Example: If you have 2 dimes and 3 pennies, how many cents do you have?", "Solve real-world and mathematical problems involving area, volume and surface area of two- and three-dimensional objects composed of triangles, quadrilaterals, polygons, cubes, and right prisms.", "Rewrite simple rational expressions in different forms; write a(x)/b(x) in the form q(x) + r(x)/b(x), where a(x), b(x), q(x), and r(x) are polynomials with the degree of r(x) less than the degree of b(x), using inspection, long division, or, for the more complicated examples, a computer algebra system.", "Measure volumes by counting unit cubes, using cubic cm, cubic in, cubic ft, and improvised units."],
            "problem_activity": "Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. We looked at addition and subtraction methods that can be used to make sense of and solve these problems. Explain to your partner why it works to use addition or subtraction to solve these problems.\u201d (You can use addition to add on to one number until you get to the total. The number you added on is the unknown number. You can use subtraction to start with the total and take away the other number you know. The number you have left is the unknown number.)\n",
        }
        formatted_prompt = self.wrapper.format_prompt(message)
        option_list = message.get('options')
        option_string = ''
        options = list(string.ascii_uppercase)
        for i in range(len(option_list)): 
            option_string += options[i] + '. ' + option_list[i] + '\n'

        expected_prompt = {'role':'user', 'content':message['prompt_template'].format(options=option_string, problem_activity=message['problem_activity'])} 
        
        try: 
            self.assertEqual(formatted_prompt, expected_prompt)
        except AssertionError as e:
            print(formatted_prompt)
            print(expected_prompt)
            raise e


    def test_truncate_messages(self):
        '''
        Want to check that labels are split first if prompt template and options are too long
        '''
        message = {
            "role": "user", 
            "prompt_template": "You are a math expert reviewing K-12 curricular materials. You will be shown a problem or activity obtained from school curriculum and a description of mathematical concepts and skills. A problem or activity aligns with a standard if it can enable students to learn the full intent of the concepts and skills outlined in the standard's description. Your task is to assign the following problem or activity to the standard/s it aligns with, and format your output as a comma-separated list of options e.g. A, B, C. Output \"none\" if none of the descriptions are relevant. DO NOT make up additional standards. \n\nProblem/activity:\n{problem_activity}\n\nHere are possible math standards:\n{options}",
            "options": ["Explain why addition and subtraction strategies work, using place value and the properties of operations. (Explanations may be supported by drawings or objects.)", "Solve word problems involving dollar bills, quarters, dimes, nickels, and pennies, using $ and \u00a2 symbols appropriately. Example: If you have 2 dimes and 3 pennies, how many cents do you have?", "Solve real-world and mathematical problems involving area, volume and surface area of two- and three-dimensional objects composed of triangles, quadrilaterals, polygons, cubes, and right prisms.", "Rewrite simple rational expressions in different forms; write a(x)/b(x) in the form q(x) + r(x)/b(x), where a(x), b(x), q(x), and r(x) are polynomials with the degree of r(x) less than the degree of b(x), using inspection, long division, or, for the more complicated examples, a computer algebra system.", "Measure volumes by counting unit cubes, using cubic cm, cubic in, cubic ft, and improvised units."],
            "problem_activity": "Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. We looked at addition and subtraction methods that can be used to make sense of and solve these problems. Explain to your partner why it works to use addition or subtraction to solve these problems.\u201d (You can use addition to add on to one number until you get to the total. The number you added on is the unknown number. You can use subtraction to start with the total and take away the other number you know. The number you have left is the unknown number.)\n",
        }

        truncated_messages = self.wrapper.truncate_messages([message])

        self.assertEqual(truncated_messages[0]['prompt_template'], message['prompt_template'])
        self.assertEqual(truncated_messages[0]['options'], message['options'])

        tokenized_content = self.wrapper.tokenizer(truncated_messages[0]['problem_activity'], return_tensors='pt', return_length=True, add_special_tokens=False)
        self.assertLess(tokenized_content['length'][0], 512) 


    def test_truncate_messages_short(self):
        '''
        Want to check that models don't truncate when not necessary
        '''
        message = {
            "role": "user", 
            "prompt_template": "You are a math expert reviewing K-12 curricular materials. You will be shown a problem or activity obtained from school curriculum and a description of mathematical concepts and skills. A problem or activity aligns with a standard if it can enable students to learn the full intent of the concepts and skills outlined in the standard's description. Your task is to assign the following problem or activity to the standard/s it aligns with, and format your output as a comma-separated list of options e.g. A, B, C. Output \"none\" if none of the descriptions are relevant. DO NOT make up additional standards. \n\nProblem/activity:\n{problem_activity}\n\nHere are possible math standards:\n{options}",
            "options": ["Explain why addition and subtraction strategies work, using place value and the properties of operations. (Explanations may be supported by drawings or objects.)", "Solve word problems involving dollar bills, quarters, dimes, nickels, and pennies, using $ and \u00a2 symbols appropriately. Example: If you have 2 dimes and 3 pennies, how many cents do you have?", "Solve real-world and mathematical problems involving area, volume and surface area of two- and three-dimensional objects composed of triangles, quadrilaterals, polygons, cubes, and right prisms.", "Rewrite simple rational expressions in different forms; write a(x)/b(x) in the form q(x) + r(x)/b(x), where a(x), b(x), q(x), and r(x) are polynomials with the degree of r(x) less than the degree of b(x), using inspection, long division, or, for the more complicated examples, a computer algebra system.", "Measure volumes by counting unit cubes, using cubic cm, cubic in, cubic ft, and improvised units."],
            "problem_activity": "Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story.",
        }

        wrapper = TaggingWrapper(openai_client=True, model_name_or_path='meta-llama/Llama-2-70b-chat-hf', tokenizer_name_or_path='meta-llama/Llama-2-70b-chat-hf', max_length=1024, buffer=20, truncate=True)
        truncated_messages = wrapper.truncate_messages([message])

        self.assertEqual(truncated_messages[0]['prompt_template'], message['prompt_template'])
        self.assertEqual(truncated_messages[0]['options'], message['options'])
        self.assertEqual(truncated_messages[0]['problem_activity'], message['problem_activity'])
  

    def test_truncate_messages_long(self):
        '''
        Want to check that models truncate very long messages
        '''
        message = {
            "role": "user", 
            "prompt_template": "You are a math expert reviewing K-12 curricular materials. You will be shown a problem or activity obtained from school curriculum and a description of mathematical concepts and skills. A problem or activity aligns with a standard if it can enable students to learn the full intent of the concepts and skills outlined in the standard's description. Your task is to assign the following problem or activity to the standard/s it aligns with, and format your output as a comma-separated list of options e.g. A, B, C. Output \"none\" if none of the descriptions are relevant. DO NOT make up additional standards. \n\nProblem/activity:\n{problem_activity}\n\nHere are possible math standards:\n{options}",
            "options": ["Explain why addition and subtraction strategies work, using place value and the properties of operations. (Explanations may be supported by drawings or objects.)", "Solve word problems involving dollar bills, quarters, dimes, nickels, and pennies, using $ and \u00a2 symbols appropriately. Example: If you have 2 dimes and 3 pennies, how many cents do you have?", "Solve real-world and mathematical problems involving area, volume and surface area of two- and three-dimensional objects composed of triangles, quadrilaterals, polygons, cubes, and right prisms.", "Rewrite simple rational expressions in different forms; write a(x)/b(x) in the form q(x) + r(x)/b(x), where a(x), b(x), q(x), and r(x) are polynomials with the degree of r(x) less than the degree of b(x), using inspection, long division, or, for the more complicated examples, a computer algebra system.", "Measure volumes by counting unit cubes, using cubic cm, cubic in, cubic ft, and improvised units."],
            "problem_activity": "Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. ",
        }

        wrapper = TaggingWrapper(openai_client=True, model_name_or_path='meta-llama/Llama-2-70b-chat-hf', tokenizer_name_or_path='meta-llama/Llama-2-70b-chat-hf', max_length=512, buffer=20, truncate=True)
        
        original_tokenized_content = wrapper.tokenizer(message['problem_activity'], return_tensors='pt', return_length=True, add_special_tokens=False)
        self.assertGreater(original_tokenized_content['length'][0], 512, msg='length: {}'.format(original_tokenized_content['length'][0])) 

        
        truncated_messages = wrapper.truncate_messages([message])

        self.assertEqual(truncated_messages[0]['prompt_template'], message['prompt_template'])
        self.assertEqual(truncated_messages[0]['options'], message['options'])

        tokenized_content = wrapper.tokenizer(truncated_messages[0]['problem_activity'], return_tensors='pt', return_length=True, add_special_tokens=False)
        self.assertLess(tokenized_content['length'][0], 512) 


    def test_run_model(self):
        wrapper = TaggingWrapper(openai_client=True, model_name_or_path='meta-llama/Llama-2-70b-chat-hf', tokenizer_name_or_path='meta-llama/Llama-2-70b-chat-hf', max_length=4097, buffer=10, truncate=True)
        message = {'role': 'user', 'prompt_template': 'You are a math instructor reviewing K-12 curricular materials. You will be shown a problem or activity obtained f\
            rom school curriculum and a list of one or more standards. Your task is to assign the problem or activity to one or more relevant standards it aligns $\
            ith. A problem or activity aligns with a standard if it can enable students to learn the full intent of the concepts and skills outlined in the standar\
            d\'s description.\n\nYour response should first begin with a paragraph explaining which standards the problem aligns with, and then output a comma-sepa\
            rated list of options. Respond "none" if the problem/activity aligns with none of the provided standards. Do not make up additional standards. Please f\
            ormat your response in two lines, as shown in the example below:\n\nThought: <your paragraph goes here>\nAnswer: A, C, E\n\nProblem/activity:\n{problem\
            _activity}\n\nStandard options:\n{options}\n\nYour response:', 
            'options': ['Find the point on a directed line segment between two given points that par\
            titions the segment in a given ratio.', 'Determine the unknown whole number in a multiplication or division equation relating three whole numbers. For \
            example, determine the unknown number that makes the equation true in each of the equations 8 × ? = 48, 5 = �� ÷ 3, 6 × 6 = ?.', 'Given a two-digit num\
            ber, mentally find 10 more or 10 less than the number, without having to count; explain the reasoning used.', 'Construct an equilateral triangle, a squ\
            are, and a regular hexagon inscribed in a circle.', 'Use multiplication and division within 100 to solve word problems in situations involving equal gr\
            oups, arrays, and measurement quantities, e.g., by using drawings and equations with a symbol for the unknown number to represent the problem.'], 
            'problem_activity': 'Narrative\nThe purpose of this activity is for students to write equations for multiplication situations and diagrams using a symbol fo\
            r the unknown number. When students write an equation to represent a situation, including a symbol for the unknown number, they model a situation with \
            mathematics (MP4).\nStudents find an unknown factor or unknown product in multiplication problems. In this task, the unknown factor diagrams and situat\
            ions only include the “how many groups” problem type and the factors 2, 5, and 10. This sets students up to skip-count to find the unknown number.\nThi\
            s problem type will be revisited extensively in future lessons and will be related to division. It is not necessary to make the connection to division \
            now. In the synthesis students explain how the equations they wrote represent the diagram or situation.\nRepresentation: Internalize Comprehension.\nSy\
            nthesis: Invite students to identify which details were important or most useful to pay attention to. Display the sentence frame, “The next time I writ\
            e an equation with an unknown number, I will . . . .“\nSupports accessibility for: Visual-Spatial Processing\nLaunch\nGroups of 2\nActivity\n“Now you w\
            ill practice writing your own equations with a symbol for the unknown.”\n2–3 minutes: independent work time\n“Share your equations with your partner. D\
            iscuss how you know each equation matches the diagram or situation.”\n2–3 minutes: partner discussion\nHave a whole-class discussion focused on how the\
            equations match the different representations.\nConsider asking:\n“How did you use the representations to write an equation with a symbol for the unkn\
            own?” (I looked for what was missing in the diagram. I thought about the situation to figure out if it was the number in each group, the number of grou\
            ps, or the total that was missing.)\n“Now find the missing number in each equation and write a new equation that includes the solution.”\n3–5 minutes: \
            partner work time\nStudent Facing\nWrite an equation to represent each diagram or situation. Use a symbol for the unknown. Be prepared to share your re\
            asoning.\nFind the number that makes each equation true. Rewrite the equation with the solution.\n| Narrative\nThe purpose of this activity is for students to write equations for multiplication situations and diagrams using a symbol fo\
            r the unknown number. When students write an equation to represent a situation, including a symbol for the unknown number, they model a situation with \
            mathematics (MP4).\nStudents find an unknown factor or unknown product in multiplication problems. In this task, the unknown factor diagrams and situat\
            ions only include the “how many groups” problem type and the factors 2, 5, and 10. This sets students up to skip-count to find the unknown number.\nThi\
            s problem type will be revisited extensively in future lessons and will be related to division. It is not necessary to make the connection to division \
            now. In the synthesis students explain how the equations they wrote represent the diagram or situation.\nRepresentation: Internalize Comprehension.\nSy\
            nthesis: Invite students to identify which details were important or most useful to pay attention to. Display the sentence frame, “The next time I writ\
            e an equation with an unknown number, I will . . . .“\nSupports accessibility for: Visual-Spatial Processing\nLaunch\nGroups of 2\nActivity\n“Now you w\
            ill practice writing your own equations with a symbol for the unknown.”\n2–3 minutes: independent work time\n“Share your equations with your partner. D\
            iscuss how you know each equation matches the diagram or situation.”\n2–3 minutes: partner discussion\nHave a whole-class discussion focused on how the\
            equations match the different representations.\nConsider asking:\n“How did you use the representations to write an equation with a symbol for the unkn\
            own?” (I looked for what was missing in the diagram. I thought about the situation to figure out if it was the number in each group, the number of grou\
            ps, or the total that was missing.)\n“Now find the missing number in each equation and write a new equation that includes the solution.”\n3–5 minutes: \
            partner work time\nStudent Facing\nWrite an equation to represent each diagram or situation. Use a symbol for the unknown. Be prepared to share your re\
            asoning.\nFind the number that makes each equation true. Rewrite the equation with the solution.\n| Narrative\nThe purpose of this activity is for students to write equations for multiplication situations and diagrams using a symbol fo\
            r the unknown number. When students write an equation to represent a situation, including a symbol for the unknown number, they model a situation with \
            mathematics (MP4).\nStudents find an unknown factor or unknown product in multiplication problems. In this task, the unknown factor diagrams and situat\
            ions only include the “how many groups” problem type and the factors 2, 5, and 10. This sets students up to skip-count to find the unknown number.\nThi\
            s problem type will be revisited extensively in future lessons and will be related to division. It is not necessary to make the connection to division \
            now. In the synthesis students explain how the equations they wrote represent the diagram or situation.\nRepresentation: Internalize Comprehension.\nSy\
            nthesis: Invite students to identify which details were important or most useful to pay attention to. Display the sentence frame, “The next time I writ\
            e an equation with an unknown number, I will . . . .“\nSupports accessibility for: Visual-Spatial Processing\nLaunch\nGroups of 2\nActivity\n“Now you w\
            ill practice writing your own equations with a symbol for the unknown.”\n2–3 minutes: independent work time\n“Share your equations with your partner. D\
            iscuss how you know each equation matches the diagram or situation.”\n2–3 minutes: partner discussion\nHave a whole-class discussion focused on how the\
            equations match the different representations.\nConsider asking:\n“How did you use the representations to write an equation with a symbol for the unkn\
            own?” (I looked for what was missing in the diagram. I thought about the situation to figure out if it was the number in each group, the number of grou\
            ps, or the total that was missing.)\n“Now find the missing number in each equation and write a new equation that includes the solution.”\n3–5 minutes: \
            partner work time\nStudent Facing\nWrite an equation to represent each diagram or situation. Use a symbol for the unknown. Be prepared to share your re\
            asoning.\nFind the number that makes each equation true. Rewrite the equation with the solution.\n|'
        }


        # truncated_messages = self.wrapper.truncate_messages([message])
        # for m in truncated_messages:
        #     tokenized_m = self.wrapper.tokenizer(m['content'], truncation=False, return_length=True)
        #     self.assertLess(tokenized_m['length'][0], 512)


    # def test_truncate_message_tokens(self):

    #     message = {
    #         "role": "user", 
    #         "prompt_template": "You are a math expert reviewing K-12 curricular materials. You will be shown a problem or activity obtained from school curriculum and a description of mathematical concepts and skills. A problem or activity aligns with a standard if it can enable students to learn the full intent of the concepts and skills outlined in the standard's description. Your task is to assign the following problem or activity to the standard/s it aligns with, and format your output as a comma-separated list of options e.g. A, B, C. Output \"none\" if none of the descriptions are relevant. DO NOT make up additional standards. \n\nProblem/activity:\n{problem_activity}\n\nHere are possible math standards:\n{options}",
    #         "options": ["Explain why addition and subtraction strategies work, using place value and the properties of operations. (Explanations may be supported by drawings or objects.)", "Solve word problems involving dollar bills, quarters, dimes, nickels, and pennies, using $ and \u00a2 symbols appropriately. Example: If you have 2 dimes and 3 pennies, how many cents do you have?", "Solve real-world and mathematical problems involving area, volume and surface area of two- and three-dimensional objects composed of triangles, quadrilaterals, polygons, cubes, and right prisms.", "Rewrite simple rational expressions in different forms; write a(x)/b(x) in the form q(x) + r(x)/b(x), where a(x), b(x), q(x), and r(x) are polynomials with the degree of r(x) less than the degree of b(x), using inspection, long division, or, for the more complicated examples, a computer algebra system.", "Measure volumes by counting unit cubes, using cubic cm, cubic in, cubic ft, and improvised units."],
    #         "problem_activity": "Lesson Synthesis\nDisplay one of the story problems from the lesson.\n\u201cToday we solved story problems where we knew how many of something there was at the beginning and at the end, but we didn't know how much it changed in the middle of the story. We looked at addition and subtraction methods that can be used to make sense of and solve these problems. Explain to your partner why it works to use addition or subtraction to solve these problems.\u201d (You can use addition to add on to one number until you get to the total. The number you added on is the unknown number. You can use subtraction to start with the total and take away the other number you know. The number you have left is the unknown number.)\n",
    #     }

    #      # tokenize everything seperately 
    #     template_tokens = self.wrapper.tokenizer(message.get('prompt_template'), truncation=False, return_length=True, add_special_tokens=False)
    #     options_tokens = self.wrapper.tokenizer(message.get('options'), truncation=False, return_length=True, add_special_tokens=False) # this will be a list
    #     problem_tokens = self.wrapper.tokenizer(message.get('problem_activity'), truncation=False, return_length=True, add_special_tokens=False)
        
    #     truncated_messages = self.wrapper._truncate_message_tokens(problem_tokens, options_tokens['input_ids'], message.get('prompt_template'), template_tokens)

    #     # check that problem text is truncated and the same for all messages
    #     min_problem_text = min([m['problem_activity'] for m in truncated_messages], key=len)
    #     min_problem_text_tokens = self.wrapper.tokenizer(min_problem_text, truncation=False, return_length=True, add_special_tokens=False)

    #     # print(min_problem_text_tokens, problem_tokens)
    #     # print(truncated_messages)
    #     assert (min_problem_text_tokens['length'][0] <= problem_tokens['length'][0])
    #     for m in truncated_messages:
    #         if 'problem_activity' in m.keys():
    #             self.assertEqual(m['problem_activity'], min_problem_text)
        
        # check that the sum of the tokens is less than the max_length





    #     # also assert that only the problem_activity is truncated
    #     instructions, problem_activity, description = self.seperate_prompt(self.wrapper.tokenizer.decode(truncated_messages['input_ids']))
    #     self.assertEqual(instructions, "You are a math expert reviewing K-12 curricular materials. You will be shown a problem or activity obtained from school curriculum and a description of mathematical concepts and skills. Your task is to determine whether the problem or activity can enable students to learn the full intent of the concepts and skills outlined in the provided description. Answer 'yes' if it does, and 'no' if it does not.")
    #     self.assertEqual(description, message['description'])



    # def test_truncate_messages(self):
    #     messages = ["Message 1", "Message 2", "Message 3"]
    #     truncated_messages = self.wrapper.truncate_messages(messages)
    #     self.assertEqual(truncated_messages, ["Truncated 1", "Truncated 2", "Truncated 3"])  # Add your assertion here

    # def test_truncate_message_tokens(self):
    #     problem_tokens = ["Token 1", "Token 2", "Token 3"]
    #     options_tokens = ["Option 1", "Option 2", "Option 3"]
    #     prompt_template = "This is a prompt template."
    #     truncated_tokens = self.wrapper._truncate_message_tokens(problem_tokens, options_tokens, prompt_template)
    #     self.assertEqual(truncated_tokens, ["Truncated Token 1", "Truncated Token 2", "Truncated Token 3"])  # Add your assertion here

if __name__ == '__main__':
    unittest.main()
