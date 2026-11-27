import sys
import os
# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import load_model, get_response  
import unittest
class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls): 
        cls.model, cls.tokenizer= load_model()
    def test_factual_query(self):
        prompt="What is LLM?"
        response=get_response(prompt, self.model, self.tokenizer, max_length=200)
        self.assertIn("Large Language Model",response)
    def test_code_generation(self):
        prompt="Write a Python function to calcutate the product of two numbers."
        response=get_response(prompt, self.model, self.tokenizer, max_length=250)
        self.assertIn("multiply",response)
        self.assertIn("*",response)
if __name__ == '__main__':
    unittest.main(verbosity=2, exit=False)
 
