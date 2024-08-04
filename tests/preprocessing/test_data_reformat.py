"""
Tests data_augment.py in preprocessors

Author: Lucy Li (@lucy3)
Email:  lucyl@allenai.org
"""


import pathlib
import unittest
from unittest import TestCase

from mathfish.preprocessors.data_reformat import DataReformatter

class TestDataReformat(TestCase):

    def test_get_table(self):
        '''
        Note that this function in DataReformatter is essentially a wrapper 
        around various other packages' functions.
        '''
        table_html = "<table>\
                <thead>\
                  <tr>\
                    <th></th>\
                    <th>apple</th>\
                    <th>pear</th>\
                    <th>banana</th>\
                    <th>peach</th>\
                  </tr>\
                </thead>\
                <tbody>\
                  <tr>\
                    <td>price</td>\
                    <td>1.00</td>\
                    <td>2.00</td>\
                    <td>1.50</td>\
                    <td>5.00</td>\
                  </tr>\
                </tbody>\
                </table>"
        
        reformatter = DataReformatter(table_style='html')
        html = reformatter.get_table(table_html)
        self.assertEqual(html, table_html)

        reformatter = DataReformatter(table_style='special_token')
        token = reformatter.get_table(table_html)
        self.assertEqual(token, "[TAB]")

        reformatter = DataReformatter(table_style='json')
        table_json = reformatter.get_table(table_html)
        table_json_correct = '{"table": [{"thead": [{"tr": [{"th": [{}, {"_value": "apple"}, {"_value": "pear"}, {"_value": "banana"}, {"_value": "peach"}]}]}], "tbody": [{"tr": [{"td": [{"_value": "price"}, {"_value": "1.00"}, {"_value": "2.00"}, {"_value": "1.50"}, {"_value": "5.00"}]}]}]}]}'
        self.assertEqual(table_json, table_json_correct)

        reformatter = DataReformatter(table_style='rst')
        table_rst = reformatter.get_table(table_html)
        table_rst_correct = "\
+-------+-------+------+--------+-------+\n\
|       | apple | pear | banana | peach |\n\
+=======+=======+======+========+=======+\n\
| price | 1.00  | 2.00 | 1.50   | 5.00  |\n\
+-------+-------+------+--------+-------+"
        self.assertEqual(table_rst, table_rst_correct)

        reformatter = DataReformatter(table_style='markdown')
        table_md = reformatter.get_table(table_html)
        table_md_correct = "\
|       | apple | pear | banana | peach |\n\
|-------|-------|------|--------|-------|\n\
| price | 1.00  | 2.00 |  1.50  | 5.00  |"
        self.assertEqual(table_md, table_md_correct)

if __name__ == "__main__":
    unittest.main()
