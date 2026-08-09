import unittest

from ga_factor_mining.common.expression_tree import (
    canonical,
    depth,
    expression_text,
    get_subtree,
    nodes,
    paths,
    replace_subtree,
    valid_expression,
)
from ga_factor_mining.common.paths import DATA_ROOT, OUTPUT_ROOT, REPORT_ROOT, REPOSITORY_ROOT


class CommonStructureTests(unittest.TestCase):
    def test_expression_tree_helpers_are_shared(self):
        expr = ["add", ["mean_5", "ret_5d"], "turnover"]
        self.assertEqual(depth(expr), 2)
        self.assertEqual(nodes(expr), 4)
        self.assertEqual(expression_text(expr), "add(mean_5(ret_5d),turnover)")
        self.assertTrue(valid_expression(expr))
        self.assertEqual(get_subtree(expr, (1, 1)), "ret_5d")
        self.assertEqual(replace_subtree(expr, (2,), "amount")[2], "amount")
        self.assertIn((1, 1), paths(expr))
        self.assertEqual(canonical("中文因子"), '"中文因子"')

    def test_repository_roots_are_isolated(self):
        self.assertEqual(DATA_ROOT.parent, REPOSITORY_ROOT)
        self.assertEqual(OUTPUT_ROOT.parent, REPOSITORY_ROOT)
        self.assertEqual(REPORT_ROOT.parent, REPOSITORY_ROOT)
        self.assertNotEqual(OUTPUT_ROOT, REPORT_ROOT)


if __name__ == "__main__":
    unittest.main()
