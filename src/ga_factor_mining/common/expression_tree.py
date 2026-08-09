"""遗传因子表达式树的通用结构操作。"""

from __future__ import annotations

import copy
import json
from typing import Any, Iterable


UNARY_OPERATORS = ("neg", "abs", "signed_log", "signed_sqrt")
BINARY_OPERATORS = ("add", "sub", "mul", "div", "min", "max")
TEMPORAL_OPERATORS = ("delta_5", "slope_5", "mean_5", "std_20", "zscore_20", "ts_rank_20")


def canonical(expr: Any) -> str:
    return json.dumps(expr, ensure_ascii=False, separators=(",", ":"))


def depth(expr: Any) -> int:
    return 0 if isinstance(expr, str) else 1 + max(depth(child) for child in expr[1:])


def nodes(expr: Any) -> int:
    return 1 if isinstance(expr, str) else 1 + sum(nodes(child) for child in expr[1:])


def expression_text(expr: Any, separator: str = ",") -> str:
    if isinstance(expr, str):
        return expr
    return f"{expr[0]}({separator.join(expression_text(child, separator) for child in expr[1:])})"


def paths(expr: Any, prefix: tuple[int, ...] = ()) -> list[tuple[int, ...]]:
    result = [prefix]
    if not isinstance(expr, str):
        for index in range(1, len(expr)):
            result.extend(paths(expr[index], prefix + (index,)))
    return result


def get_subtree(expr: Any, path: Iterable[int]) -> Any:
    for index in path:
        expr = expr[index]
    return expr


def replace_subtree(expr: Any, path: tuple[int, ...], value: Any) -> Any:
    if not path:
        return copy.deepcopy(value)
    result = copy.deepcopy(expr)
    cursor = result
    for index in path[:-1]:
        cursor = cursor[index]
    cursor[path[-1]] = copy.deepcopy(value)
    return result


def valid_expression(expr: Any, temporal_operators: Iterable[str] = TEMPORAL_OPERATORS) -> bool:
    if isinstance(expr, str):
        return True
    temporal = set(temporal_operators)
    if expr[0] in temporal:
        return len(expr) == 2 and isinstance(expr[1], str)
    return all(valid_expression(child, temporal) for child in expr[1:])
