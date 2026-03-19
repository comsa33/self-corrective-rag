"""Calculator tool for agentic retrieval — financial and numerical computations."""

from __future__ import annotations

import json
import math


def make_calculate():
    """Create a calculate tool for safe mathematical expression evaluation."""

    def calculate(expression: str) -> str:
        """Evaluate a mathematical expression and return the result.

        Use this tool when the question requires numerical computation
        (e.g. financial ratios, growth rates, DPO, margins, percentages).

        Supported operations: +, -, *, /, **, (, ), and math functions
        (sqrt, log, abs, round, min, max, sum).

        Examples:
            "365 * 480 / 1200"                          → days payable
            "(2018 - 1755) / 1755 * 100"                → growth rate %
            "round((3502 - 3017) / 3017 * 100, 1)"     → margin change %

        Args:
            expression: A Python-style mathematical expression string.

        Returns:
            JSON string: {"expression": ..., "result": ..., "formatted": ...}
        """
        safe_names = {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "sqrt": math.sqrt,
            "log": math.log,
            "log10": math.log10,
            "exp": math.exp,
            "pi": math.pi,
            "e": math.e,
        }

        try:
            result = eval(expression, {"__builtins__": {}}, safe_names)
            if isinstance(result, float):
                formatted = f"{result:,.4f}".rstrip("0").rstrip(".")
            else:
                formatted = str(result)
            return json.dumps({"expression": expression, "result": result, "formatted": formatted})
        except ZeroDivisionError:
            return json.dumps({"error": "Division by zero", "expression": expression})
        except Exception as e:
            return json.dumps({"error": str(e), "expression": expression})

    return calculate
