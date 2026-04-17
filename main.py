from __future__ import annotations

import ast
import cmath
import math
import operator
import statistics
from typing import Any, Callable

from fastmcp import FastMCP

mcp = FastMCP("Technical Calculator")

Number = int | float | complex

_BINARY_OPERATORS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
}

_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Invert: operator.invert,
}

_BASE_DIGITS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _normalize_number(value: Number) -> Number | str:
    if isinstance(value, complex):
        real = 0.0 if math.isclose(value.real, 0.0, abs_tol=1e-12) else value.real
        imag = 0.0 if math.isclose(value.imag, 0.0, abs_tol=1e-12) else value.imag
        if imag == 0.0:
            return _normalize_number(real)
        return f"{real}{imag:+}j"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        if value.is_integer():
            return int(value)
    return value


def _serialize(value: Any) -> Any:
    if isinstance(value, (int, float, complex)) and not isinstance(value, bool):
        return _normalize_number(value)
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    return value


def _coerce_values(*values: float | list[float]) -> list[float]:
    if len(values) == 1 and isinstance(values[0], list):
        return values[0]
    return list(values)


def _nth_root(value: float, n: int) -> float:
    if n == 0:
        raise ValueError("The root degree must be non-zero.")
    if value < 0 and n % 2 == 0:
        raise ValueError("Even roots of negative numbers are not real.")
    if value < 0:
        return -((-value) ** (1 / n))
    return value ** (1 / n)


def _factorial(value: int) -> int:
    if value < 0:
        raise ValueError("Factorial is only defined for non-negative integers.")
    return math.factorial(value)


def _convert_angle_in(angle: float, degrees: bool) -> float:
    return math.radians(angle) if degrees else angle


def _convert_angle_out(angle: float, degrees: bool) -> float:
    return math.degrees(angle) if degrees else angle


def _is_prime_number(value: int) -> bool:
    if value < 2:
        return False
    if value in (2, 3):
        return True
    if value % 2 == 0:
        return False
    limit = math.isqrt(value)
    for candidate in range(3, limit + 1, 2):
        if value % candidate == 0:
            return False
    return True


def _prime_factorization(value: int) -> list[int]:
    if value == 0:
        raise ValueError("Prime factorization is undefined for zero.")
    n = abs(value)
    factors: list[int] = []
    while n % 2 == 0:
        factors.append(2)
        n //= 2
    divisor = 3
    while divisor * divisor <= n:
        while n % divisor == 0:
            factors.append(divisor)
            n //= divisor
        divisor += 2
    if n > 1:
        factors.append(n)
    if value < 0:
        return [-1, *factors]
    return factors


def _format_in_base(value: int, base: int) -> str:
    if not 2 <= base <= 36:
        raise ValueError("Bases must be between 2 and 36.")
    if value == 0:
        return "0"
    sign = "-" if value < 0 else ""
    n = abs(value)
    digits: list[str] = []
    while n > 0:
        n, remainder = divmod(n, base)
        digits.append(_BASE_DIGITS[remainder])
    return sign + "".join(reversed(digits))


def _sequence_sum(*values: float | list[float]) -> float:
    return math.fsum(_coerce_values(*values))


def _sequence_product(*values: float | list[float]) -> float:
    return math.prod(_coerce_values(*values))


def _sequence_mean(*values: float | list[float]) -> float:
    sequence = _coerce_values(*values)
    if not sequence:
        raise ValueError("At least one value is required.")
    return statistics.fmean(sequence)


def _sequence_median(*values: float | list[float]) -> float:
    sequence = _coerce_values(*values)
    if not sequence:
        raise ValueError("At least one value is required.")
    return statistics.median(sequence)


def _sequence_pstdev(*values: float | list[float]) -> float:
    sequence = _coerce_values(*values)
    if not sequence:
        raise ValueError("At least one value is required.")
    return statistics.pstdev(sequence)


def _sequence_stdev(*values: float | list[float]) -> float:
    sequence = _coerce_values(*values)
    if len(sequence) < 2:
        raise ValueError("At least two values are required for sample standard deviation.")
    return statistics.stdev(sequence)


def _build_expression_context(degrees: bool) -> dict[str, Any]:
    return {
        "pi": math.pi,
        "tau": math.tau,
        "e": math.e,
        "inf": math.inf,
        "nan": math.nan,
        "j": 1j,
        "abs": abs,
        "round": round,
        "floor": math.floor,
        "ceil": math.ceil,
        "trunc": math.trunc,
        "sqrt": math.sqrt,
        "cbrt": lambda x: _nth_root(x, 3),
        "root": _nth_root,
        "pow": pow,
        "exp": math.exp,
        "ln": math.log,
        "log": math.log,
        "log10": math.log10,
        "log2": math.log2,
        "sin": lambda x: math.sin(_convert_angle_in(x, degrees)),
        "cos": lambda x: math.cos(_convert_angle_in(x, degrees)),
        "tan": lambda x: math.tan(_convert_angle_in(x, degrees)),
        "asin": lambda x: _convert_angle_out(math.asin(x), degrees),
        "acos": lambda x: _convert_angle_out(math.acos(x), degrees),
        "atan": lambda x: _convert_angle_out(math.atan(x), degrees),
        "sinh": math.sinh,
        "cosh": math.cosh,
        "tanh": math.tanh,
        "degrees": math.degrees,
        "radians": math.radians,
        "hypot": math.hypot,
        "factorial": _factorial,
        "comb": math.comb,
        "perm": math.perm,
        "gcd": math.gcd,
        "lcm": math.lcm,
        "sum": _sequence_sum,
        "prod": _sequence_product,
        "mean": _sequence_mean,
        "median": _sequence_median,
        "pstdev": _sequence_pstdev,
        "stdev": _sequence_stdev,
        "remainder": math.remainder,
    }


def _evaluate_expression(node: ast.AST, context: dict[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _evaluate_expression(node.body, context)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float, complex)):
            raise ValueError("Only numeric constants are allowed.")
        return node.value

    if isinstance(node, ast.Name):
        if node.id not in context:
            raise ValueError(f"Unknown name '{node.id}'.")
        return context[node.id]

    if isinstance(node, ast.List):
        return [_evaluate_expression(element, context) for element in node.elts]

    if isinstance(node, ast.Tuple):
        return tuple(_evaluate_expression(element, context) for element in node.elts)

    if isinstance(node, ast.BinOp):
        operator_fn = _BINARY_OPERATORS.get(type(node.op))
        if operator_fn is None:
            raise ValueError("That operator is not supported.")
        return operator_fn(
            _evaluate_expression(node.left, context),
            _evaluate_expression(node.right, context),
        )

    if isinstance(node, ast.UnaryOp):
        operator_fn = _UNARY_OPERATORS.get(type(node.op))
        if operator_fn is None:
            raise ValueError("That unary operator is not supported.")
        return operator_fn(_evaluate_expression(node.operand, context))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct function calls are allowed.")
        function = context.get(node.func.id)
        if not callable(function):
            raise ValueError(f"Unknown function '{node.func.id}'.")
        args = [_evaluate_expression(argument, context) for argument in node.args]
        kwargs = {
            keyword.arg: _evaluate_expression(keyword.value, context)
            for keyword in node.keywords
            if keyword.arg is not None
        }
        if len(kwargs) != len(node.keywords):
            raise ValueError("Dictionary unpacking is not supported.")
        return function(*args, **kwargs)

    raise ValueError("The expression contains unsupported syntax.")


@mcp.tool
def calculate(expression: str, degrees: bool = False) -> dict[str, Any]:
    """Safely evaluate a calculator expression with arithmetic, trig, logs, statistics, and bitwise operators."""
    parsed = ast.parse(expression, mode="eval")
    result = _evaluate_expression(parsed, _build_expression_context(degrees))
    return {
        "expression": expression,
        "degrees_mode": degrees,
        "result": _serialize(result),
    }


@mcp.tool
def constants() -> dict[str, float]:
    """Return common mathematical constants."""
    return {
        "pi": math.pi,
        "tau": math.tau,
        "e": math.e,
        "phi": (1 + math.sqrt(5)) / 2,
        "sqrt2": math.sqrt(2),
        "sqrt3": math.sqrt(3),
    }


@mcp.tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@mcp.tool
def subtract(a: float, b: float) -> float:
    """Subtract the second number from the first."""
    return a - b


@mcp.tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@mcp.tool
def divide(a: float, b: float) -> float:
    """Divide one number by another."""
    return a / b


@mcp.tool
def floor_divide(a: float, b: float) -> float:
    """Perform floor division."""
    return a // b


@mcp.tool
def modulo(a: float, b: float) -> float:
    """Return the remainder after division."""
    return a % b


@mcp.tool
def power(base: float, exponent: float) -> float:
    """Raise a number to a power."""
    return base**exponent


@mcp.tool
def nth_root(value: float, n: int) -> float:
    """Return the real nth root of a number."""
    return _nth_root(value, n)


@mcp.tool
def reciprocal(value: float) -> float:
    """Return the multiplicative inverse of a number."""
    return 1 / value


@mcp.tool
def percentage(part: float, whole: float) -> float:
    """Return what percent one value is of another."""
    return (part / whole) * 100


@mcp.tool
def percentage_change(original: float, new: float) -> float:
    """Return the percentage change from the original value to the new value."""
    return ((new - original) / original) * 100


@mcp.tool
def sin(angle: float, degrees: bool = False) -> float:
    """Return the sine of an angle."""
    return math.sin(_convert_angle_in(angle, degrees))


@mcp.tool
def cos(angle: float, degrees: bool = False) -> float:
    """Return the cosine of an angle."""
    return math.cos(_convert_angle_in(angle, degrees))


@mcp.tool
def tan(angle: float, degrees: bool = False) -> float:
    """Return the tangent of an angle."""
    return math.tan(_convert_angle_in(angle, degrees))


@mcp.tool
def asin(value: float, degrees: bool = False) -> float:
    """Return the inverse sine."""
    return _convert_angle_out(math.asin(value), degrees)


@mcp.tool
def acos(value: float, degrees: bool = False) -> float:
    """Return the inverse cosine."""
    return _convert_angle_out(math.acos(value), degrees)


@mcp.tool
def atan(value: float, degrees: bool = False) -> float:
    """Return the inverse tangent."""
    return _convert_angle_out(math.atan(value), degrees)


@mcp.tool
def logarithm(value: float, base: float = math.e) -> float:
    """Return the logarithm of a value in the requested base."""
    return math.log(value, base)


@mcp.tool
def ln(value: float) -> float:
    """Return the natural logarithm."""
    return math.log(value)


@mcp.tool
def log10(value: float) -> float:
    """Return the base-10 logarithm."""
    return math.log10(value)


@mcp.tool
def sqrt(value: float) -> float:
    """Return the square root of a non-negative number."""
    return math.sqrt(value)


@mcp.tool
def factorial(n: int) -> int:
    """Return n! for a non-negative integer."""
    return _factorial(n)


@mcp.tool
def combination(n: int, r: int) -> int:
    """Return the number of combinations of r items from n items."""
    return math.comb(n, r)


@mcp.tool
def permutation(n: int, r: int) -> int:
    """Return the number of permutations of r items from n items."""
    return math.perm(n, r)


@mcp.tool
def gcd(a: int, b: int) -> int:
    """Return the greatest common divisor of two integers."""
    return math.gcd(a, b)


@mcp.tool
def lcm(a: int, b: int) -> int:
    """Return the least common multiple of two integers."""
    return math.lcm(a, b)


@mcp.tool
def is_prime(n: int) -> bool:
    """Return whether an integer is prime."""
    return _is_prime_number(n)


@mcp.tool
def prime_factors(n: int) -> list[int]:
    """Return the prime factorization of an integer."""
    return _prime_factorization(n)


@mcp.tool
def sum_values(values: list[float]) -> float:
    """Return the sum of a list of numbers."""
    return math.fsum(values)


@mcp.tool
def product(values: list[float]) -> float:
    """Return the product of a list of numbers."""
    return math.prod(values)


@mcp.tool
def mean(values: list[float]) -> float:
    """Return the arithmetic mean of a list of numbers."""
    if not values:
        raise ValueError("At least one value is required.")
    return statistics.fmean(values)


@mcp.tool
def median(values: list[float]) -> float:
    """Return the median of a list of numbers."""
    if not values:
        raise ValueError("At least one value is required.")
    return statistics.median(values)


@mcp.tool
def standard_deviation(values: list[float], sample: bool = False) -> float:
    """Return the population or sample standard deviation of a list of numbers."""
    if not values:
        raise ValueError("At least one value is required.")
    if sample:
        if len(values) < 2:
            raise ValueError("At least two values are required for sample standard deviation.")
        return statistics.stdev(values)
    return statistics.pstdev(values)


@mcp.tool
def quadratic_roots(a: float, b: float, c: float) -> dict[str, Any]:
    """Solve a quadratic equation and return its roots."""
    if a == 0:
        raise ValueError("Coefficient 'a' must be non-zero for a quadratic equation.")
    discriminant = b**2 - 4 * a * c
    root = cmath.sqrt(discriminant)
    roots = [(-b + root) / (2 * a), (-b - root) / (2 * a)]
    return {
        "discriminant": _serialize(discriminant),
        "roots": _serialize(roots),
    }


@mcp.tool
def convert_base(value: str, from_base: int, to_base: int) -> str:
    """Convert an integer string from one base to another."""
    if not 2 <= from_base <= 36 or not 2 <= to_base <= 36:
        raise ValueError("Bases must be between 2 and 36.")
    decimal_value = int(value.strip(), from_base)
    return _format_in_base(decimal_value, to_base)


if __name__ == "__main__":
    mcp.run()
