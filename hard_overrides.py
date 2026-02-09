import re
import ast

# -----------------------------
# Helper: safe AST parsing
# -----------------------------
def get_ast(code):
    try:
        return ast.parse(code)
    except:
        return None


# -----------------------------
# FACTORIAL DETECTION
# -----------------------------

def is_iterative_factorial(code: str) -> bool:
    code_l = code.lower()

    # Fast keyword path
    if "factorial" in code_l:
        return True

    # Structural heuristic
    if "range" in code and "*" in code:
        if re.search(r"\b=\s*1\b", code):  # accumulator init
            if re.search(r"\*\=", code) or re.search(r"=\s*\w+\s*\*", code):
                return True

    return False


def is_recursive_factorial(code: str) -> bool:
    tree = get_ast(code)
    if not tree:
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Return):
            src = ast.unparse(node) if hasattr(ast, "unparse") else ""
            if re.search(r"n\s*\*\s*\w+\(n\s*-\s*1\)", src):
                return True

    return False


# -----------------------------
# FIBONACCI DETECTION
# -----------------------------

def is_recursive_fibonacci(code: str) -> bool:
    tree = get_ast(code)
    if not tree:
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Return):
            src = ast.unparse(node) if hasattr(ast, "unparse") else ""
            if re.search(r"\w+\(n\s*-\s*1\)\s*\+\s*\w+\(n\s*-\s*2\)", src):
                return True

    return False


def is_iterative_fibonacci(code: str) -> bool:
    code_l = code.lower()

    # tuple unpacking pattern
    if re.search(r"\w+\s*,\s*\w+\s*=\s*\w+\s*,\s*\w+\s*\+\s*\w+", code):
        return True

    return False


# -----------------------------
# STRING / NUMBER UTILITIES
# -----------------------------

def is_palindrome(code: str) -> bool:
    return (
        "[::-1]" in code or
        "reversed" in code.lower()
    )


def is_reverse_string(code: str) -> bool:
    return (
        "[::-1]" in code or
        re.search(r"for\s+\w+\s+in\s+.*:\s*\n\s*\w+\s*=\s*\w+\s*\+\s*\w+", code)
    )


def is_prime_number(code: str) -> bool:
    return (
        "%" in code and
        ("range" in code) and
        ("** 0.5" in code or "sqrt" in code.lower())
    )


# -----------------------------
# MASTER OVERRIDE DISPATCHER
# -----------------------------

def detect_hard_override(code: str):
    """
    Returns:
        dict {category, topic, confidence}
        OR None if no hard match
    """

    # Factorial
    if is_recursive_factorial(code):
        return {
            "category": "Basic Python",
            "topic": "Recursive Factorial",
            "confidence": 1.0
        }

    if is_iterative_factorial(code):
        return {
            "category": "Basic Python",
            "topic": "Iterative Factorial",
            "confidence": 0.95
        }

    # Fibonacci
    if is_recursive_fibonacci(code):
        return {
            "category": "Basic Python",
            "topic": "Recursive Fibonacci",
            "confidence": 1.0
        }

    if is_iterative_fibonacci(code):
        return {
            "category": "Basic Python",
            "topic": "Iterative Fibonacci",
            "confidence": 0.95
        }

    # String / number programs
    if is_palindrome(code):
        return {
            "category": "Basic Python",
            "topic": "Palindrome",
            "confidence": 0.9
        }

    if is_reverse_string(code):
        return {
            "category": "Basic Python",
            "topic": "Reverse String",
            "confidence": 0.9
        }

    if is_prime_number(code):
        return {
            "category": "Basic Python",
            "topic": "Prime Number",
            "confidence": 0.9
        }

    return None
