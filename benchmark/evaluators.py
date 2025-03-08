"""
Evaluation functions for assessing model responses in instruction following tests.
"""

import re
import string
from typing import Optional, Callable, Dict, Any, Union, List

def normalize_text(text: str) -> str:
    """
    Normalize text for comparison by converting to lowercase,
    removing extra whitespace, and trailing punctuation.
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    # Remove trailing punctuation
    text = text.rstrip(string.punctuation)
    
    return text.strip()

def exact_match(response: str, expected: str) -> float:
    """
    Check if the response exactly matches the expected output,
    ignoring case, extra whitespace, and trailing punctuation.
    
    Args:
        response: Model response
        expected: Expected output
        
    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    return 1.0 if normalize_text(response) == normalize_text(expected) else 0.0

def contains_match(response: str, expected: str) -> float:
    """
    Check if the response contains the expected output,
    ignoring case and whitespace.
    
    Args:
        response: Model response
        expected: Expected output
        
    Returns:
        1.0 if expected text is in response, 0.0 otherwise
    """
    return 1.0 if normalize_text(expected) in normalize_text(response) else 0.0

def extract_first_word(response: str) -> str:
    """
    Extract the first word from a response, useful for one-word answer tests.
    
    Args:
        response: Model response
        
    Returns:
        First word in the response
    """
    # Extract the first word, ignoring punctuation
    words = re.findall(r'\b\w+\b', response.lower())
    return words[0] if words else ""

def one_word_match(response: str, expected: str) -> float:
    """
    Check if the first word in the response matches the expected output.
    
    Args:
        response: Model response
        expected: Expected output (one word)
        
    Returns:
        1.0 if first word matches, 0.0 otherwise
    """
    first_word = extract_first_word(response)
    return 1.0 if first_word == normalize_text(expected) else 0.0

def number_match(response: str, expected: str, tolerance: float = 0.0001) -> float:
    """
    Extract numbers from response and check if any match the expected number.
    
    Args:
        response: Model response
        expected: Expected numerical output as string
        tolerance: Acceptable difference for floating point comparison
        
    Returns:
        1.0 if a matching number is found, 0.0 otherwise
    """
    # Try to convert expected to float
    try:
        expected_num = float(expected)
    except ValueError:
        return 0.0  # Not a number
    
    # Extract all numbers from the response
    numbers = re.findall(r'-?\d*\.?\d+', response)
    
    # Check if any extracted number matches the expected value
    for num_str in numbers:
        try:
            num = float(num_str)
            if abs(num - expected_num) <= tolerance:
                return 1.0
        except ValueError:
            continue
    
    return 0.0

def decimal_precision_match(response: str, expected: str, decimals: int) -> float:
    """
    Check if the response contains a number with the exact decimal precision.
    
    Args:
        response: Model response
        expected: Expected output with specific decimal places
        decimals: Required number of decimal places
        
    Returns:
        1.0 if matching number with correct precision is found, 0.0 otherwise
    """
    # Extract all decimal numbers from the response
    decimal_pattern = rf'\d+\.\d{{{decimals}}}'
    matches = re.findall(decimal_pattern, response)
    
    # Check if any extracted number matches the expected value
    for match in matches:
        if match == expected or float(match) == float(expected):
            return 1.0
    
    return 0.0

def sentence_count_match(response: str, expected_count: int) -> float:
    """
    Check if the response contains the expected number of sentences.
    
    Args:
        response: Model response
        expected_count: Expected number of sentences
        
    Returns:
        1.0 if sentence count matches, 0.0 otherwise
    """
    # Simple sentence detection using common sentence terminators
    sentences = re.split(r'[.!?]+\s*', response)
    # Remove empty strings that might result from the split
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return 1.0 if len(sentences) == expected_count else 0.0

def word_count_match(response: str, expected_count: int, relation: str = "exactly") -> float:
    """
    Check if the response contains the expected number of words.
    
    Args:
        response: Model response
        expected_count: Expected word count
        relation: Relation type ("exactly", "at least", "at most")
        
    Returns:
        1.0 if word count meets the relation, 0.0 otherwise
    """
    # Count words (sequences of non-whitespace characters)
    words = re.findall(r'\b\w+\b', response)
    count = len(words)
    
    if relation == "exactly":
        return 1.0 if count == expected_count else 0.0
    elif relation == "at least":
        return 1.0 if count >= expected_count else 0.0
    elif relation == "at most":
        return 1.0 if count <= expected_count else 0.0
    else:
        return 0.0

def no_punctuation_match(response: str, punctuation_char: str) -> float:
    """
    Check if the response doesn't contain a specific punctuation character.
    
    Args:
        response: Model response
        punctuation_char: Punctuation character to check for absence
        
    Returns:
        1.0 if character is absent, 0.0 otherwise
    """
    return 1.0 if punctuation_char not in response else 0.0

def format_match(response: str, pattern: str) -> float:
    """
    Check if the response matches a specific format pattern.
    
    Args:
        response: Model response
        pattern: Regex pattern to match
        
    Returns:
        1.0 if pattern matches, 0.0 otherwise
    """
    return 1.0 if re.search(pattern, response) is not None else 0.0

def placeholder_count_match(response: str, expected_count: int, relation: str = "exactly") -> float:
    """
    Check if the response contains the expected number of placeholders [like this].
    
    Args:
        response: Model response
        expected_count: Expected number of placeholders
        relation: Relation type ("exactly", "at least", "at most")
        
    Returns:
        1.0 if placeholder count meets the relation, 0.0 otherwise
    """
    # Count placeholders like [this]
    placeholders = re.findall(r'\[[^\]]+\]', response)
    count = len(placeholders)
    
    if relation == "exactly":
        return 1.0 if count == expected_count else 0.0
    elif relation == "at least":
        return 1.0 if count >= expected_count else 0.0
    elif relation == "at most":
        return 1.0 if count <= expected_count else 0.0
    else:
        return 0.0

def case_match(response: str, case_type: str) -> float:
    """
    Check if the response follows specific case requirements.
    
    Args:
        response: Model response
        case_type: Type of case ("lowercase", "uppercase", "title", "sentence")
        
    Returns:
        1.0 if case requirements are met, 0.0 otherwise
    """
    if case_type == "lowercase":
        return 1.0 if response == response.lower() else 0.0
    elif case_type == "uppercase":
        return 1.0 if response == response.upper() else 0.0
    elif case_type == "title":
        # Check if each word starts with uppercase
        words = response.split()
        if not words:
            return 0.0
        return 1.0 if all(w[0].isupper() for w in words if w) else 0.0
    elif case_type == "sentence":
        # Check if the first letter of each sentence is uppercase
        sentences = re.split(r'[.!?]+\s*', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return 0.0
        return 1.0 if all(s[0].isupper() for s in sentences if s) else 0.0
    else:
        return 0.0

def get_evaluator(eval_type: str) -> Callable[[str, str], float]:
    """
    Get the appropriate evaluation function based on the evaluation type.
    
    Args:
        eval_type: Type of evaluation
        
    Returns:
        Evaluation function
    """
    evaluators = {
        "exact_match": exact_match,
        "contains": contains_match,
        "first_word": one_word_match,
        "number": number_match,
    }
    
    return evaluators.get(eval_type, exact_match)

def evaluate_response(response: str, expected: str, eval_type: str, **kwargs) -> float:
    """
    Evaluate a response using the specified evaluation type and parameters.
    
    Args:
        response: Model response
        expected: Expected output
        eval_type: Type of evaluation
        **kwargs: Additional parameters for the evaluation
        
    Returns:
        Score between 0.0 and 1.0
    """
    if not response:
        return 0.0
        
    if eval_type == "exact_match":
        return exact_match(response, expected)
    elif eval_type == "contains":
        return contains_match(response, expected)
    elif eval_type == "one_word":
        return one_word_match(response, expected)
    elif eval_type == "number":
        tolerance = kwargs.get("tolerance", 0.0001)
        return number_match(response, expected, tolerance)
    elif eval_type == "decimal_precision":
        decimals = kwargs.get("decimals", 4)
        return decimal_precision_match(response, expected, decimals)
    elif eval_type == "word_count":
        count = kwargs.get("count", 0)
        relation = kwargs.get("relation", "exactly")
        return word_count_match(response, count, relation)
    elif eval_type == "no_punctuation":
        char = kwargs.get("char", ",")
        return no_punctuation_match(response, char)
    elif eval_type == "format":
        pattern = kwargs.get("pattern", "")
        return format_match(response, pattern)
    elif eval_type == "placeholder_count":
        count = kwargs.get("count", 0)
        relation = kwargs.get("relation", "exactly")
        return placeholder_count_match(response, count, relation)
    elif eval_type == "case":
        case_type = kwargs.get("case_type", "lowercase")
        return case_match(response, case_type)
    else:
        # Default to exact match
        return exact_match(response, expected)