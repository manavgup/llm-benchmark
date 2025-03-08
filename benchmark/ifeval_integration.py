"""
IFEval integration for LLM benchmarking.

This module provides integration with IFEval, a framework for evaluating instruction following
capabilities of language models.
"""

import os
import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseEvaluator(ABC):
    """Base class for all evaluators."""
    
    @abstractmethod
    def evaluate(self, response: str, instruction_type: str, kwargs: Dict) -> Tuple[bool, str]:
        """
        Evaluate a response against an instruction.
        
        Args:
            response: Model's response
            instruction_type: Type of instruction to evaluate
            kwargs: Additional parameters for evaluation
            
        Returns:
            Tuple of (passed, explanation)
        """
        pass

class PunctuationEvaluator(BaseEvaluator):
    """Evaluator for punctuation-related instructions."""
    
    def evaluate(self, response: str, instruction_type: str, kwargs: Dict) -> Tuple[bool, str]:
        if instruction_type == "punctuation:no_comma":
            passed = "," not in response
            return passed, f"No comma check: {'Passed' if passed else 'Failed'}"
        elif instruction_type == "punctuation:no_period":
            passed = "." not in response
            return passed, f"No period check: {'Passed' if passed else 'Failed'}"
        else:
            return False, f"Unknown punctuation instruction: {instruction_type}"

class FormatEvaluator(BaseEvaluator):
    """Evaluator for format-related instructions."""
    
    def evaluate(self, response: str, instruction_type: str, kwargs: Dict) -> Tuple[bool, str]:
        if "detectable_format:number_highlighted_sections" in instruction_type:
            num_highlights = kwargs.get("num_highlights", 3)
            pattern = r'\*([^*]+)\*'
            highlights = re.findall(pattern, response)
            passed = len(highlights) >= num_highlights
            return passed, f"Highlighted sections (expected {num_highlights}): {'Passed' if passed else 'Failed'}"
        elif "detectable_format:title" in instruction_type:
            pattern = r'<<[^>]+>>'
            passed = re.search(pattern, response) is not None
            return passed, f"Title format check: {'Passed' if passed else 'Failed'}"
        else:
            return False, f"Unknown format instruction: {instruction_type}"

class ContentEvaluator(BaseEvaluator):
    """Evaluator for content-related instructions."""
    
    def evaluate(self, response: str, instruction_type: str, kwargs: Dict) -> Tuple[bool, str]:
        if "detectable_content:number_placeholders" in instruction_type:
            num_placeholders = kwargs.get("num_placeholders", 0)
            pattern = r'\[[^\]]+\]'
            placeholders = re.findall(pattern, response)
            passed = len(placeholders) >= num_placeholders
            return passed, f"Placeholder count (expected {num_placeholders}): {'Passed' if passed else 'Failed'}"
        else:
            return False, f"Unknown content instruction: {instruction_type}"

class LengthEvaluator(BaseEvaluator):
    """Evaluator for length-related instructions."""
    
    def evaluate(self, response: str, instruction_type: str, kwargs: Dict) -> Tuple[bool, str]:
        if "length_constraints:number_words" in instruction_type:
            relation = kwargs.get("relation", "exactly")
            num_words = kwargs.get("num_words", 0)
            words = re.findall(r'\b\w+\b', response)
            word_count = len(words)
            
            if relation == "exactly":
                passed = word_count == num_words
            elif relation == "at least":
                passed = word_count >= num_words
            elif relation == "at most":
                passed = word_count <= num_words
            else:
                passed = False
                
            return passed, f"Word count check ({relation} {num_words}): {'Passed' if passed else 'Failed'} (found {word_count})"
        else:
            return False, f"Unknown length instruction: {instruction_type}"

class CaseEvaluator(BaseEvaluator):
    """Evaluator for text case instructions."""
    
    def evaluate(self, response: str, instruction_type: str, kwargs: Dict) -> Tuple[bool, str]:
        if "change_case:english_lowercase" in instruction_type:
            passed = response == response.lower()
            return passed, f"Lowercase check: {'Passed' if passed else 'Failed'}"
        elif "change_case:english_uppercase" in instruction_type:
            passed = response == response.upper()
            return passed, f"Uppercase check: {'Passed' if passed else 'Failed'}"
        else:
            return False, f"Unknown case instruction: {instruction_type}"

class CombinedEvaluator(BaseEvaluator):
    """Evaluator for combined or special instructions."""
    
    def evaluate(self, response: str, instruction_type: str, kwargs: Dict) -> Tuple[bool, str]:
        if "combination:repeat_prompt" in instruction_type:
            prompt_to_repeat = kwargs.get("prompt_to_repeat", "")
            # Check if the response contains the prompt to repeat
            passed = prompt_to_repeat in response
            return passed, f"Prompt repetition check: {'Passed' if passed else 'Failed'}"
        else:
            return False, f"Unknown combination instruction: {instruction_type}"

class IFEvalIntegration:
    """Integration with IFEval framework for instruction following evaluation."""
    
    def __init__(self, ifeval_dataset_path: Optional[str] = None):
        """
        Initialize IFEval integration.
        
        Args:
            ifeval_dataset_path: Path to IFEval dataset JSON file
        """
        self.ifeval_dataset_path = ifeval_dataset_path
        self.dataset = self._load_dataset()
        
        # Register evaluators
        self.evaluators = {
            "punctuation": PunctuationEvaluator(),
            "detectable_format": FormatEvaluator(),
            "detectable_content": ContentEvaluator(),
            "length_constraints": LengthEvaluator(),
            "change_case": CaseEvaluator(),
            "combination": CombinedEvaluator()
        }
        
    def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load IFEval dataset from file."""
        if not self.ifeval_dataset_path or not os.path.exists(self.ifeval_dataset_path):
            logger.warning(f"IFEval dataset not found: {self.ifeval_dataset_path}")
            return []
            
        try:
            with open(self.ifeval_dataset_path, 'r') as f:
                # IFEval data is stored as JSONL (one JSON object per line)
                return [json.loads(line) for line in f]
        except Exception as e:
            logger.error(f"Error loading IFEval dataset: {str(e)}")
            return []
    
    def convert_to_test_cases(self) -> List[Dict[str, Any]]:
        """
        Convert IFEval dataset to benchmark test cases.
        
        Returns:
            List of test case dictionaries
        """
        test_cases = []
        
        for item in self.dataset:
            # Extract key components from IFEval item
            key = item.get("key", "unknown")
            prompt = item.get("prompt", "")
            instruction_types = item.get("instruction_id_list", [])
            kwargs = item.get("kwargs", [{}])
            
            # Create a test case compatible with our benchmark
            test_case = {
                "name": f"IFEval_{key}",
                "instruction": prompt,
                "category": self._categorize_instructions(instruction_types),
                "eval_fn": "ifeval",
                "instruction_types": instruction_types,
                "kwargs": kwargs
            }
            
            test_cases.append(test_case)
            
        return test_cases
    
    def _categorize_instructions(self, instruction_types: List[str]) -> str:
        """
        Categorize IFEval instructions.
        
        Args:
            instruction_types: List of instruction type identifiers
            
        Returns:
            Category string
        """
        categories = {
            "punctuation": "Punctuation Constraints",
            "detectable_format": "Format Requirements",
            "detectable_content": "Content Requirements",
            "length_constraints": "Length Constraints",
            "combination": "Combined Requirements",
            "change_case": "Text Case Requirements"
        }
        
        # Check for specific categories
        for inst_type in instruction_types:
            for key, value in categories.items():
                if key in inst_type:
                    return value
        
        return "General Instruction Following"
    
    def evaluate_response(self, response: str, instruction_types: List[str], kwargs_list: List[Dict]) -> Tuple[float, str]:
        """
        Evaluate a response according to IFEval criteria.
        
        Args:
            response: Model's response
            instruction_types: List of instruction type identifiers
            kwargs_list: List of parameter dictionaries for each instruction type
            
        Returns:
            Tuple of (score, explanation)
        """
        # Initialize score and explanations
        all_passed = True
        explanations = []
        
        # Evaluate each instruction type
        for i, inst_type in enumerate(instruction_types):
            # Get kwargs for this instruction (or empty dict if not available)
            kwargs = kwargs_list[i] if i < len(kwargs_list) else {}
            
            # Check if the instruction was followed
            passed, explanation = self._evaluate_instruction(response, inst_type, kwargs)
            
            if not passed:
                all_passed = False
                
            explanations.append(explanation)
            
        # Calculate final score and combine explanations
        score = 1.0 if all_passed else 0.0
        explanation = "; ".join(explanations)
        
        return score, explanation
    
    def _evaluate_instruction(self, response: str, instruction_type: str, kwargs: Dict) -> Tuple[bool, str]:
        """
        Evaluate a specific instruction type by delegating to the appropriate evaluator.
        
        Args:
            response: Model's response
            instruction_type: Instruction type identifier
            kwargs: Parameters for this instruction
            
        Returns:
            Tuple of (passed, explanation)
        """
        # Find the appropriate evaluator based on the instruction type prefix
        for prefix, evaluator in self.evaluators.items():
            if prefix in instruction_type:
                return evaluator.evaluate(response, instruction_type, kwargs)
        
        # Default case - unknown instruction type
        return False, f"Unknown instruction type: {instruction_type}"
        
    @staticmethod
    def download_ifeval_dataset(output_path: str = "examples/data/ifeval_sample.jsonl") -> str:
        """
        Download a sample of the IFEval dataset.
        
        Args:
            output_path: Path to save the dataset
            
        Returns:
            Path to the downloaded dataset
        """
        import requests
        from pathlib import Path
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # URL for a sample of the IFEval dataset
        # This is a subset of the full dataset for testing purposes
        
        url = "https://raw.githubusercontent.com/google-research/google-research/refs/heads/master/instruction_following_eval/data/input_data.jsonl"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            with open(output_path, 'w') as f:
                f.write(response.text)
                
            logger.info(f"Downloaded IFEval sample dataset to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error downloading IFEval dataset: {str(e)}")
            return ""