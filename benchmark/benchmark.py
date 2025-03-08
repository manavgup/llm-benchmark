#!/usr/bin/env python3
"""
LLM Benchmark core module: Handles test execution and result collection.
"""

import os
import json
import time
import re
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple
from tqdm import tqdm
from dotenv import load_dotenv

# Import custom evaluators
from .evaluators import evaluate_response, normalize_text, exact_match, contains_match, decimal_precision_match
from .ifeval_integration import IFEvalIntegration
from .concurrent_executor import ConcurrentTestExecutor


# Load environment variables from .env file
load_dotenv()

class LLMBenchmark:
    """Core benchmarking tool for evaluating capabilities of various LLMs."""
    
    def __init__(self, 
                 test_cases_path: str = "examples/data/simple_tests.json", 
                 results_dir: str = "results",
                 verbose: bool = True,
                 use_ifeval: bool = False,
                 ifeval_dataset_path: Optional[str] = None):
        """
        Initialize the LLM benchmarking tool.
        
        Args:
            test_cases_path: Path to JSON file containing test cases
            results_dir: Directory to save results
            verbose: Whether to print verbose output
            use_ifeval: Whether to use IFEval integration
            ifeval_dataset_path: Path to IFEval dataset
        """
        self.test_cases_path = test_cases_path
        self.results_dir = results_dir
        self.verbose = verbose
        self.models = {}
        self.use_ifeval = use_ifeval
        self.ifeval_dataset_path = ifeval_dataset_path 
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Create results directory if it doesn't exist
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        # Initialize IFEval if requested
        if self.use_ifeval:
            self.ifeval = IFEvalIntegration(ifeval_dataset_path)
            
            # If no dataset path was provided, try to download a sample
            if not ifeval_dataset_path and self.ifeval.dataset == []:
                sample_path = self.ifeval.download_ifeval_dataset()
                if sample_path:
                    self.ifeval = IFEvalIntegration(sample_path)
        
        # Load test cases
        self._load_test_cases()
    
    def _load_test_cases(self):
        """Load test cases from JSON file or IFEval dataset."""
        # If IFEval is enabled, only use IFEval test cases
        if self.use_ifeval:
            # Initialize IFEval if not already done
            if not hasattr(self, 'ifeval') or not self.ifeval.dataset:
                if not self.ifeval_dataset_path:
                    self.logger.info("No IFEval dataset provided, downloading sample...")
                    self.ifeval_dataset_path = self.ifeval.download_ifeval_dataset()
                    if self.ifeval_dataset_path:
                        self.ifeval = IFEvalIntegration(self.ifeval_dataset_path)
                    else:
                        self.logger.error("Failed to download IFEval sample dataset")
                        self.test_cases = []
                        return

            # Only use IFEval test cases
            self.test_cases = self.ifeval.convert_to_test_cases()
            if self.verbose:
                print(f"Loaded {len(self.test_cases)} IFEval test cases")
            return
        
        # If IFEval is not enabled, load standard test cases
        try:
            with open(self.test_cases_path, 'r') as f:
                self.test_cases = json.load(f)
                
            if self.verbose:
                print(f"Loaded {len(self.test_cases)} test cases from {self.test_cases_path}")
                    
        except FileNotFoundError:
            print(f"Test cases file not found: {self.test_cases_path}")
            self.test_cases = []
    
    def register_model(self, 
                      name: str, 
                      provider: str,
                      model_id: Optional[str] = None,
                      description: str = None):
        """
        Register a model to be tested using the unified LLM client interface.
        
        Args:
            name: Name identifier for the model in reports
            provider: The provider name (anthropic, openai, watsonx, ollama)
            model_id: Specific model ID (if None, uses provider default)
            description: Optional description of the model
        """
        try:
            # Import here to avoid circular imports
            from llm_clients import get_llm_client
            
            # Get a client for this provider/model
            client = get_llm_client(provider, model_id)
            
            # Create a wrapper function that matches the expected interface
            def call_fn(prompt):
                response = client.generate(prompt)
                # Be sure to close the connection if needed
                if hasattr(client, 'close'):
                    client.close()
                return response
            
            self.models[name] = {
                "call_fn": call_fn,
                "provider": provider,
                "model_id": model_id,
                "description": description
            }
            
            if self.verbose:
                print(f"Registered model: {name} ({provider}/{model_id if model_id else 'default'})")
        except Exception as e:
            print(f"Failed to register model {name}: {str(e)}")
    
    def register_custom_model(self, 
                             name: str, 
                             call_fn: Callable[[str], str],
                             description: str = None):
        """
        Register a custom model with a custom calling function.
        
        Args:
            name: Name identifier for the model
            call_fn: Function that takes a prompt and returns a response
            description: Optional description of the model
        """
        self.models[name] = {
            "call_fn": call_fn,
            "provider": "custom",
            "model_id": None,
            "description": description
        }
        if self.verbose:
            print(f"Registered custom model: {name}")
    
    def _evaluate_response(self, response: str, test_case: Dict[str, Any]) -> Tuple[float, str]:
        """
        Evaluate a response against a test case using the appropriate evaluation function.
        
        Args:
            response: Model's response
            test_case: Test case definition
            
        Returns:
            Tuple of (score, explanation)
        """
        # Get evaluation parameters
        eval_type = test_case.get("eval_fn", "exact_match")
        expected = test_case.get("expected_output")
        kwargs = test_case.get("eval_kwargs", {})
        explanation = ""
        
        # IFEval evaluation
        if eval_type == "ifeval" and self.use_ifeval:
            instruction_types = test_case.get("instruction_types", [])
            kwargs_list = test_case.get("kwargs", [{}])
            score, explanation = self.ifeval.evaluate_response(response, instruction_types, kwargs_list)
            return score, explanation
        
        # Handle specific evaluation types with our evaluators
        if eval_type == "exact_match":
            score = exact_match(response, expected)
            explanation = f"Exact match (normalized): {'Passed' if score > 0 else 'Failed'}"
            
        elif eval_type == "contains":
            score = contains_match(response, expected)
            explanation = f"Contains check: {'Passed' if score > 0 else 'Failed'}"
            
        elif eval_type == "one_word":
            from .evaluators import one_word_match
            score = one_word_match(response, expected)
            explanation = f"One word check: {'Passed' if score > 0 else 'Failed'}"
            
        elif eval_type == "decimal_precision":
            decimals = kwargs.get("decimals", 4)
            score = decimal_precision_match(response, expected, decimals)
            explanation = f"Decimal precision check ({decimals} places): {'Passed' if score > 0 else 'Failed'}"
            
        elif eval_type == "number":
            from .evaluators import number_match
            tolerance = kwargs.get("tolerance", 0.0001)
            score = number_match(response, expected, tolerance)
            explanation = f"Number match check (tolerance {tolerance}): {'Passed' if score > 0 else 'Failed'}"
            
        elif eval_type == "no_punctuation":
            from .evaluators import no_punctuation_match
            char = kwargs.get("char", ",")
            score = no_punctuation_match(response, char)
            explanation = f"No punctuation '{char}' check: {'Passed' if score > 0 else 'Failed'}"
            
        elif eval_type == "format":
            from .evaluators import format_match
            pattern = kwargs.get("pattern", "")
            score = format_match(response, pattern)
            explanation = f"Format pattern check: {'Passed' if score > 0 else 'Failed'}"
            
        elif eval_type == "word_count":
            from .evaluators import word_count_match
            count = kwargs.get("count", 0)
            relation = kwargs.get("relation", "exactly")
            score = word_count_match(response, count, relation)
            explanation = f"Word count check ({relation} {count}): {'Passed' if score > 0 else 'Failed'}"
            
        elif eval_type == "case":
            from .evaluators import case_match
            case_type = kwargs.get("case_type", "lowercase")
            score = case_match(response, case_type)
            explanation = f"Case check ({case_type}): {'Passed' if score > 0 else 'Failed'}"
            
        elif eval_type == "placeholder_count":
            from .evaluators import placeholder_count_match
            count = kwargs.get("count", 0)
            relation = kwargs.get("relation", "exactly")
            score = placeholder_count_match(response, count, relation)
            explanation = f"Placeholder count check ({relation} {count}): {'Passed' if score > 0 else 'Failed'}"
            
        else:
            # Use the generic evaluate_response function for other types
            # This will default to exact_match if the eval_type is unknown
            score = evaluate_response(response, expected, eval_type, **kwargs)
            explanation = f"Custom evaluation ({eval_type}): {'Passed' if score > 0 else 'Failed'}"
            
        return score, explanation
    
    def run_tests(self, 
                models: Optional[List[str]] = None, 
                test_indices: Optional[List[int]] = None,
                max_concurrent: int = 1,
                delay_seconds: float = 1.0):
        """
        Run tests on specified models.
        
        Args:
            models: List of model names to test (all if None)
            test_indices: Indices of test cases to run (all if None)
            max_concurrent: Maximum number of concurrent requests
            delay_seconds: Delay between requests to avoid rate limits
        
        Returns:
            Dict containing results of all tests
        """
        # Configure logger for this module
        logger = logging.getLogger(__name__)
        
        # Use all models if none specified
        if models is None:
            models = list(self.models.keys())
        
        # Use all test cases if none specified
        if test_indices is None:
            test_indices = list(range(len(self.test_cases)))
            
        results = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "models_tested": models,
                "test_indices": test_indices,
                "test_file": self.test_cases_path,
                "concurrency": max_concurrent
            },
            "results": []
        }
        
        # Check if we should use concurrent execution
        use_concurrent = max_concurrent > 1
        executor = None
        if use_concurrent:
            logger.info(f"Using concurrent execution with {max_concurrent} workers")
            executor = ConcurrentTestExecutor(max_workers=max_concurrent, delay_seconds=delay_seconds)
        
        # Run tests
        for model_name in models:
            if model_name not in self.models:
                logger.error(f"Model not found: {model_name}")
                continue
                
            model = self.models[model_name]
            
            logger.info(f"\nTesting model: {model_name}")
            if self.verbose:
                print(f"\nTesting model: {model_name}")
            
            # Run tests either concurrently or sequentially
            if use_concurrent and executor:
                model_results = executor.execute_tests(
                    test_cases=self.test_cases,
                    test_indices=test_indices,
                    model_name=model_name,
                    model=model,
                    evaluate_fn=self._evaluate_response
                )
            else:
                # Original sequential execution
                model_results = []
                for idx in tqdm(test_indices, desc=f"Running tests for {model_name}"):
                    if idx >= len(self.test_cases):
                        logger.error(f"Test case index out of range: {idx}")
                        continue
                    
                    test_case = self.test_cases[idx]
                    
                    # Get the prompt based on whether it's an instruction test or regular prompt
                    if "instruction" in test_case:
                        prompt = test_case["instruction"]
                    else:
                        prompt = test_case["prompt"]
                    
                    expected = test_case.get("expected_output", None)
                    category = test_case.get("category", "Uncategorized")
                    
                    # Log the test case being executed
                    logger.info(f"Test {idx} [{category}]: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
                    
                    try:
                        start_time = time.time()
                        response = model["call_fn"](prompt)
                        end_time = time.time()
                        
                        result = {
                            "test_id": idx,
                            "category": category,
                            "prompt": prompt,
                            "response": response,
                            "expected": expected,
                            "latency_seconds": end_time - start_time,
                            "error": None
                        }
                        
                        # Log the response
                        logger.info(f"Response from {model_name} (Test {idx}) in {end_time - start_time:.2f}s:")
                        logger.info("-" * 40)
                        # Log first 500 chars of response to keep logs manageable
                        logger.info(f"{response[:500]}{'...' if len(response) > 500 else ''}")
                        logger.info("-" * 40)
                        
                        # Evaluate the response
                        if "eval_fn" in test_case:
                            score, explanation = self._evaluate_response(response, test_case)
                            result["score"] = score
                            result["explanation"] = explanation
                            logger.info(f"Score: {score} ({explanation})")
                        else:
                            # No eval function specified
                            result["score"] = None
                            logger.info("Score: no evaluation function specified")
                            
                    except Exception as e:
                        result = {
                            "test_id": idx,
                            "category": category,
                            "prompt": prompt,
                            "response": None,
                            "expected": expected,
                            "latency_seconds": time.time() - start_time,
                            "error": str(e),
                            "score": 0.0
                        }
                        
                        # Log the error
                        logger.error(f"Error in test {idx} with model {model_name}: {str(e)}")
                    
                    model_results.append(result)
                    time.sleep(delay_seconds)  # Delay to avoid rate limits

            results["results"].append({
                "model": model_name,
                "provider": model["provider"],
                "model_id": model["model_id"],
                "description": model["description"],
                "test_results": model_results
            })
        
        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(self.results_dir, f"benchmark_results_{timestamp}.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"\nResults saved to {results_path}")
        if self.verbose:
            print(f"\nResults saved to {results_path}")
            
        return results