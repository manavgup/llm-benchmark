"""
Concurrent test execution module for LLM Benchmark.
Provides efficient parallel execution of test cases.
"""

import time
import logging
import queue
from typing import Dict, List, Any, Optional, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ConcurrentTestExecutor:
    """Handles concurrent execution of benchmark test cases."""
    
    def __init__(self, max_workers: int = 8, delay_seconds: float = 0.2):
        """
        Initialize the concurrent test executor.
        
        Args:
            max_workers: Maximum number of concurrent worker threads
            delay_seconds: Delay between request submissions to avoid rate limits
        """
        self.max_workers = max_workers
        self.delay_seconds = delay_seconds
    
    def execute_tests(self, 
                     test_cases: List[Dict],
                     test_indices: List[int],
                     model_name: str,
                     model: Dict,
                     evaluate_fn: Callable) -> List[Dict]:
        """
        Execute tests concurrently for a specific model.
        
        Args:
            test_cases: List of all test cases
            test_indices: Indices of test cases to run
            model_name: Name of the model being tested
            model: Model configuration dictionary
            evaluate_fn: Function to evaluate responses
            
        Returns:
            List of test results
        """
        # Set up progress bar
        progress_bar = tqdm(total=len(test_indices), desc=f"Running tests for {model_name}")
        
        # Queue to collect results
        results_queue = queue.Queue()
        
        # Function to execute single test case
        def execute_test(idx: int) -> Optional[Tuple[int, Dict]]:
            """Execute a single test case and return the result."""
            if idx >= len(test_cases):
                logger.error(f"Test case index out of range: {idx}")
                return None
            
            test_case = test_cases[idx]
            
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
                    score, explanation = evaluate_fn(response, test_case)
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
            
            return (idx, result)
        
        # Use ThreadPoolExecutor to run tests concurrently
        completed_indices = set()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_idx = {}
            for idx in test_indices:
                if idx >= len(test_cases):
                    continue
                future = executor.submit(execute_test, idx)
                future_to_idx[future] = idx
                
                # Add small delay between submissions to avoid rate limits
                time.sleep(self.delay_seconds / self.max_workers)
            
            # Process results as they complete
            for future in as_completed(future_to_idx):
                result = future.result()
                if result:
                    idx, test_result = result
                    results_queue.put((idx, test_result))
                    completed_indices.add(idx)
                    progress_bar.update(1)
        
        # Close progress bar
        progress_bar.close()
        
        # Get results from queue
        model_results = []
        while not results_queue.empty():
            _, result = results_queue.get()
            model_results.append(result)
        
        # Sort results by test_id for consistency
        model_results.sort(key=lambda x: x["test_id"])
        
        # Check if any tests were skipped
        for idx in test_indices:
            if idx not in completed_indices and idx < len(test_cases):
                logger.warning(f"Test {idx} for model {model_name} was skipped or failed to complete")
        
        return model_results