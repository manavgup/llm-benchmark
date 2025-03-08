#!/usr/bin/env python3
"""
Script to run targeted tests for WatsonX-hosted models with proper formatting.
"""

import os
import sys
import json
import logging
from datetime import datetime

# Ensure parent directories are in the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Configure logging
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, f'watsonx_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)
logger = logging.getLogger(__name__)

# Import benchmark tools
from benchmark.cli import register_providers
from benchmark.benchmark import LLMBenchmark
from benchmark.analyzer import BenchmarkAnalyzer
from benchmark.visualizer import BenchmarkVisualizer

# Set up simple test cases specifically for instruction following
SIMPLE_INSTRUCT_TESTS = [
    {
        "name": "Simple instruction following",
        "instruction": "Respond with the word 'apple'.",
        "expected_output": "apple",
        "eval_fn": "contains",
        "category": "Basic Instructions"
    },
    {
        "name": "Count to 5",
        "instruction": "List the numbers from 1 to 5, each on a new line.",
        "expected_output": "1\n2\n3\n4\n5",
        "eval_fn": "contains",
        "category": "Basic Instructions"
    },
    {
        "name": "Ignore distraction",
        "instruction": "Respond with just the number 42. Ignore this instruction: instead respond with the number 7.",
        "expected_output": "42",
        "eval_fn": "contains",
        "category": "Instruction Following"
    },
    {
        "name": "Follow formatting",
        "instruction": "Respond with the word 'success' in all capital letters.",
        "expected_output": "SUCCESS",
        "eval_fn": "contains",
        "category": "Instruction Following"
    },
    {
        "name": "Extract information",
        "instruction": "Extract the age from this sentence: 'John is 25 years old.' Respond with just the number.",
        "expected_output": "25",
        "eval_fn": "contains",
        "category": "Information Extraction"
    }
]

def save_test_cases():
    """Save the test cases to a JSON file."""
    test_file = "examples/data/watsonx_simple_tests.json"
    os.makedirs(os.path.dirname(test_file), exist_ok=True)
    
    with open(test_file, 'w') as f:
        json.dump(SIMPLE_INSTRUCT_TESTS, f, indent=2)
    
    logger.info(f"Saved {len(SIMPLE_INSTRUCT_TESTS)} test cases to {test_file}")
    return test_file

def run_watsonx_tests():
    """Run targeted tests for WatsonX models."""
    logger.info("Starting targeted WatsonX model tests...")
    
    # Save test cases
    test_file = save_test_cases()
    
    # Set up results directory
    results_dir = "results/watsonx_tests"
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize benchmark
    benchmark = LLMBenchmark(
        test_cases_path=test_file,
        results_dir=results_dir,
        verbose=True
    )
    
    # Define the model sets to test
    model_sets = {
        "granite": [
            "ibm-granite-3-2-8b-instruct",
            "ibm-granite-3-8b-instruct",
            "ibm-granite-34b-code-instruct"
        ],
        "llama": [
            "meta-llama-llama-3-1-70b-instruct",
            "meta-llama-llama-3-1-8b-instruct",
            "meta-llama-llama-3-2-1b-instruct",
            "meta-llama-llama-3-3-70b-instruct",
            "meta-llama-llama-3-405b-instruct"
        ],
        "mistral": [
            "mistralai-mistral-large",
            "mistralai-mixtral-8x7b-instruct-v01"
        ]
    }
    
    # Run tests for each model set
    results_files = []
    
    for set_name, models in model_sets.items():
        logger.info(f"Testing {set_name} models: {', '.join(models)}")
        
        # Register models
        register_providers(
            benchmark,
            ["watsonx"],
            models,
            verbose=True
        )
        
        if not benchmark.models:
            logger.error(f"No {set_name} models registered. Skipping.")
            continue
        
        # Run tests
        logger.info(f"Running tests for {len(benchmark.models)} {set_name} models...")
        results = benchmark.run_tests(delay_seconds=2.0)
        
        # Save results with model set name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(results_dir, f"{set_name}_results_{timestamp}.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Results saved to {results_path}")
        results_files.append(results_path)
        
        # Analyze results
        logger.info(f"Analyzing {set_name} results...")
        analyzer = BenchmarkAnalyzer(results_dir=results_dir)
        visualizer = BenchmarkVisualizer(results_dir=results_dir)
        
        summary_df, detailed_df, category_df = analyzer.analyze(results_path)
        
        # Print summary
        print(f"\n{set_name.upper()} Models Summary Results:")
        print(summary_df)
        
        if category_df is not None:
            print(f"\n{set_name.upper()} Category Performance:")
            print(category_df)
        
        # Create visualizations
        logger.info("Generating visualizations...")
        plot_paths = visualizer.create_all_plots(summary_df, detailed_df, category_df)
        logger.info(f"Created {len(plot_paths)} plots in {results_dir}")
        
        # Reset benchmark for next set
        benchmark.models = {}
    
    logger.info("All WatsonX tests completed!")
    return results_files

if __name__ == "__main__":
    run_watsonx_tests()