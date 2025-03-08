#!/usr/bin/env python3
"""
Advanced example that compares responses from multiple LLM providers
for the same prompt and provides a side-by-side comparison.
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from dotenv import load_dotenv
from tabulate import tabulate
from concurrent.futures import ThreadPoolExecutor

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the LLM clients
from llm_clients import get_llm_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

class ModelComparison:
    """Class to handle comparing multiple LLM models."""
    
    def __init__(self, output_dir="results"):
        """Initialize the comparison tool."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def run_model(self, provider, model_name, prompt, max_tokens=200, temperature=0.7):
        """Run a single model with the given prompt and parameters."""
        try:
            logger.info(f"Running {provider}/{model_name}...")
            start_time = time.time()
            
            client = get_llm_client(provider, model_name)
            response = client.generate(
                prompt, 
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            end_time = time.time()
            elapsed_time = round(end_time - start_time, 2)
            
            # Close connection if needed
            if hasattr(client, 'close'):
                client.close()
            
            logger.info(f"Completed {provider}/{model_name} in {elapsed_time} seconds")
            
            return {
                "provider": provider,
                "model": model_name,
                "response": response,
                "time_seconds": elapsed_time,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error running {provider}/{model_name}: {str(e)}")
            return {
                "provider": provider,
                "model": model_name,
                "response": f"ERROR: {str(e)}",
                "time_seconds": 0,
                "success": False
            }

    def compare_models(self, prompt, models, max_tokens=200, temperature=0.7, parallel=True):
        """
        Compare responses from multiple models for the same prompt.
        
        Args:
            prompt: The text prompt to send to all models
            models: List of (provider, model_name) tuples
            max_tokens: Maximum tokens to generate
            temperature: Temperature setting for generation
            parallel: Whether to run models in parallel
        
        Returns:
            List of response dictionaries
        """
        logger.info(f"Comparing {len(models)} models on prompt: {prompt[:50]}...")
        
        if parallel:
            # Run models in parallel
            with ThreadPoolExecutor(max_workers=len(models)) as executor:
                futures = [
                    executor.submit(
                        self.run_model, 
                        provider, 
                        model_name, 
                        prompt, 
                        max_tokens, 
                        temperature
                    )
                    for provider, model_name in models
                ]
                results = [future.result() for future in futures]
        else:
            # Run models sequentially
            results = []
            for provider, model_name in models:
                result = self.run_model(
                    provider, 
                    model_name, 
                    prompt, 
                    max_tokens, 
                    temperature
                )
                results.append(result)
        
        return results

    def display_results(self, results, format="table"):
        """
        Display comparison results in the specified format.
        
        Args:
            results: List of response dictionaries
            format: Output format ("table", "json")
        """
        if format == "json":
            print(json.dumps(results, indent=2))
        else:
            # Prepare table data
            table_data = []
            for r in results:
                # Truncate response if too long
                response = r["response"]
                if len(response) > 300:
                    response = response[:297] + "..."
                
                # Format status
                status = "✅" if r["success"] else "❌"
                
                table_data.append([
                    f"{r['provider']}/{r['model']}", 
                    response, 
                    f"{r['time_seconds']} sec",
                    status
                ])
            
            # Print table
            print(tabulate(
                table_data, 
                headers=["Model", "Response", "Time", "Status"], 
                tablefmt="grid", 
                maxcolwidths=[30, 50, 10, 5]
            ))

    def save_results(self, test_name, prompt, results):
        """
        Save comparison results to a file.
        
        Args:
            test_name: Name of the test
            prompt: The prompt that was used
            results: List of response dictionaries
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/comparison_{test_name}_{timestamp}.json"
        
        output = {
            "test_name": test_name,
            "prompt": prompt,
            "timestamp": timestamp,
            "results": results
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Results saved to {filename}")
        return filename

def load_prompts(json_file):
    """Load test prompts from a JSON file."""
    try:
        with open(json_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading prompts from {json_file}: {str(e)}")
        return []

def parse_arguments():
    """Parse command line arguments."""
    import argparse
    parser = argparse.ArgumentParser(description="Compare responses from multiple LLM providers")
    
    parser.add_argument("--prompt-file", type=str, default="simple_tests.json",
                        help="JSON file containing test prompts")
    
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory to save results")
    
    parser.add_argument("--max-tokens", type=int, default=200,
                        help="Maximum number of tokens to generate")
    
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for generation")
    
    parser.add_argument("--sequential", action="store_true",
                        help="Run models sequentially instead of in parallel")
    
    parser.add_argument("--format", choices=["table", "json"], default="table",
                        help="Output format")
    
    parser.add_argument("--providers", type=str, nargs="+", 
                        default=["anthropic", "openai", "watsonx", "ollama"],
                        help="List of providers to compare")
    
    return parser.parse_args()

def main():
    """Run the model comparison."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Initialize comparison tool
    comparison = ModelComparison(output_dir=args.output_dir)
    
    # Define models to compare based on selected providers
    models = []
    if "anthropic" in args.providers:
        models.append(("anthropic", "claude-3-sonnet-20240229"))
    if "openai" in args.providers:
        models.append(("openai", "gpt-4-turbo"))
    if "watsonx" in args.providers:
        models.append(("watsonx", "ibm/granite-13b-instruct-v2"))
    if "ollama" in args.providers:
        models.append(("ollama", "llama3"))
    
    # Check if we have any models to test
    if not models:
        logger.error("No valid providers selected for testing")
        return
    
    # Load test prompts
    prompts = load_prompts(args.prompt_file)
    
    if not prompts:
        # Fallback to hardcoded prompts
        prompts = [
            {"category": "Explanation", "prompt": "Explain quantum computing in simple terms"},
            {"category": "Creative", "prompt": "Write a short poem about technology"},
            {"category": "Analysis", "prompt": "What are the pros and cons of remote work?"}
        ]
    
    # Run comparison for each prompt
    for i, prompt_info in enumerate(prompts):
        test_name = prompt_info['category'].lower().replace(" ", "_")
        prompt = prompt_info['prompt']
        
        print(f"\n\n=== Test {i+1}: {prompt_info['category']} ===")
        print(f"Prompt: {prompt}")
        
        results = comparison.compare_models(
            prompt,
            models,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            parallel=not args.sequential
        )
        
        # Display results
        comparison.display_results(results, format=args.format)
        
        # Save results
        comparison.save_results(test_name, prompt, results)
        
        print("\n" + "="*80)
        
        # Short pause between tests
        if i < len(prompts) - 1:
            time.sleep(1)

if __name__ == "__main__":
    main()