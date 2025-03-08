#!/usr/bin/env python3
"""
LLM Benchmark CLI: Command line interface for the LLM benchmarking tool.
"""

import os
import sys
import argparse
import logging
from typing import List, Optional
from datetime import datetime

# Configure logging
# Create output directory for logs
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set up both console and file logging
log_file = os.path.join(OUTPUT_DIR, f'llm_benchmark_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)
logger = logging.getLogger(__name__)

def filter_models(available_models, filter_type):
    """
    Filter models based on the specified filter type.
    
    Args:
        available_models: Dictionary of available models from get_available_models()
        filter_type: Type of filter to apply ('instruct', 'chat', 'stable', 'all')
        
    Returns:
        Dictionary of filtered models
    """
    if filter_type is None or filter_type == 'all':
        return available_models
    
    filtered_models = {}
    
    for key, model_info in available_models.items():
        model_id = model_info.model_id.lower()
        description = model_info.description.lower() if model_info.description else ""
        
        if filter_type == 'instruct':
            # Include models explicitly designed for instruction following
            if ('instruct' in model_id and 'code' not in model_id and 'vision' not in model_id):
                filtered_models[key] = model_info
                
        elif filter_type == 'chat':
            # Include models designed for chat/conversation
            if ('chat' in model_id or 'chat' in description or
                'conversation' in model_id or 'conversation' in description):
                filtered_models[key] = model_info
                
        elif filter_type == 'stable':
            # Exclude preview/experimental models
            if not any(x in model_id for x in ['preview', 'experimental', 'beta']):
                # For OpenAI, include only specific stable models
                if model_info.provider == 'openai':
                    stable_openai = [
                        'gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo', 
                        'gpt-3.5-turbo-instruct'
                    ]
                    if any(model_id.startswith(m) for m in stable_openai):
                        filtered_models[key] = model_info
                # For Anthropic, include only released models
                elif model_info.provider == 'anthropic':
                    filtered_models[key] = model_info
                # For IBM WatsonX, include only non-preview models
                elif model_info.provider == 'watsonx':
                    filtered_models[key] = model_info
                # For Ollama, include most models as they're locally run
                elif model_info.provider == 'ollama':
                    filtered_models[key] = model_info
    
    return filtered_models

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark and compare LLM performance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--test-file", 
        type=str, 
        default="examples/data/simple_tests.json",
        help="JSON file containing test cases"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="results",
        help="Directory to save results and visualizations"
    )
    
    parser.add_argument(
        "--providers", 
        type=str, 
        nargs="+", 
        default=["anthropic", "openai", "watsonx"],
        help="List of providers to test (anthropic, openai, watsonx, ollama)"
    )
    
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Specific models to test (e.g., claude-3-opus gpt-4-turbo ibm-granite-3-8b-instruct)"
    )
    
    parser.add_argument(
        "--model-filter",
        type=str,
        default=None,
        choices=["instruct", "chat", "stable", "all"],
        help="Filter models by type: 'instruct' for instruction fine-tuned models, 'chat' for chat-optimized models, 'stable' for production-ready models, 'all' for all available models"
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models and exit"
    )
    
    parser.add_argument(
        "--delay", 
        type=float, 
        default=1.0,
        help="Delay between requests in seconds"
    )
    
    parser.add_argument(
        "--analyze-only", 
        action="store_true",
        help="Only analyze the latest results without running new tests"
    )
    
    parser.add_argument(
        "--results-file", 
        type=str, 
        default=None,
        help="Specific results file to analyze (used with --analyze-only)"
    )
    
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots (faster analysis)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    # Add predefined model sets
    parser.add_argument(
        "--model-set",
        type=str,
        choices=["all-instruct", "anthropic", "openai", "watsonx-instruct", "watsonx-llama", "watsonx-mistral"],
        help="Use a predefined set of models (overrides --models and --model-filter)"
    )
    
    # Add option for log file
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Custom log file path (if not specified, logs go to output directory)"
    )
    
    # Add IFEval integration options
    parser.add_argument(
        "--use-ifeval",
        action="store_true",
        help="Use only IFEval test cases for instruction following evaluation (ignores --test-file)"
    )
    
    parser.add_argument(
        "--ifeval-dataset",
        type=str,
        default=None,
        help="Path to IFEval dataset file (will download sample if not provided)"
    )
    
    parser.add_argument(
        "--ifeval-results-dir",
        type=str,
        default="results/ifeval",
        help="Directory to save IFEval results and visualizations"
    )

    parser.add_argument(
        "--max-concurrent", 
        type=int, 
        default=1,
        help="Maximum number of concurrent test requests (set to >1 to enable concurrent execution)"
    )
    
    return parser.parse_args()

def register_providers(benchmark, providers: List[str], models: Optional[List[str]] = None, 
                       model_filter: Optional[str] = None, model_set: Optional[str] = None,
                       verbose: bool = False) -> bool:
    """
    Register models from specified providers.
    
    Args:
        benchmark: LLMBenchmark instance
        providers: List of provider names to register
        models: List of specific models to register (all available if None)
        model_filter: Type of filter to apply ('instruct', 'chat', 'stable', 'all')
        model_set: Predefined set of models to use (overrides models and model_filter)
        verbose: Whether to log detailed information
        
    Returns:
        Boolean indicating if any models were successfully registered
    """
    from llm_clients import get_available_models
    
    models_registered = False
    
    # Handle predefined model sets
    if model_set:
        models = get_models_for_set(model_set)
        model_filter = None  # Model set overrides filter
        if verbose:
            logger.info(f"Using predefined model set '{model_set}' with {len(models)} models")
    
    # Process each provider
    for provider in providers:
        try:
            # Get available models for this provider
            available_models = get_available_models(provider)
            
            if not available_models:
                logger.warning(f"No models available for provider: {provider}")
                continue
            
            # Apply model filter if specified
            if model_filter:
                available_models = filter_models(available_models, model_filter)
                if verbose:
                    logger.info(f"Applied '{model_filter}' filter to {provider} models, {len(available_models)} models remaining")
            
            # Filter to specific models if requested
            if models:
                # For Watson X, clean up the model keys by removing provider prefix
                if provider == "watsonx":
                    provider_model_keys = []
                    for m in models:
                        # Try both with and without ibm- prefix
                        if m in available_models:
                            provider_model_keys.append(m)
                        # Check if the model is specified with the provider prefix
                        elif m.replace("ibm-", "").replace("meta-llama-", "").replace("mistralai-", "") in available_models:
                            cleaned_key = m.replace("ibm-", "").replace("meta-llama-", "").replace("mistralai-", "")
                            provider_model_keys.append(cleaned_key)
                    model_keys = provider_model_keys
                else:
                    model_keys = [k for k in available_models.keys() if k in models]
            else:
                model_keys = list(available_models.keys())
                
            if not model_keys:
                if models:
                    logger.warning(f"None of the specified models are available for {provider}")
                else:
                    logger.warning(f"No models available for {provider} after filtering")
                continue
            
            # Register each model
            for model_key in model_keys:
                model_info = available_models[model_key]
                try:
                    benchmark.register_model(
                        name=model_key,
                        provider=provider,
                        model_id=model_info.model_id,
                        description=model_info.description
                    )
                    models_registered = True
                    if verbose:
                        logger.info(f"Registered {provider} model: {model_key}")
                except Exception as e:
                    logger.error(f"Failed to register {provider}/{model_key}: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing provider {provider}: {e}")
    
    return models_registered

def get_models_for_set(model_set: str) -> List[str]:
    """
    Get a list of models for a predefined model set.
    
    Args:
        model_set: The name of the model set
        
    Returns:
        List of model identifiers
    """
    if model_set == "all-instruct":
        # All instruction-following models from all providers
        return [
            # Anthropic
            "claude-3-opus", "claude-3-sonnet", "claude-3-haiku",
            # OpenAI
            "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo",
            # WatsonX - IBM Granite
            "ibm-granite-3-2-8b-instruct", "ibm-granite-3-8b-instruct", "ibm-granite-34b-code-instruct",
            # WatsonX - LLaMA
            "meta-llama-llama-3-1-70b-instruct", "meta-llama-llama-3-1-8b-instruct", 
            "meta-llama-llama-3-2-1b-instruct", "meta-llama-llama-3-3-70b-instruct", "meta-llama-llama-3-405b-instruct",
            # WatsonX - Mistral
            "mistralai-mistral-large", "mistralai-mixtral-8x7b-instruct-v01"
        ]
    elif model_set == "anthropic":
        # All Anthropic Claude models
        return ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku", "claude-2"]
    elif model_set == "openai":
        # OpenAI chat models
        return ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
    elif model_set == "watsonx-instruct":
        # IBM Granite instruction models
        return [
            "ibm-granite-3-2-8b-instruct", "ibm-granite-3-8b-instruct", "ibm-granite-34b-code-instruct"
        ]
    elif model_set == "watsonx-llama":
        # Meta LLaMA models via WatsonX
        return [
            "meta-llama-llama-3-1-70b-instruct", "meta-llama-llama-3-1-8b-instruct", 
            "meta-llama-llama-3-2-1b-instruct", "meta-llama-llama-3-3-70b-instruct", "meta-llama-llama-3-405b-instruct"
        ]
    elif model_set == "watsonx-mistral":
        # Mistral models via WatsonX
        return ["mistralai-mistral-large", "mistralai-mixtral-8x7b-instruct-v01"]
    else:
        # Default to empty list if unknown set
        logger.warning(f"Unknown model set: {model_set}")
        return []

def main():
    """Run the LLM benchmark CLI."""
    args = parse_arguments()
    
    # Configure custom log file if specified
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        logger.info(f"Logging to custom file: {args.log_file}")
    else:
        logger.info(f"Logging to default file: {log_file}")
    
    # Ensure parent directories are in the Python path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    # Import client library
    from llm_clients import list_available_models
    
    # If --list-models is specified, print models and exit
    if args.list_models:
        print(list_available_models())
        return 0
    
    # Import other modules
    from benchmark.benchmark import LLMBenchmark
    from benchmark.analyzer import BenchmarkAnalyzer
    from benchmark.visualizer import BenchmarkVisualizer
    
    # Determine results directory
    results_dir = args.ifeval_results_dir if args.use_ifeval else args.output_dir
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize components
    benchmark = LLMBenchmark(
        test_cases_path=args.test_file,
        results_dir=results_dir,
        verbose=args.verbose,
        use_ifeval=args.use_ifeval,
        ifeval_dataset_path=args.ifeval_dataset
    )
    
    analyzer = BenchmarkAnalyzer(results_dir=results_dir)
    visualizer = BenchmarkVisualizer(results_dir=results_dir)
    
    # Log IFEval info if enabled
    if args.use_ifeval:
        logger.info("IFEval integration enabled")
        if args.ifeval_dataset:
            logger.info(f"Using IFEval dataset: {args.ifeval_dataset}")
        else:
            logger.info("No IFEval dataset specified, will attempt to download a sample")
    
    if args.analyze_only:
        # Only analyze results
        logger.info("Analysis-only mode, skipping test execution")
        
        try:
            # Load and analyze results
            summary_df, detailed_df, category_df = analyzer.analyze(args.results_file)
            
            # Print summary
            print("\nSummary Results:")
            print(summary_df)
            
            if category_df is not None:
                print("\nCategory Performance:")
                print(category_df)
            
            # Create visualizations
            if not args.no_plots:
                logger.info("Generating visualizations...")
                plot_paths = visualizer.create_all_plots(summary_df, detailed_df, category_df)
                logger.info(f"Created {len(plot_paths)} plots in {results_dir}")
            
        except Exception as e:
            logger.error(f"Error analyzing results: {e}")
            return 1
        
        return 0
    
    # Register models from specified providers
    models_registered = register_providers(
        benchmark, 
        args.providers, 
        args.models,
        args.model_filter,
        args.model_set,
        args.verbose
    )
    
    if not models_registered:
        logger.error("No models registered. Check API keys and provider availability.")
        return 1
    
    # Run the benchmark
    logger.info("Running benchmark tests...")
    if args.use_ifeval:
        logger.info("Including IFEval instruction following tests")
        
    try:
        results = benchmark.run_tests(
            delay_seconds=args.delay,
            max_concurrent=args.max_concurrent
        )
        
        # Analyze results
        logger.info("Analyzing results...")
        summary_df, detailed_df, category_df = analyzer.analyze()
        
        # Print summary
        print("\nSummary Results:")
        print(summary_df)
        
        if category_df is not None:
            print("\nCategory Performance:")
            print(category_df)
        
        # Create visualizations
        if not args.no_plots:
            logger.info("Generating visualizations...")
            plot_paths = visualizer.create_all_plots(summary_df, detailed_df, category_df)
            logger.info(f"Created {len(plot_paths)} plots in {results_dir}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error running benchmark: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())