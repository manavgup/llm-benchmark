# LLM Benchmark

A comprehensive tool for benchmarking and evaluating large language models, with a focus on instruction following capabilities.

## Features

- **Multi-Provider Support**: Test models from Anthropic, OpenAI, IBM WatsonX, and Ollama
- **Instruction Following Evaluation**: Built-in IFEval integration for standardized instruction following testing
- **Concurrent Execution**: Parallelize test execution for faster benchmarking
- **Comprehensive Analysis**: Generate detailed performance metrics and visualizations
- **Flexible Configuration**: Filter models, customize test cases, and control execution parameters
- **Provider-Specific Optimizations**: Model-specific prompt formatting for each provider

## Installation

1. Clone this repository
```bash
git clone https://github.com/manavgup/llm-benchmark.git
cd llm-benchmark
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with your API keys:
```
ANTHROPIC_API_KEY=your_anthropic_api_key
OPENAI_API_KEY=your_openai_api_key
WATSONX_API_KEY=your_watsonx_api_key
WATSONX_PROJECT_ID=your_watsonx_project_id
WATSONX_URL=https://us-south.ml.cloud.ibm.com
```

## Usage

### Command Line Interface

The benchmark tool provides a full-featured command line interface:

```bash
# Run benchmark with IFEval tests on all providers
python llm_benchmark.py --use-ifeval --verbose

# Run benchmark on specific providers with concurrent execution
python llm_benchmark.py --use-ifeval --providers watsonx --max-concurrent 8

# Filter for instruction-following models
python llm_benchmark.py --use-ifeval --providers watsonx --model-filter instruct

# Use predefined model sets
python llm_benchmark.py --use-ifeval --model-set watsonx-instruct

# List all available models
python llm_benchmark.py --list-models
```

### Key Command Line Options

- `--use-ifeval`: Use IFEval for standardized instruction following evaluation
- `--providers`: Specify which providers to test (anthropic, openai, watsonx, ollama)
- `--models`: Specify specific models to test
- `--model-filter`: Filter models by type (instruct, chat, stable)
- `--model-set`: Use predefined sets of models (anthropic, openai, watsonx-instruct, etc.)
- `--max-concurrent`: Number of concurrent test executions (for faster benchmarking)
- `--analyze-only`: Only analyze existing results without running tests
- `--verbose`: Show detailed output during execution

### Programmatic Usage

```python
from benchmark.benchmark import LLMBenchmark
from benchmark.analyzer import BenchmarkAnalyzer
from benchmark.visualizer import BenchmarkVisualizer

# Initialize benchmark with IFEval
benchmark = LLMBenchmark(
    verbose=True,
    use_ifeval=True,
    results_dir="results/ifeval"
)

# Register specific models
benchmark.register_model(
    name="claude-3-opus",
    provider="anthropic",
    model_id="claude-3-opus-20240229",
    description="Claude 3 Opus by Anthropic"
)

# Run the benchmark with concurrent execution
results = benchmark.run_tests(max_concurrent=8)

# Analyze results
analyzer = BenchmarkAnalyzer(results_dir="results/ifeval")
summary_df, detailed_df, category_df = analyzer.analyze()

# Create visualizations
visualizer = BenchmarkVisualizer(results_dir="results/ifeval")
plot_paths = visualizer.create_all_plots(summary_df, detailed_df, category_df)
```

## Architecture

- **benchmark module**: Core benchmarking logic and framework
  - `benchmark.py`: Main benchmarking functionality
  - `analyzer.py`: Results analysis
  - `visualizer.py`: Visualization generation
  - `evaluators.py`: Response evaluation functions
  - `ifeval_integration.py`: Integration with IFEval
  - `concurrent_executor.py`: Parallel test execution
  - `cli.py`: Command line interface

- **llm_clients module**: Provider-specific client implementations
  - `anthropic.py`, `openai.py`, `watsonx.py`, `ollama.py`: Provider clients
  - `prompt_formatters.py`: Provider-specific prompt formatting
  - `factory.py`: Client factory for easy instantiation

## Extending

### Adding New Models

Register new models using the `register_model` method:

```python
benchmark.register_model(
    name="my-new-model",
    provider="provider_name",
    model_id="specific_model_id",
    description="Description of model"
)
```

### Custom Test Cases

Create custom test cases in JSON format. Example:

```json
[
  {
    "name": "Format Test",
    "instruction": "Respond with your answer in the format: '<<Answer: X>>'",
    "expected_output": "<<Answer:",
    "eval_fn": "contains"
  }
]
```

### Custom Evaluators

Extend the evaluator functions in `evaluators.py` for custom evaluation logic.

## License

This project is licensed under the MIT License - see the LICENSE file for details.