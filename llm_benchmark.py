#!/usr/bin/env python3
"""
LLM Benchmark: A comprehensive benchmarking tool for evaluating LLM capabilities.

This script provides a simple entry point to the benchmark CLI.
"""

import sys
from benchmark.cli import main

if __name__ == "__main__":
    sys.exit(main())