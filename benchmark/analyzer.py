#!/usr/bin/env python3
"""
LLM Benchmark analyzer module: Processes benchmark results and generates statistics.
"""

import os
import json
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class BenchmarkAnalyzer:
    """Analyzes benchmark results and generates statistics."""
    
    def __init__(self, results_dir: str = "results"):
        """
        Initialize the benchmark analyzer.
        
        Args:
            results_dir: Directory containing benchmark results
        """
        self.results_dir = results_dir
        
    def find_latest_results(self) -> Optional[str]:
        """Find the path to the latest results file."""
        result_files = [f for f in os.listdir(self.results_dir) 
                      if f.startswith("benchmark_results_")]
        if not result_files:
            logger.warning("No results files found")
            return None
        
        return os.path.join(self.results_dir, sorted(result_files)[-1])
    
    def load_results(self, results_path: Optional[str] = None) -> Dict:
        """
        Load benchmark results from a file.
        
        Args:
            results_path: Path to results JSON file (uses latest if None)
            
        Returns:
            Dict containing the loaded results
        """
        if results_path is None:
            results_path = self.find_latest_results()
            if results_path is None:
                raise FileNotFoundError("No benchmark results files found")
        
        logger.info(f"Loading results from {results_path}")
        with open(results_path, 'r') as f:
            return json.load(f)
    
    def create_summary_dataframe(self, results: Dict) -> pd.DataFrame:
        """
        Create a summary DataFrame from benchmark results.
        
        Args:
            results: Dict containing benchmark results
            
        Returns:
            DataFrame with summary statistics
        """
        rows = []
        for model_results in results["results"]:
            model_name = model_results["model"]
            
            # Overall model metrics
            total_tests = len(model_results["test_results"])
            error_count = sum(1 for r in model_results["test_results"] if r["error"] is not None)
            
            # Only calculate metrics for tests with scores
            scored_tests = [r for r in model_results["test_results"] if r["score"] is not None]
            if scored_tests:
                avg_score = sum(r["score"] for r in scored_tests) / len(scored_tests)
                avg_latency = sum(r["latency_seconds"] for r in scored_tests) / len(scored_tests)
            else:
                avg_score = float('nan')
                avg_latency = float('nan')
                
            rows.append({
                "Model": model_name,
                "Provider": model_results.get("provider", "Unknown"),
                "Model ID": model_results.get("model_id", "Unknown"),
                "Description": model_results["description"],
                "Total Tests": total_tests,
                "Errors": error_count,
                "Average Score": avg_score if not np.isnan(avg_score) else None,
                "Average Latency (s)": avg_latency if not np.isnan(avg_latency) else None
            })
            
        return pd.DataFrame(rows)
    
    def create_detailed_dataframe(self, results: Dict) -> pd.DataFrame:
        """
        Create a detailed DataFrame from benchmark results.
        
        Args:
            results: Dict containing benchmark results
            
        Returns:
            DataFrame with detailed test results
        """
        detailed_rows = []
        for model_results in results["results"]:
            model_name = model_results["model"]
            
            for test_result in model_results["test_results"]:
                test_idx = test_result["test_id"]
                category = test_result.get("category", "Uncategorized")
                
                detailed_rows.append({
                    "Model": model_name,
                    "Provider": model_results.get("provider", "Unknown"),
                    "Test ID": test_idx,
                    "Category": category,
                    "Prompt": test_result["prompt"][:50] + "..." if len(test_result["prompt"]) > 50 else test_result["prompt"],
                    "Score": test_result["score"],
                    "Latency (s)": test_result["latency_seconds"],
                    "Error": test_result["error"] is not None
                })
        
        return pd.DataFrame(detailed_rows)
    
    def create_category_dataframe(self, detailed_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Create a category performance DataFrame from detailed results.
        
        Args:
            detailed_df: DataFrame with detailed test results
            
        Returns:
            DataFrame with category performance statistics or None if categories not available
        """
        if "Category" not in detailed_df.columns:
            logger.warning("No category information found in results")
            return None
            
        category_rows = []
        for model_name in detailed_df["Model"].unique():
            model_df = detailed_df[detailed_df["Model"] == model_name]
            
            for category in model_df["Category"].unique():
                category_df = model_df[model_df["Category"] == category]
                scored_df = category_df[category_df["Score"].notna()]
                
                if len(scored_df) > 0:
                    avg_score = scored_df["Score"].mean()
                    avg_latency = scored_df["Latency (s)"].mean()
                else:
                    avg_score = None
                    avg_latency = None
                    
                category_rows.append({
                    "Model": model_name,
                    "Provider": model_df["Provider"].iloc[0],
                    "Category": category,
                    "Test Count": len(category_df),
                    "Average Score": avg_score,
                    "Average Latency (s)": avg_latency,
                    "Error Count": sum(category_df["Error"])
                })
        
        return pd.DataFrame(category_rows)
    
    def analyze(self, results_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Analyze benchmark results and generate statistics.
        
        Args:
            results_path: Path to results JSON file (uses latest if None)
            
        Returns:
            Tuple of DataFrames: (summary_df, detailed_df, category_df)
        """
        # Load results
        results = self.load_results(results_path)
        
        # Create DataFrames
        summary_df = self.create_summary_dataframe(results)
        detailed_df = self.create_detailed_dataframe(results)
        category_df = self.create_category_dataframe(detailed_df)
        
        return summary_df, detailed_df, category_df