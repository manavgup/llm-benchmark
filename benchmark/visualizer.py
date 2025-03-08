#!/usr/bin/env python3
"""
LLM Benchmark visualizer module: Creates visualizations from benchmark results.
"""

import os
import time
import logging
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class BenchmarkVisualizer:
    """Creates visualizations from benchmark results."""
    
    def __init__(self, results_dir: str = "results"):
        """
        Initialize the benchmark visualizer.
        
        Args:
            results_dir: Directory to save visualization outputs
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
    def create_timestamp(self) -> str:
        """Create a timestamp string for file naming."""
        return time.strftime("%Y%m%d_%H%M%S")
    
    def plot_model_comparison(self, summary_df: pd.DataFrame) -> Optional[str]:
        """
        Create a comparison plot of model performance.
        
        Args:
            summary_df: DataFrame with summary statistics
            
        Returns:
            Path to the saved plot or None if plotting failed
        """
        if 'Average Score' not in summary_df.columns or summary_df['Average Score'].isna().all():
            logger.warning("No score data available for model comparison plot")
            return None
            
        timestamp = self.create_timestamp()
        output_path = os.path.join(self.results_dir, f'model_comparison_{timestamp}.png')
        
        try:
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            ax = summary_df.plot.bar(x='Model', y='Average Score', ax=plt.gca())
            plt.title('Average Score by Model')
            plt.ylim(0, 1.1)
            plt.xlabel('')
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels
            for i, v in enumerate(summary_df['Average Score']):
                if not pd.isna(v):
                    ax.text(i, v + 0.05, f"{v:.2f}", ha='center')
            
            # Plot average latency by model
            plt.subplot(1, 2, 2)
            ax = summary_df.plot.bar(x='Model', y='Average Latency (s)', ax=plt.gca())
            plt.title('Average Latency by Model')
            plt.xlabel('')
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels
            for i, v in enumerate(summary_df['Average Latency (s)']):
                if not pd.isna(v):
                    ax.text(i, v + 0.05, f"{v:.2f}s", ha='center')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Model comparison plot saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error creating model comparison plot: {str(e)}")
            return None
    
    def plot_test_comparison(self, detailed_df: pd.DataFrame) -> Optional[str]:
        """
        Create a comparison plot of test case performance.
        
        Args:
            detailed_df: DataFrame with detailed test results
            
        Returns:
            Path to the saved plot or None if plotting failed
        """
        if 'Score' not in detailed_df.columns or detailed_df['Score'].isna().all():
            logger.warning("No score data available for test comparison plot")
            return None
            
        timestamp = self.create_timestamp()
        output_path = os.path.join(self.results_dir, f'test_case_comparison_{timestamp}.png')
        
        try:
            # Filter to only rows with scores
            scored_df = detailed_df[detailed_df['Score'].notna()]
            
            # Create pivot table
            pivot_df = scored_df.pivot(index='Test ID', columns='Model', values='Score')
            
            # Plot
            plt.figure(figsize=(14, 8))
            ax = pivot_df.plot(kind='bar', figsize=(14, 8))
            plt.title('Scores by Test Case and Model')
            plt.ylim(0, 1.1)
            plt.xlabel('Test ID')
            plt.ylabel('Score')
            plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Test case comparison plot saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error creating test case comparison plot: {str(e)}")
            return None
    
    def plot_category_heatmap(self, category_df: pd.DataFrame) -> Optional[str]:
        """
        Create a heatmap of model performance by category.
        
        Args:
            category_df: DataFrame with category performance statistics
            
        Returns:
            Path to the saved plot or None if plotting failed
        """
        if category_df is None or 'Average Score' not in category_df.columns:
            logger.warning("No category data available for heatmap")
            return None
            
        timestamp = self.create_timestamp()
        output_path = os.path.join(self.results_dir, f'category_heatmap_{timestamp}.png')
        
        try:
            # Create pivot table for heatmap
            pivot_df = category_df.pivot(index='Category', columns='Model', values='Average Score')
            
            # Plot
            plt.figure(figsize=(14, 10))
            
            # Use seaborn for better heatmap visualization
            ax = sns.heatmap(
                pivot_df, 
                annot=True, 
                cmap='Blues', 
                vmin=0, 
                vmax=1,
                fmt='.2f',
                linewidths=.5,
                cbar_kws={'label': 'Average Score'}
            )
            
            plt.title('Performance by Category and Model', fontsize=16)
            plt.tight_layout()
            
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Category heatmap saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error creating category heatmap: {str(e)}")
            return None
    
    def plot_error_analysis(self, summary_df: pd.DataFrame) -> Optional[str]:
        """
        Create a plot showing error rates by model.
        
        Args:
            summary_df: DataFrame with summary statistics
            
        Returns:
            Path to the saved plot or None if plotting failed
        """
        if 'Errors' not in summary_df.columns or 'Total Tests' not in summary_df.columns:
            logger.warning("No error data available for error analysis plot")
            return None
            
        timestamp = self.create_timestamp()
        output_path = os.path.join(self.results_dir, f'error_analysis_{timestamp}.png')
        
        try:
            # Calculate error rate
            summary_df['Error Rate'] = summary_df['Errors'] / summary_df['Total Tests']
            
            # Plot
            plt.figure(figsize=(10, 6))
            ax = summary_df.plot.bar(x='Model', y='Error Rate', ax=plt.gca(), color='salmon')
            plt.title('Error Rate by Model')
            plt.ylim(0, 1.0)
            plt.xlabel('')
            plt.ylabel('Error Rate')
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels
            for i, v in enumerate(summary_df['Error Rate']):
                if not pd.isna(v):
                    ax.text(i, v + 0.05, f"{v:.2%}", ha='center')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Error analysis plot saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error creating error analysis plot: {str(e)}")
            return None
    
    def plot_latency_distribution(self, detailed_df: pd.DataFrame) -> Optional[str]:
        """
        Create a box plot showing the distribution of latencies by model.
        
        Args:
            detailed_df: DataFrame with detailed test results
            
        Returns:
            Path to the saved plot or None if plotting failed
        """
        if 'Latency (s)' not in detailed_df.columns:
            logger.warning("No latency data available for latency distribution plot")
            return None
            
        timestamp = self.create_timestamp()
        output_path = os.path.join(self.results_dir, f'latency_distribution_{timestamp}.png')
        
        try:
            plt.figure(figsize=(12, 6))
            
            # Create box plot
            ax = sns.boxplot(x='Model', y='Latency (s)', data=detailed_df)
            
            plt.title('Latency Distribution by Model')
            plt.xlabel('')
            plt.ylabel('Latency (seconds)')
            plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Latency distribution plot saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error creating latency distribution plot: {str(e)}")
            return None
    
    def create_all_plots(self, summary_df: pd.DataFrame, detailed_df: pd.DataFrame, 
                        category_df: Optional[pd.DataFrame] = None) -> List[str]:
        """
        Create all available plots from benchmark results.
        
        Args:
            summary_df: DataFrame with summary statistics
            detailed_df: DataFrame with detailed test results
            category_df: DataFrame with category performance statistics
            
        Returns:
            List of paths to saved plots
        """
        plot_paths = []
        
        # Model comparison
        model_plot = self.plot_model_comparison(summary_df)
        if model_plot:
            plot_paths.append(model_plot)
        
        # Test comparison
        test_plot = self.plot_test_comparison(detailed_df)
        if test_plot:
            plot_paths.append(test_plot)
        
        # Category heatmap
        if category_df is not None:
            category_plot = self.plot_category_heatmap(category_df)
            if category_plot:
                plot_paths.append(category_plot)
        
        # Error analysis
        error_plot = self.plot_error_analysis(summary_df)
        if error_plot:
            plot_paths.append(error_plot)
        
        # Latency distribution
        latency_plot = self.plot_latency_distribution(detailed_df)
        if latency_plot:
            plot_paths.append(latency_plot)
        
        return plot_paths