#!/usr/bin/env python3
"""
Lambda Score Graph Generator

This script generates a PDF line graph showing how score values change as lambda values
increase from 0 to 0.3 in increments of 0.01. The input file is expected to contain
performance data with scores calculated at lambda=0.2.

Usage:
    python lambda_score_graph.py [--input FILENAME]

Arguments:
    --input FILENAME    Path to the input JSON file (default: performance_20250913_113011.json)

Dependencies:
    This script requires numpy and matplotlib. Install them using:
    pip install numpy matplotlib
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_json(file_path):
    """Load and parse a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_score(combination_data, lambda_val):
    """
    Calculate score for a given combination with the specified lambda value.
    
    Since the exact formula is not provided, we'll use a simple model where:
    - At lambda=0.2, the score matches the value in the input file
    - For other lambda values, we'll adjust the score proportionally
    
    This assumes a linear relationship between lambda and score, which may not be accurate
    but serves as a demonstration.
    """
    # Extract relevant metrics from the combination data
    total_throughput = combination_data["total"]["total_throughput_fps"]
    drop_rate = combination_data["derived"]["drop_rate_fps"]
    
    # The score at lambda=0.2 is provided in the file
    original_score = combination_data["score"]
    
    # Calculate a new score based on the lambda value
    # This is a simplified model: score = throughput - lambda * drop_rate
    # We'll normalize it to match the original score at lambda=0.2
    
    # First, reverse-engineer what the score would be at lambda=0
    base_score = original_score + 0.2 * drop_rate
    
    # Then calculate the new score with the provided lambda
    new_score = base_score - lambda_val * drop_rate
    
    return new_score


def generate_lambda_score_graph(data, output_path="lambda_score_graph.pdf"):
    """
    Generate a line graph showing how scores change with lambda values.
    
    Args:
        data: The loaded JSON data
        output_path: Path where the PDF graph will be saved
    """
    # Extract all combinations from the data
    combinations = data["data"]
    
    # Lambda values from 0 to 0.3 with step 0.01
    lambda_values = np.arange(0, 0.31, 0.01)
    
    plt.figure(figsize=(12, 8))
    
    # Set the font to Times New Roman for all text elements
    plt.rcParams['font.family'] = 'Times New Roman'
    
    # Store all scores for each combination at each lambda value
    all_scores = {}
    for combo in combinations:
        combo_name = combo["combination"]
        scores = [calculate_score(combo, lambda_val) for lambda_val in lambda_values]
        all_scores[combo_name] = scores
        
        # Plot the line for this combination (dark gray color, no legend)
        plt.plot(lambda_values, scores, marker='o', markersize=3, color='#444444')
    
    # Highlight the best deployment
    best_deployment = data["best deployment"]
    # plt.title(f"Score vs Lambda (Best Deployment: {best_deployment})")
    # Add x and y axis labels as requested
    plt.xlabel('λ', fontsize=32)
    plt.ylabel('Score', fontsize=32)
    # Increase font size for tick labels
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Find and mark the optimal batch at each lambda value with a blue circle
    best_scores = []
    for i, lambda_val in enumerate(lambda_values):
        # Get scores for all combinations at this lambda value
        lambda_scores = {combo: all_scores[combo][i] for combo in all_scores}
        
        # Find the combination with the highest score
        best_combo = max(lambda_scores, key=lambda_scores.get)
        best_score = lambda_scores[best_combo]
        best_scores.append(best_score)
    
    # Plot the best scores with blue circles and add to legend
    plt.plot(lambda_values, best_scores, marker='o', markersize=8, color='blue', label='Best deployment')
    
    # Add legend to the bottom left with only the blue circles
    plt.legend(loc='lower left', fontsize=26, handles=[plt.Line2D([], [], marker='o', markersize=8, color='blue', label='Best deployment', linestyle='')])
    
    # Save the figure as PDF
    plt.savefig(output_path)
    print(f"Graph saved to {output_path}")
    
    # Also display the graph if running in an interactive environment
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Generate a PDF graph showing score vs lambda values")
    parser.add_argument("--input", default="performance_20250913_113011.json", 
                        help="Input JSON file (default: performance_20250913_113011.json)")
    args = parser.parse_args()
    
    # Resolve the input file path
    input_path = Path("../performance") / args.input if "/" not in args.input else args.input
    
    # Load the data
    try:
        data = load_json(input_path)
        print(f"Loaded data from {input_path}")
    except FileNotFoundError:
        print(f"Error: File {input_path} not found")
        return
    except json.JSONDecodeError:
        print(f"Error: File {input_path} is not a valid JSON file")
        return
    
    # Generate the graph
    output_path = f"lambda_score_graph_{Path(args.input).stem}.pdf"
    generate_lambda_score_graph(data, output_path)


if __name__ == "__main__":
    main()