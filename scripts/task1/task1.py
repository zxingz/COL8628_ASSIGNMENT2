#!/usr/bin/env python3
"""
Task 1: Zero-Shot Evaluation Script

This script performs zero-shot evaluation of the model on a specified test set
and reports the required metrics.

Usage:
    python task1.py --test_path <path_to_test_dataset> --prompt <text_prompt> [--model_path <path>] [--output_dir <dir>]

Example:
    python task1.py --test_path data/test.json --prompt "A photo of a {}" --output_dir results/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torchmetrics.detection import MeanAveragePrecision

import pandas as pd

from PIL import Image

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from tqdm import tqdm

from utils.data_utils import DataSet

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Zero-shot evaluation of the model')
    
    # parser.add_argument('--test_path', type=str, required=True,
    #                     help='Path to the test dataset')
    # parser.add_argument('--prompt', type=str, required=True,
    #                     help='Text prompt template for evaluation')
    # parser.add_argument('--model_path', type=str, default=None,
    #                     help='Path to pre-trained model weights')
    # parser.add_argument('--output_dir', type=str, default='results/',
    #                     help='Directory to save evaluation results')
    # parser.add_argument('--batch_size', type=int, default=32,
    #                     help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for computation (cuda/cpu)')
    
    return parser.parse_args()

def load_test_data(test_path):
    """Load test dataset from the specified path."""
    # TODO: Implement data loading logic
    print(f"Loading test data from: {test_path}")
    pass

def evaluate_model(model, test_data, prompt, args):
    """Perform zero-shot evaluation."""
    # TODO: Implement evaluation logic
    print(f"Evaluating model with prompt: {prompt}")
    pass

def compute_metrics(predictions, ground_truth):
    """Compute evaluation metrics."""
    # TODO: Implement metric computation
    metrics = {
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1_score': 0.0
    }
    return metrics

def save_results(metrics, predictions, output_dir):
    """Save evaluation results to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save predictions
    with open(os.path.join(output_dir, 'predictions.json'), 'w') as f:
        json.dump(predictions, f, indent=2)

def load_model(device):
    """Loads the Grounding-DINO model and processor."""
    print("Loading Grounding-DINO model... (This may take a moment)")
    model_id = "IDEA-Research/grounding-dino-base"
    
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    
    print("Model loaded successfully.")
    return model, processor

def main():
    """Main execution function."""
    args = parse_arguments()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading model...")
    model, processor = load_model(device)
    
    # Initialize metric
    # 'xyxy' format is [xmin, ymin, xmax, ymax]
    metric = MeanAveragePrecision(box_format='xyxy').to(device)
    
    # print("=" * 50)
    # print("Task 1: Zero-Shot Evaluation")
    # print("=" * 50)
    # print(f"Test dataset: {args.test_path}")
    # print(f"Prompt template: {args.prompt}")
    # print(f"Output directory: {args.output_dir}")
    
    # # Load test data
    # test_data = load_test_data(args.test_path)
    
    # # Load model
    # # TODO: Implement model loading
    # model = None
    
    # # Perform evaluation
    # predictions = evaluate_model(model, test_data, args.prompt, args)
    
    # # Compute metrics
    # # TODO: Get ground truth labels
    # ground_truth = None
    # metrics = compute_metrics(predictions, ground_truth)
    
    # # Save results
    # save_results(metrics, predictions, args.output_dir)
    
    # print("\nEvaluation Results:")
    # for metric, value in metrics.items():
    #     print(f"{metric}: {value:.4f}")
    
    # print(f"\nResults saved to: {args.output_dir}")

if __name__ == "__main__":
    main()