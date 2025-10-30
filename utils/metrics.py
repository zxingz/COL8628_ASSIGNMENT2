"""
Evaluation metrics and utilities.
"""

import numpy as np
from typing import List, Dict, Any
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, top_k_accuracy_score

def compute_classification_metrics(predictions: List[int], 
                                 ground_truth: List[int], 
                                 class_names: List[str] = None) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        predictions: List of predicted class indices
        ground_truth: List of ground truth class indices
        class_names: Optional list of class names
        
    Returns:
        Dictionary containing computed metrics
    """
    accuracy = accuracy_score(ground_truth, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        ground_truth, predictions, average='weighted'
    )
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    }
    
    return metrics

def compute_top_k_accuracy(predictions: np.ndarray, 
                          ground_truth: List[int], 
                          k: int = 5) -> float:
    """
    Compute top-k accuracy.
    
    Args:
        predictions: Array of prediction probabilities/scores
        ground_truth: List of ground truth class indices
        k: Value of k for top-k accuracy
        
    Returns:
        Top-k accuracy score
    """
    return float(top_k_accuracy_score(ground_truth, predictions, k=k))

def format_metrics_for_display(metrics: Dict[str, float]) -> str:
    """
    Format metrics dictionary for pretty printing.
    
    Args:
        metrics: Dictionary of metric names and values
        
    Returns:
        Formatted string for display
    """
    formatted_lines = []
    for metric, value in metrics.items():
        formatted_lines.append(f"{metric.replace('_', ' ').title()}: {value:.4f}")
    
    return "\n".join(formatted_lines)