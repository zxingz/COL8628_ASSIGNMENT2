"""
Utility functions for data loading and preprocessing.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any

def load_dataset(data_path: str) -> Dict[str, Any]:
    """
    Load dataset from JSON file.
    
    Args:
        data_path: Path to the JSON dataset file
        
    Returns:
        Dictionary containing dataset information
    """
    with open(data_path, 'r') as f:
        dataset = json.load(f)
    return dataset

def save_results(results: Dict[str, Any], output_path: str) -> None:
    """
    Save results to JSON file.
    
    Args:
        results: Dictionary containing results to save
        output_path: Path to save the results
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

def validate_dataset_format(dataset: Dict[str, Any]) -> bool:
    """
    Validate that the dataset follows the expected format.
    
    Args:
        dataset: Dataset dictionary to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_keys = ['images', 'classes']
    
    if not all(key in dataset for key in required_keys):
        return False
    
    # Validate image entries
    for image in dataset['images']:
        if not all(key in image for key in ['id', 'path', 'label']):
            return False
    
    return True