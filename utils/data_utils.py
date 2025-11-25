"""
Utility functions for data loading and preprocessing.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
from glob import glob

import pandas as pd

from PIL import Image

def load_rgb_image(path: Path) -> Image.Image:
    """Load Image"""
    return Image.open(path).convert("RGB")

class DataSet:
    """Dataset class to handle data loading and preprocessing."""

    def __init__(self, data_path: str = 'data', name: str = 'A', label: str = 'train'):
        self.data_path = Path(data_path)
        self.name = str(name).upper()
        self.label = label
        csv_path = self.data_path / f'dataset_{self.name}' / self.label / f'{self.label}.csv'
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset CSV not found at {csv_path}")
        self.df = pd.read_csv(csv_path)
        
    def load_image(self, image_name: str) -> Image.Image:
        """Load an image given its name."""
        image_path = self.data_path / f'dataset_{self.name}' / self.label / image_name
        return load_rgb_image(image_path)
        
if __name__ == "__main__":
    # Example usage
    dataset = DataSet(data_path='data', name='A', label='train')
    print(dataset.df.head())

