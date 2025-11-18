#!/usr/bin/env python3

#%%
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
import collections
import json
from uuid import uuid4

import numpy as np

sys.path.insert(0, os.path.join(str(Path(__file__).resolve().parent.parent.parent)))
from utils.data_utils import DataSet

import torch
from torchmetrics.detection import MeanAveragePrecision

import pandas as pd

from PIL import Image

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from tqdm import tqdm


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

def evaluate_model(model, processor, test_data, prompt, device, box_threshold=0.3, text_threshold=0.3):
    """
    Perform zero-shot evaluation.
    
    Args:
        model: The Grounding-DINO model
        processor: The model's processor
        test_data: Dataset containing images and annotations
        prompt: Text prompt template for detection
        device: Device to run inference on
    """
    
    results = dict()
    model.eval()
    
    with torch.no_grad():
        for idx, row in tqdm(test_data.df.iterrows(), total=len(test_data.df), desc="Processing images"):
            # Load and preprocess image
            image = test_data.load_image(row['image_name'])
            
            # Prepare inputs
            inputs = processor(images=image, text=prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Run inference
            outputs = model(**inputs)
            
            # Get image size for post-processing
            target_sizes = torch.tensor([image.size[::-1]], device=device)
            
            processed_outputs = processor.post_process_grounded_object_detection(
                outputs,
                inputs['input_ids'], 
                target_sizes=target_sizes,
                threshold=box_threshold,    # Box threshold name
                text_threshold=text_threshold  # Text threshold name
                )[0]
            
            # Store results
            # image_results = {
            #     'image_id': row['image_name'],
            #     'predictions': {
            #         'boxes': processed_outputs['boxes'].cpu().numpy(),
            #         'scores': processed_outputs['scores'].cpu().numpy(),
            #         'labels': processed_outputs['labels'],
            #     }
            # }
            
            # Store results
            results[row['image_name']] = collections.defaultdict(list)
            results[row['image_name']]["boxes"] += processed_outputs['boxes'].cpu().numpy().tolist()
            results[row['image_name']]["scores"] += processed_outputs['scores'].cpu().numpy().tolist()
    
    return results

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
    # model_id = "IDEA-Research/grounding-dino-base"
    model_id = "IDEA-Research/grounding-dino-tiny"
    
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    
    print("Model loaded successfully.")
    return model, processor

# %%

def main():
    """Main execution function."""
    # args = parse_arguments()
    
    # Setup device
    # device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # List prompts
    mammography_prompts = [
        # Prompt 1: Margins
        "A mass with smooth, circumscribed margins is a condition for a benign finding, whereas a mass with spiculated, irregular margins is a condition for malignancy.",
        
        # Prompt 2: Calcifications (Shape)
        "The observation is considered malignant if it shows a cluster of fine, pleomorphic microcalcifications, but it is benign if the calcifications are coarse, large, and popcorn-like.",
        
        # Prompt 3: Density & Shape
        "An observation is benign if it's a round, fat-containing, low-density mass; conversely, it is malignant if it's an irregular, high-density mass.",
        
        # Prompt 4: Associated Features
        "Architectural distortion or skin retraction is a condition for a malignant tumor, while a mass that is stable and unchanged for several years is a benign condition.",
        
        # Prompt 5: Calcifications (Distribution)
        "A segmental or linear-branching distribution of calcifications indicates malignancy, whereas diffuse, scattered, bilateral calcifications are a benign finding.",
        
        # Prompt 6: Specific Entities
        "The condition is benign if ultrasound confirms a simple cyst, but the condition is suspicious for malignancy if the mass is solid with indistinct margins.",
        
        # Prompt 7: Margins (Variation)
        "A well-defined or circumscribed mass is observed in benign cases; an indistinct, microlobulated, or obscured mass is observed in malignant cases.",
        
        # Prompt 8: Specific Benign vs. Malignant Feature
        "A small, oval mass with a lucent center (fatty hilum) is a benign intramammary lymph node; a new, developing focal asymmetry is a malignant sign.",
        
        # Prompt 9: Calcifications (Type)
        "The presence of benign vascular calcifications (parallel 'tram-tracks') is not suspicious; the presence of clustered, amorphous calcifications is suspicious for malignancy.",
        
        # Prompt 10: Density (Variation)
        "A radiolucent (dark) mass, such as a lipoma or oil cyst, is a benign condition, while a hyperdense (white) and spiculated mass is a malignant condition."
    ]
    
    # Load datasets
    data_path = os.path.join(str(Path(__file__).resolve().parent.parent.parent), "data")
    results_path = os.path.join(str(Path(__file__).resolve().parent.parent.parent), "results", "task1")
    
    # Dataset A
    test_A = DataSet(data_path=data_path, name="A", label="test")
    
    # Dataset B
    test_B = DataSet(data_path=data_path, name="B", label="test")
    
    # Dataset C
    test_C = DataSet(data_path=data_path, name="C", label="test")
    
    # Ground truth bounding boxes dictionary. Key is image name and value is list of boxes
    # empty list for benign cases and boxe(s) for malign cases
    df = pd.concat([test_A.df, test_B.df, test_C.df], ignore_index=True) \
                .reset_index(drop=True)
    df["coords"] = df.apply(lambda x: (x["xmin"], x["ymin"], x["xmax"], x["ymax"]), axis=1)
    ground_truth_boxes_dict = df.groupby("image_name")["coords"].apply(set).to_dict()
    ground_truth_boxes_dict = {k: [l for l in v if not np.any(np.isnan(l))] for k, v in ground_truth_boxes_dict.items()}
    
    # Load model and processor
    print("Loading model...")
    model, processor = load_model(device)
    
    print("=" * 50)
    print("Task 1: Zero-Shot Evaluation")
    print("=" * 50)
    
    thresholds = [0.3, 0.5, 0.7, 0.9]
    
    for box_threshold in thresholds:
        for text_threshold in thresholds:
            for prompt in mammography_prompts:
                # Run evaluation
                results_A = evaluate_model(
                    model=model,
                    processor=processor,
                    test_data=test_A,
                    prompt=prompt,
                    device=device,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                )
                
                res = {
                    "dataset": "A",
                    "ground_truth": ground_truth_boxes_dict,
                    "prediction_A": results_A,
                    "prompt": prompt,
                    "box_threshold": box_threshold,
                    "text_threshold": text_threshold,
                }
                
                with open(os.path.join(results_path, str(uuid4()) + ".json"), "w") as f:
                    json.dump(res, f, indent=2)
                
                results_B = evaluate_model(
                    model=model,
                    processor=processor,
                    test_data=test_B,
                    prompt=prompt,
                    device=device,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                )
                
                res = {
                    "dataset": "B",
                    "ground_truth": ground_truth_boxes_dict,
                    "prediction_B": results_B,
                    "prompt": prompt,
                    "box_threshold": box_threshold,
                    "text_threshold": text_threshold,
                }
                
                with open(os.path.join(results_path, str(uuid4()) + ".json"), "w") as f:
                    json.dump(res, f, indent=2)
                
                results_C = evaluate_model(
                    model=model,
                    processor=processor,
                    test_data=test_C,
                    prompt=prompt,
                    device=device,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                )
                
                res = {
                    "dataset": "C",
                    "ground_truth": ground_truth_boxes_dict,
                    "prediction_C": results_C,
                    "prompt": prompt,
                    "box_threshold": box_threshold,
                    "text_threshold": text_threshold,
                }
                
                with open(os.path.join(results_path, str(uuid4()) + "___.json"), "w") as f:
                    json.dump(res, f, indent=2)


# %%
main()

# if __name__ == "__main__":
#     main()

# %%
