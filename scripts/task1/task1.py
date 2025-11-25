#!/usr/bin/env python3

#%%
"""
Task 1: Zero-Shot Evaluation Script

This script performs zero-shot evaluation of the model on a specified test set
and reports the required metrics.

Usage:
    python task1.py [--data_root <data_dir>] [--results_dir <results_dir>]
                    [--device {cuda,cpu}] [--box_thresholds <float float ...>]
                    [--text_thresholds <float float ...>] [--prompts <prompt ...>]
                    [--run_mode {inference,results}]

Example:
    python task1.py --results_dir results/task1/train_custom --box_thresholds 0.3 0.5 --text_thresholds 0.5
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


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "task1" / "train"
DEFAULT_THRESHOLDS = [0.3, 0.5, 0.7, 0.9]
DEFAULT_PROMPTS = [
    "A mass with smooth, circumscribed margins is a condition for a benign finding, whereas a mass with spiculated, irregular margins is a condition for malignancy.",
    "The observation is considered malignant if it shows a cluster of fine, pleomorphic microcalcifications, but it is benign if the calcifications are coarse, large, and popcorn-like.",
    "An observation is benign if it's a round, fat-containing, low-density mass; conversely, it is malignant if it's an irregular, high-density mass.",
    "Architectural distortion or skin retraction is a condition for a malignant tumor, while a mass that is stable and unchanged for several years is a benign condition.",
    "A segmental or linear-branching distribution of calcifications indicates malignancy, whereas diffuse, scattered, bilateral calcifications are a benign finding.",
    "The condition is benign if ultrasound confirms a simple cyst, but the condition is suspicious for malignancy if the mass is solid with indistinct margins.",
    "A well-defined or circumscribed mass is observed in benign cases; an indistinct, microlobulated, or obscured mass is observed in malignant cases.",
    "A small, oval mass with a lucent center (fatty hilum) is a benign intramammary lymph node; a new, developing focal asymmetry is a malignant sign.",
    "The presence of benign vascular calcifications (parallel 'tram-tracks') is not suspicious; the presence of clustered, amorphous calcifications is suspicious for malignancy.",
    "A radiolucent (dark) mass, such as a lipoma or oil cyst, is a benign condition, while a hyperdense (white) and spiculated mass is a malignant condition.",
]


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Zero-shot evaluation of the model')

    parser.add_argument('--data_root', type=str, default=str(DEFAULT_DATA_ROOT),
                        help='Root directory containing dataset folders (default: project data folder)')
    parser.add_argument('--results_dir', type=str, default=str(DEFAULT_RESULTS_DIR),
                        help='Directory to save evaluation outputs (default: project results/task1/train)')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda',
                        help='Device to use for computation (default: cuda, falls back to cpu if unavailable)')
    parser.add_argument('--model_id', type=str, default='IDEA-Research/grounding-dino-tiny',
                        help='Hugging Face model identifier to load (default mirrors current script)')
    parser.add_argument('--datasets', nargs='+', default=['A', 'B', 'C'],
                        help='Dataset identifiers to evaluate (default: A B C)')
    parser.add_argument('--test_split', type=str, default='test',
                        help='Split label to use for evaluation samples (default: test)')
    parser.add_argument('--box_thresholds', type=float, nargs='+', default=DEFAULT_THRESHOLDS,
                        help='List of box thresholds to sweep (default mirrors hard-coded list)')
    parser.add_argument('--text_thresholds', type=float, nargs='+', default=DEFAULT_THRESHOLDS,
                        help='List of text thresholds to sweep (default mirrors hard-coded list)')
    parser.add_argument('--prompts', nargs='+', default=DEFAULT_PROMPTS,
                        help='Prompts to evaluate; defaults to built-in mammography prompts')
    parser.add_argument('--run_mode', type=str, choices=['inference', 'results'], default='inference',
                        help='Select whether to run inference (default) or summarize existing results only')

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

def compute_map(predictions_dict, ground_truth_dict):
    """Compute mAP@0.5 for provided predictions and ground truths."""
    metric = MeanAveragePrecision(iou_type='bbox', iou_thresholds=[0.5])

    for image_name, prediction in predictions_dict.items():
        pred_boxes = prediction.get("boxes", []) or []
        pred_scores = prediction.get("scores", []) or []

        pred_boxes_tensor = torch.tensor(pred_boxes, dtype=torch.float32)
        if pred_boxes_tensor.ndim == 1:
            pred_boxes_tensor = pred_boxes_tensor.unsqueeze(0)
        if pred_boxes_tensor.numel() == 0:
            pred_boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)

        pred_scores_tensor = torch.tensor(pred_scores, dtype=torch.float32)
        if pred_scores_tensor.numel() == 0:
            pred_scores_tensor = torch.zeros((0,), dtype=torch.float32)

        pred_labels_tensor = torch.zeros((pred_boxes_tensor.shape[0],), dtype=torch.int64)

        preds = {
            "boxes": pred_boxes_tensor,
            "scores": pred_scores_tensor,
            "labels": pred_labels_tensor,
        }

        gt_boxes = ground_truth_dict.get(image_name, []) or []
        gt_boxes_tensor = torch.tensor(gt_boxes, dtype=torch.float32)
        if gt_boxes_tensor.ndim == 1:
            gt_boxes_tensor = gt_boxes_tensor.unsqueeze(0)
        if gt_boxes_tensor.numel() == 0:
            gt_boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)

        gt_labels_tensor = torch.zeros((gt_boxes_tensor.shape[0],), dtype=torch.int64)

        target = {
            "boxes": gt_boxes_tensor,
            "labels": gt_labels_tensor,
        }

        metric.update([preds], [target])

    computed = metric.compute()
    map50 = computed.get("map_50")
    if map50 is None:
        return 0.0
    return float(map50.cpu().item())

def build_ground_truth_boxes(df: pd.DataFrame):
    """Generate per-image ground-truth box dictionary from dataframe."""
    if df.empty:
        return {}

    coords_series = df.copy()
    coords_series["coords"] = coords_series.apply(
        lambda x: (x["xmin"], x["ymin"], x["xmax"], x["ymax"]), axis=1
    )
    grouped = coords_series.groupby("image_name")["coords"].apply(list).to_dict()
    clean = {k: [box for box in v if not np.any(np.isnan(box))] for k, v in grouped.items()}
    return clean

def run_inference(args, device):
    """Execute the full inference pipeline and persist predictions."""
    data_path = args.data_root
    results_path = Path(args.results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    dataset_objects = {}
    ground_truth_maps = {}
    for dataset_name in args.datasets:
        dataset = DataSet(data_path=data_path, name=dataset_name, label=args.test_split)
        dataset_objects[dataset_name] = dataset
        ground_truth_maps[dataset_name] = build_ground_truth_boxes(dataset.df)

    if not dataset_objects:
        raise ValueError("No datasets provided for evaluation.")

    print("Loading model...")
    model, processor = load_model(device=device, model_id=args.model_id)

    print("=" * 50)
    print("Task 1: Zero-Shot Evaluation (Inference Mode)")
    print("=" * 50)

    for box_threshold in args.box_thresholds:
        for text_threshold in args.text_thresholds:
            for prompt in args.prompts:
                for dataset_name, test_dataset in dataset_objects.items():
                    ground_truth_boxes_dict = ground_truth_maps.get(dataset_name, {})
                    results = evaluate_model(
                        model=model,
                        processor=processor,
                        test_data=test_dataset,
                        prompt=prompt,
                        device=device,
                        box_threshold=box_threshold,
                        text_threshold=text_threshold,
                    )

                    res = {
                        "dataset": dataset_name,
                        "ground_truth": ground_truth_boxes_dict,
                        f"prediction_{dataset_name}": results,
                        "prompt": prompt,
                        "box_threshold": box_threshold,
                        "text_threshold": text_threshold,
                    }

                    with open(results_path / f"{uuid4()}.json", "w") as f:
                        json.dump(res, f, indent=2)

    print(f"Saved inference outputs to {results_path}")

def summarize_results(results_dir, preview_count=5):
    """Summarize previously generated inference artifacts and persist mAP table."""
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Results directory {results_path} does not exist.")
        return

    json_files = sorted(results_path.glob("*.json"))
    if not json_files:
        print(f"No JSON result files found in {results_path}.")
        return

    dataset_counter = collections.Counter()
    threshold_counter = collections.Counter()
    prompt_counter = collections.Counter()
    processed = 0
    summary_rows = []

    for file_path in json_files:
        try:
            with open(file_path, "r") as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        dataset = payload.get("dataset", "unknown")
        prompt = payload.get("prompt")
        box_threshold = payload.get("box_threshold")
        text_threshold = payload.get("text_threshold")
        pred_key = f"prediction_{dataset}"
        predictions_dict = payload.get(pred_key)

        if not isinstance(predictions_dict, dict):
            continue

        ground_truth_dict = payload.get("ground_truth", {})
        map_value = compute_map(predictions_dict, ground_truth_dict)

        summary_rows.append({
            "dataset": dataset,
            "prompt": prompt,
            "box_threshold": box_threshold,
            "text_threshold": text_threshold,
            "map": map_value,
        })

        dataset_counter[dataset] += 1
        combo = (box_threshold, text_threshold)
        threshold_counter[combo] += 1
        if prompt:
            prompt_counter[prompt] += 1

        processed += 1

    if not summary_rows:
        print("No valid result payloads found to summarize.")
        return

    summary_df = pd.DataFrame(summary_rows)
    summary_save_dir = results_path.parent
    summary_save_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_save_dir / "task1_results_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary dataframe to {summary_path}")

    print(f"Loaded {processed} result files from {results_path}.")
    print("Per-dataset counts:")
    for dataset, count in dataset_counter.items():
        print(f"  {dataset}: {count}")

    print("Most common threshold pairs:")
    for (box_t, text_t), count in threshold_counter.most_common(5):
        print(f"  box={box_t}, text={text_t}: {count}")

    if prompt_counter:
        print("Top prompts:")
        for prompt, count in prompt_counter.most_common(3):
            print(f"  {count}x - {prompt[:80]}{'...' if len(prompt) > 80 else ''}")

    print("Sample files:")
    for file_path in json_files[:preview_count]:
        print(f"  {file_path.name}")

def load_model(device, model_id):
    """Loads the Grounding-DINO model and processor."""
    print(f"Loading Grounding-DINO model ({model_id})... (This may take a moment)")

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    print("Model loaded successfully.")
    return model, processor

# %%

def main():
    """Main execution function."""
    args = parse_arguments()

    if args.run_mode == 'results':
        summarize_results(args.results_dir)
        return

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    run_inference(args, device)


# %%
if __name__ == "__main__":
    main()
