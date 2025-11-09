#!/usr/bin/env python3
"""
Task 2.1: CoOp Training Script

This script trains CoOp (Context Optimization) prompts on labeled datasets.

Usage:
    python train_task2_1.py --train_path <path> --save_path <path> [--epochs <num>] [--lr <rate>]

Example:
    python train_task2_1.py --train_path data/train.json --save_path models/coop_model.pth --epochs 50 --lr 0.002
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(str(Path(__file__).resolve().parent.parent.parent)))
from utils.data_utils import DataSet

import torch
import torch.optim as optim
import torch.nn as nn
from torchmetrics.detection import MeanAveragePrecision

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


import torch
import torch.nn as nn

import torch
import torch.nn as nn

import torch
import torch.nn as nn

import torch
import torch.nn as nn

class GDinoCoop(nn.Module):
    def __init__(self, model, prompt_length=16):
        super().__init__()
        self.model = model
        
        # Freeze the entire base model
        for param in self.model.parameters():
            param.requires_grad = False

        # Access the text backbone directly
        self.text_backbone = self.model.model.text_backbone
        embed_dim = self.model.config.text_config.hidden_size

        # Learnable prompt
        self.prompt_len = prompt_length
        self.prompt = nn.Parameter(torch.randn(1, self.prompt_len, embed_dim))
        self.prompt.requires_grad = True

    def forward(self, **kwargs):
        input_ids = kwargs.pop("input_ids")
        attention_mask = kwargs.pop("attention_mask")
        token_type_ids = kwargs.pop("token_type_ids", None)
        labels = kwargs.pop("labels", None)
        
        device = input_ids.device

        # =========================================
        # 1. Prepare Custom Embeddings (Injection)
        # =========================================
        # Get original embeddings for the input text
        word_embeddings = self.text_backbone.embeddings.word_embeddings(input_ids)
        
        # Expand learnable prompt to batch size
        prompt_embeds = self.prompt.expand(word_embeddings.size(0), -1, -1)
        
        # Concatenate: [Prompt Embeds] + [Original Text Embeds (minus CLS)]
        # We skip the first token ([CLS]) of the original text as the prompt takes its place.
        shifted_word_embeddings = word_embeddings[:, 1:, :]
        inputs_embeds = torch.cat([prompt_embeds, shifted_word_embeddings], dim=1)

        # =========================================
        # 2. Adjust Masks and IDs for new length
        # =========================================
        # Extend attention mask to cover the prompt
        prompt_attention_mask = torch.ones(attention_mask.size(0), self.prompt_len, device=device, dtype=attention_mask.dtype)
        new_attention_mask = torch.cat([prompt_attention_mask, attention_mask[:, 1:]], dim=1)

        # Adjust token_type_ids if they exist (needed for some BERT-like backbones)
        prompt_token_type_ids = torch.zeros(token_type_ids.size(0), self.prompt_len, dtype=token_type_ids.dtype, device=device)
        new_token_type_ids = torch.cat([prompt_token_type_ids, token_type_ids[:, 1:]], dim=1)
            
            
        # =========================================
        # 3. Monkey-Patch embeddings
        # =========================================
        # Save the original
        
        original_word_embeddings = self.text_backbone.embeddings.word_embeddings

        # Define the temporary patch
        def patched_word_embeddings(*args, **kwargs):
            return inputs_embeds

        # Apply the patch
        self.text_backbone.word_embeddings = patched_word_embeddings

        # Create dummy input_ids just to satisfy top-level shape checks.
        # The values don't matter as our patch ignores them.
        dummy_input_ids = torch.zeros(
            (input_ids.shape[0], inputs_embeds.shape[1]), 
            dtype=torch.long, 
            device=device
        )

        # Call the main model
        outputs = self.model(
            input_ids=dummy_input_ids,
            attention_mask=new_attention_mask,
            token_type_ids=new_token_type_ids,
            labels=labels,
            return_dict=True,
            **kwargs # Pass any remaining kwargs (like pixel_values)
        )
        
        # CRITICAL: Always restore the original method
        self.text_backbone.embeddings.word_embeddings = original_word_embeddings
        

        return outputs
    

def load_model(device):
    """Loads the Grounding-DINO model and processor."""
    print("Loading Grounding-DINO model... (This may take a moment)")
    model_id = "IDEA-Research/grounding-dino-tiny"
    
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    
    print("Model loaded successfully.")
    return model, processor

def get_batches(dataset, default_prompt="This is a Mamogram", batch_size=8):
    
    dataset.df = dataset.df.reset_index(drop=True)
    for i in range(0, len(dataset.df), batch_size):
        
        # list of prompts
        batch_prompts = [default_prompt] * batch_size
        
        # list of labels
        batch_labels = dataset.df.iloc[i:i+batch_size]["pathology"]
        
        # list of bounding box
        batch_boxes = dataset.df.iloc[i:i+batch_size] \
                    .apply(lambda x: (x["xmin"], x["ymin"], x["xmax"], x["ymax"]), axis=1) \
                    .tolist()
        
        # list of PIL images
        images = dataset.df.iloc[i:i+batch_size].apply(lambda x: dataset.load_image(x["image_name"]), axis=1).tolist()
        
        yield batch_prompts, batch_labels, batch_boxes, images


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train CoOp prompts on labeled datasets')
    
    # parser.add_argument('--train_path', type=str, required=True,
    #                     help='Path to the training dataset')
    # parser.add_argument('--save_path', type=str, required=True,
    #                     help='Path to save the trained model')
    # parser.add_argument('--epochs', type=int, default=50,
    #                     help='Number of training epochs')
    # parser.add_argument('--lr', type=float, default=0.002,
    #                     help='Learning rate')
    # parser.add_argument('--batch_size', type=int, default=32,
    #                     help='Batch size for training')
    # parser.add_argument('--context_length', type=int, default=16,
    #                     help='Length of context vectors')
    # parser.add_argument('--device', type=str, default='cuda',
    #                     help='Device to use for computation (cuda/cpu)')
    # parser.add_argument('--seed', type=int, default=42,
    #                     help='Random seed for reproducibility')
    
    return parser.parse_args()

def load_training_data(train_path):
    """Load training dataset from the specified path."""
    # TODO: Implement data loading logic
    print(f"Loading training data from: {train_path}")
    pass

def initialize_coop_model(args):
    """Initialize CoOp model with learnable context vectors."""
    # TODO: Implement CoOp model initialization
    print("Initializing CoOp model...")
    pass

def train_model(model, processor, train_data, args):
    """Train the CoOp model."""
    print("Starting CoOp training...")
    device = next(model.parameters()).device

    # The optimizer will only update the learnable prompt, as other parameters are frozen.
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    model.train()

    for epoch in range(args.epochs):
        total_loss = 0
        num_batches = 0

        batch_generator = get_batches(train_data, default_prompt="malignant", batch_size=args.batch_size)

        for batch_prompts, batch_class_labels, batch_boxes, batch_pil_images in batch_generator:
            optimizer.zero_grad()

            # Prepare labels in the format expected by the model
            labels = []
            for boxes, class_labels in zip(batch_boxes, batch_class_labels):
                # The model expects normalized [center_x, center_y, width, height]
                # For now, we assume boxes are for the first class (index 0)
                # and that there are no boxes for benign cases.
                if len(boxes) > 0:
                    labels.append({
                        "class_labels": torch.zeros(len(boxes), dtype=torch.long, device=device),
                        "boxes": torch.tensor(boxes, dtype=torch.float, device=device)
                    })
                else: # Benign cases
                    labels.append({
                        "class_labels": torch.tensor([], dtype=torch.long, device=device),
                        "boxes": torch.tensor([], dtype=torch.float, device=device).reshape(-1, 4)
                    })

            # Process inputs
            inputs = processor(images=batch_pil_images, text=batch_prompts, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Forward pass
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{args.epochs}, Average Loss: {avg_loss:.4f}")

    print("Training completed!")

def save_model(model, save_path):
    """Save the trained model."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # TODO: Implement model saving
    # We only need to save the learnable prompt
    print(f"Saving model to: {save_path}")
    state_dict = {k: v for k, v in model.state_dict().items() if 'prompt' in k}
    torch.save(state_dict, save_path)

def main():
    """Main execution function."""
    args = parse_arguments()
    
    print("=" * 50)
    print("Task 2.1: CoOp Training")
    print("=" * 50)
    # print(f"Training dataset: {args.train_path}")
    # print(f"Save path: {args.save_path}")
    # print(f"Epochs: {args.epochs}")
    # print(f"Learning rate: {args.lr}")
    # print(f"Batch size: {args.batch_size}")
    # print(f"Context length: {args.context_length}")

    # For demonstration, using hardcoded args
    from argparse import Namespace
    args = Namespace(epochs=10, lr=0.001, batch_size=4, context_length=16)
    
    # Set random seed
    torch.manual_seed(10)
    
    # Setup device
    # device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Paths
    data_path = os.path.join(str(Path(__file__).resolve().parent.parent.parent), "data")
    results_path = os.path.join(str(Path(__file__).resolve().parent.parent.parent), "results", "task2-1")
    save_path = os.path.join(str(Path(__file__).resolve().parent.parent.parent), "models", "coop_model.pth")

    # Load model and processor
    base_model, processor = load_model(device)
    
    # Dataset A
    train_A = DataSet(data_path=data_path, name="A", label="train")
    test_A = DataSet(data_path=data_path, name="A", label="test")
    
    # Dataset B
    train_B = DataSet(data_path=data_path, name="B", label="train")
    test_B = DataSet(data_path=data_path, name="B", label="test")
    
    # Dataset C
    train_C = DataSet(data_path=data_path, name="C", label="train")
    test_C = DataSet(data_path=data_path, name="C", label="test")
    
    # Initialize CoOp model
    coop_model = GDinoCoop(base_model, prompt_length=args.context_length).to(device)

    # For this example, we'll train on Dataset A
    print("\nTraining on Dataset A...")
    train_model(coop_model, processor, train_A, args)
    
    # Save the trained model prompt
    save_model(coop_model, save_path)

if __name__ == "__main__":
    main()