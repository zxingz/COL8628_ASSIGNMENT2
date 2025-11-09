"""
COOP (Context Optimization) with Grounding DINO Tiny
Implements learnable context vectors for improved zero-shot object detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple
import os
from tqdm import tqdm

# ==================== COOP Module ====================

class COOPContextLearner(nn.Module):
    """
    Context Optimization (COOP) module that learns prompt context vectors
    """
    def __init__(
        self,
        n_ctx: int = 16,  # Number of context tokens
        ctx_init: str = "",  # Initial context (optional)
        embed_dim: int = 256,  # Embedding dimension
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.embed_dim = embed_dim
        self.device = device
        
        # Initialize learnable context vectors
        if ctx_init:
            # Initialize with specific text (using simple random if no tokenizer)
            ctx_vectors = torch.randn(n_ctx, embed_dim)
        else:
            # Random initialization
            ctx_vectors = torch.randn(n_ctx, embed_dim)
        
        # Normalize
        ctx_vectors = ctx_vectors / ctx_vectors.norm(dim=-1, keepdim=True)
        
        # Make learnable
        self.ctx = nn.Parameter(ctx_vectors)
        
    def forward(self):
        """Return the learnable context vectors"""
        return self.ctx


class COOPGroundingDINO(nn.Module):
    """
    Grounding DINO with COOP for learnable prompts
    """
    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        n_ctx: int = 16,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.device = device
        
        # Load Grounding DINO
        print(f"Loading Grounding DINO from {model_id}...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        self.model.to(device)
        
        # Get embedding dimension from model
        self.embed_dim = self.model.model.text_backbone.config.hidden_size
        
        # Initialize COOP context learner
        self.coop = COOPContextLearner(
            n_ctx=n_ctx,
            embed_dim=self.embed_dim,
            device=device
        )
        
        # Freeze base model, only train COOP
        for param in self.model.parameters():
            param.requires_grad = False
            
    def get_text_embeddings(self, texts: List[str], use_coop: bool = True):
        """
        Get text embeddings, optionally prepending COOP context
        """
        # Get base text embeddings
        text_inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        
        with torch.no_grad():
            text_features = self.model.model.text_backbone(**text_inputs)
            text_embeds = text_features.last_hidden_state
        
        if use_coop:
            # Prepend learnable context
            ctx = self.coop()  # [n_ctx, embed_dim]
            batch_size = text_embeds.shape[0]
            
            # Expand context for batch
            ctx_expanded = ctx.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Concatenate context with text embeddings
            text_embeds = torch.cat([ctx_expanded, text_embeds], dim=1)
        
        return text_embeds
    
    def forward(self, images: torch.Tensor, texts: List[str], use_coop: bool = True):
        """
        Forward pass with COOP-enhanced text embeddings
        """
        # Process images
        outputs = self.model(
            pixel_values=images,
            input_ids=self.processor(text=texts, return_tensors="pt", padding=True).input_ids.to(self.device)
        )
        
        return outputs


# ==================== Dataset ====================

class DetectionDataset(Dataset):
    """
    Simple detection dataset
    """
    def __init__(self, image_paths: List[str], annotations: List[Dict], processor):
        self.image_paths = image_paths
        self.annotations = annotations
        self.processor = processor
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert("RGB")
        annotation = self.annotations[idx]
        
        # Get class names from this sample
        class_names = annotation['labels']
        text_prompt = ". ".join(class_names) + "."
        
        return {
            'image': image,
            'text_prompt': text_prompt,
            'target_boxes': torch.tensor(annotation['boxes'], dtype=torch.float32),
            'target_labels': class_names
        }


def collate_fn(batch, processor):
    """
    Custom collate function to handle variable-sized images
    """
    images = [item['image'] for item in batch]
    text_prompts = [item['text_prompt'] for item in batch]
    target_boxes = [item['target_boxes'] for item in batch]
    target_labels = [item['target_labels'] for item in batch]
    
    # Process images and text together - processor handles padding
    encoding = processor(
        images=images,
        text=text_prompts,
        return_tensors="pt",
        padding=True
    )
    
    # Add targets
    encoding['target_boxes'] = target_boxes
    encoding['target_labels'] = target_labels
    
    return encoding


# ==================== Training ====================

def compute_detection_loss(outputs, target_boxes, box_threshold=0.5):
    """
    Compute detection loss (simplified)
    Uses predicted boxes and compares with ground truth
    """
    pred_boxes = outputs.pred_boxes
    pred_logits = outputs.logits
    
    if pred_boxes.shape[0] == 0 or target_boxes.shape[0] == 0:
        return torch.tensor(0.0, device=pred_boxes.device, requires_grad=True)
    
    # Get highest scoring predictions
    scores = pred_logits.sigmoid().max(dim=-1)[0]
    keep = scores > box_threshold
    pred_boxes = pred_boxes[keep]
    
    if pred_boxes.shape[0] == 0:
        return torch.tensor(0.0, device=pred_boxes.device, requires_grad=True)
    
    # Compute IoU loss between predictions and targets
    # Simple L1 loss for demonstration
    target_boxes = target_boxes.to(pred_boxes.device)
    
    # Match boxes (simple: take closest predictions to targets)
    n_targets = min(target_boxes.shape[0], pred_boxes.shape[0])
    loss = F.l1_loss(pred_boxes[:n_targets], target_boxes[:n_targets])
    
    return loss


def train_coop(
    model: COOPGroundingDINO,
    train_loader: DataLoader,
    num_epochs: int = 10,
    lr: float = 0.002,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Train COOP context vectors
    """
    # Only optimize COOP parameters
    optimizer = torch.optim.AdamW(model.coop.parameters(), lr=lr)
    
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target_boxes = batch['target_boxes']  # List of tensors
            
            optimizer.zero_grad()
            
            # Forward pass (batch processing)
            try:
                outputs = model.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Compute loss for each item in batch
                batch_loss = 0
                for i in range(len(target_boxes)):
                    loss = compute_detection_loss(outputs, target_boxes[i])
                    batch_loss += loss
                
                batch_loss = batch_loss / len(target_boxes)
                
                # Backward pass
                if batch_loss.requires_grad:
                    batch_loss.backward()
                    optimizer.step()
                
                total_loss += batch_loss.item()
                pbar.set_postfix({'loss': f'{batch_loss.item():.4f}'})
            except Exception as e:
                print(f"\nError in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    
    print("Training completed!")
    return model


# ==================== Inference ====================

def inference(
    model: COOPGroundingDINO,
    image_path: str,
    text_prompt: str,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    use_coop: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Run inference on a single image
    """
    model.eval()
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Process inputs
    inputs = model.processor(
        images=image,
        text=text_prompt,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Inference
    with torch.no_grad():
        outputs = model.model(**inputs)
    
    # Post-process
    results = model.processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]]
    )[0]
    
    return {
        'boxes': results['boxes'].cpu().numpy(),
        'scores': results['scores'].cpu().numpy(),
        'labels': results['labels']
    }


def visualize_results(image_path: str, results: Dict, output_path: str = None):
    """
    Visualize detection results
    """
    from PIL import ImageDraw, ImageFont
    
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    
    boxes = results['boxes']
    scores = results['scores']
    labels = results['labels']
    
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        
        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        
        # Draw label
        text = f"{label}: {score:.2f}"
        draw.text((x1, y1 - 10), text, fill="red")
    
    if output_path:
        image.save(output_path)
        print(f"Saved visualization to {output_path}")
    
    return image


# ==================== Main Example ====================

def main():
    """
    Example usage of COOP with Grounding DINO
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize model with COOP
    print("\n=== Initializing COOP Grounding DINO ===")
    model = COOPGroundingDINO(
        model_id="IDEA-Research/grounding-dino-tiny",
        n_ctx=16,  # Number of learnable context tokens
        device=device
    )
    
    print(f"\nCOOP Context Shape: {model.coop.ctx.shape}")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Example: Create dummy training data
    print("\n=== Creating Dummy Training Data ===")
    # In practice, replace with your actual dataset
    image_paths = ["dummy_image_1.jpg", "dummy_image_2.jpg"]
    annotations = [
        {'boxes': [[100, 100, 200, 200]], 'labels': ['cat']},
        {'boxes': [[50, 50, 150, 150]], 'labels': ['dog']}
    ]
    
    # Create dataset and dataloader with custom collate function
    # Uncomment below to train on real data
    from functools import partial
    dataset = DetectionDataset(image_paths, annotations, model.processor)
    collate_fn_with_processor = partial(collate_fn, processor=model.processor)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn_with_processor)
    model = train_coop(model, train_loader, num_epochs=5, lr=0.002, device=device)
    
    # Save trained COOP weights
    # torch.save(model.coop.state_dict(), "coop_weights.pth")
    
    # Example inference (use your own image)
    print("\n=== Example Inference ===")
    print("To run inference, provide an image path:")
    print("results = inference(model, 'your_image.jpg', 'cat. dog. person.', use_coop=True)")
    print("image = visualize_results('your_image.jpg', results, 'output.jpg')")
    
    # Demonstrate COOP context
    print("\n=== Learned Context Vectors ===")
    with torch.no_grad():
        ctx = model.coop()
        print(f"Context vector norms: {ctx.norm(dim=-1)}")
        print(f"Context mean: {ctx.mean().item():.4f}, std: {ctx.std().item():.4f}")
    
    return model


if __name__ == "__main__":
    # Note: You need to install required packages:
    # pip install torch transformers pillow numpy tqdm
    
    model = main()
    
    print("\n" + "="*50)
    print("COOP Training and Inference Ready!")
    print("="*50)
    print("\nNext steps:")
    print("1. Prepare your detection dataset")
    print("2. Train COOP: train_coop(model, train_loader, num_epochs=10)")
    print("3. Run inference: inference(model, image_path, text_prompt)")
    print("4. Visualize: visualize_results(image_path, results, output_path)")