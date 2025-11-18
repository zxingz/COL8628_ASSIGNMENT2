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
from typing import Optional, Union

sys.path.insert(0, os.path.join(str(Path(__file__).resolve().parent.parent.parent)))
from utils.data_utils import DataSet

import torch
import torch.optim as optim
import torch.nn as nn
from torchmetrics.detection import MeanAveragePrecision

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers.models.grounding_dino.modeling_grounding_dino import GroundingDinoForObjectDetection
from transformers.models.grounding_dino.configuration_grounding_dino import GroundingDinoConfig
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder, BertPooler
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.utils.auto_docstring import auto_docstring
from transformers.models.bert.modeling_bert import Cache
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa, _prepare_4d_causal_attention_mask_for_sdpa

class BertModel(BertPreTrainedModel):
    _no_split_modules = ["BertEmbeddings", "BertLayer"]

    def __init__(self, config, add_pooling_layer=False):
        r"""
        add_pooling_layer (bool, *optional*, defaults to `True`):
            Whether to add a pooling layer
        """
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.attn_implementation = config._attn_implementation
        self.position_embedding_type = config.position_embedding_type

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Union[tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")

        # Detect whether CoOp prompt is attached to this text backbone
        use_coop = hasattr(self, "coop_prompt") and self.coop_prompt is not None

        # If using CoOp, we will build inputs_embeds by concatenating the prompt embeddings
        if use_coop:
            # Build token embeddings from ids if necessary
            if inputs_embeds is None:
                if input_ids is None:
                    raise ValueError("CoOp mode requires input_ids or inputs_embeds")
                self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
                token_embeds = self.embeddings.word_embeddings(input_ids)
                device = input_ids.device
                batch_size = input_ids.size(0)
            else:
                token_embeds = inputs_embeds
                device = token_embeds.device
                batch_size = token_embeds.size(0)

            # Expand and prepend learnable prompt
            prompt_len = self.coop_prompt.size(1)
            prompt = self.coop_prompt.to(token_embeds.device).expand(batch_size, -1, -1)
            inputs_embeds = torch.cat([prompt, token_embeds], dim=1)  # [B, P+L, D]
            input_ids = None  # ensure embeddings path is used below

            # Build/extend attention mask to cover prompt tokens
            if attention_mask is None:
                attention_mask = torch.ones((batch_size, inputs_embeds.size(1)), device=device)
            else:
                ones = torch.ones((batch_size, prompt_len), device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([ones, attention_mask], dim=1)

            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            if input_ids is not None:
                self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
                input_shape = input_ids.size()
                device = input_ids.device
            elif inputs_embeds is not None:
                input_shape = inputs_embeds.size()[:-1]
                device = inputs_embeds.device
            else:
                raise ValueError("You have to specify either input_ids or inputs_embeds")

            batch_size, seq_length = input_shape

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = (
                past_key_values[0][0].shape[-2]
                if not isinstance(past_key_values, Cache)
                else past_key_values.get_seq_length()
            )

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # Build final embedding output (word + position + token type)
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length + past_key_values_length), device=device)

        use_sdpa_attention_masks = (
            self.attn_implementation == "sdpa"
            and self.position_embedding_type == "absolute"
            and head_mask is None
            and not output_attentions
        )

        # Expand the attention mask
        if use_sdpa_attention_masks and attention_mask.dim() == 2:
            # Expand the attention mask for SDPA.
            # [bsz, seq_len] -> [bsz, 1, seq_len, seq_len]
            if self.config.is_decoder:
                extended_attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask,
                    input_shape,
                    embedding_output,
                    past_key_values_length,
                )
            else:
                extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    attention_mask, embedding_output.dtype, tgt_len=seq_length
                )
        else:
            # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
            # ourselves in which case we just need to make it broadcastable to all heads.
            extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if use_sdpa_attention_masks and encoder_attention_mask.dim() == 2:
                # Expand the attention mask for SDPA.
                # [bsz, seq_len] -> [bsz, 1, seq_len, seq_len]
                encoder_extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    encoder_attention_mask, embedding_output.dtype, tgt_len=seq_length
                )
            else:
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )
    
class GDinoCoop(nn.Module):
    def __init__(self, model, prompt_length=16):
        super().__init__()
        self.model = model
        
        # Freeze the entire base model
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Text backbone
        text_backbone = model.model.text_backbone

        print(self.model.config.text_config)

        # Access the text backbone directly
        embed_dim = self.model.config.text_config.hidden_size

        # Learnable prompt
        self.prompt_len = prompt_length
        self.prompt = nn.Parameter(torch.randn(1, self.prompt_len, embed_dim))
        self.prompt.requires_grad = True

    def forward(self, **kwargs):
        # Attach the learnable prompt to the BERT text backbone for this forward
        # It will be consumed inside BertModel.forward by concatenating to token embeddings
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'text_backbone'):
            self.model.model.text_backbone.coop_prompt = self.prompt
        else:
            raise RuntimeError("Underlying model does not expose a BERT text_backbone")

        return self.model(**kwargs)
    

def load_model(device):
    """Loads the Grounding-DINO model and processor."""
    print("Loading Grounding-DINO model... (This may take a moment)")
    
    load_path = "models/grounding_dino_tiny.pth"
    state = torch.load(load_path, map_location=device)
    cfg = GroundingDinoConfig()
    
    model = GroundingDinoForObjectDetection(cfg)
    bert_model = BertModel(cfg.text_config)
    model.model.text_backbone = bert_model
    
    model.load_state_dict(state)
    model.to(device)
    
    model_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_id)
    
    # model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    
    print("Model loaded successfully.")
    return model, processor

def get_batches(dataset, default_prompt="This is a Mamogram", batch_size=8):
    
    dataset.df = dataset.df.reset_index(drop=True)
    for i in range(0, len(dataset.df), batch_size):
        
        # list of prompts
        batch_prompts = [default_prompt] * min(batch_size, len(dataset.df) - i)
        
        # list of labels
        batch_labels = dataset.df.iloc[i:i+batch_size]["pathology"].tolist()
        
        # list of bounding box
        batch_boxes = []
        for _, row in dataset.df.iloc[i:i+batch_size].iterrows():
            if row.isnull().any():  # Benign case has NaNs
                batch_boxes.append([])
            else:
                batch_boxes.append([(row["xmin"], row["ymin"], row["xmax"], row["ymax"])])
        
        # list of PIL images
        images = dataset.df.iloc[i:i+batch_size].apply(lambda x: dataset.load_image(x["image_name"]), axis=1).tolist()
        
        yield batch_prompts, batch_labels, batch_boxes, images

def smoke_test_forward_minimal(model, processor, device):
    """Run a tiny forward pass on synthetic data to validate CoOp wiring."""
    from PIL import Image
    # Create two dummy RGB images
    img1 = Image.new('RGB', (256, 256), color=(128, 128, 128))
    img2 = Image.new('RGB', (256, 256), color=(64, 64, 64))
    prompts = ["This is a Mammogram", "This is a Mammogram"]

    inputs = processor(images=[img1, img2], text=prompts, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Forward without labels to just ensure it runs and returns outputs
    with torch.no_grad():
        out = model(**inputs, return_dict=True)

    print("Smoke test forward: ok. Keys:", list(out.keys()) if hasattr(out, 'keys') else type(out))
    # Quick check that prompt participates and is trainable
    prompt_param = None
    for n, p in model.named_parameters():
        if 'prompt' in n:
            prompt_param = p
            break
    if prompt_param is not None:
        print(f"Prompt shape: {tuple(prompt_param.shape)}, requires_grad={prompt_param.requires_grad}")
    else:
        print("Warning: prompt parameter not found in model parameters")

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
                else:  # Benign cases
                    labels.append({
                        "class_labels": torch.tensor([], dtype=torch.long, device=device),
                        "boxes": torch.tensor([], dtype=torch.float, device=device).reshape(-1, 4)
                    })

            # Process inputs
            inputs = processor(images=batch_pil_images, text=batch_prompts, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Forward pass
            outputs = model(**inputs, labels=labels, return_dict=True)
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
    # Keep defaults small for quick iteration; adjust as needed
    args = Namespace(epochs=1, lr=0.001, batch_size=2, context_length=16)
    
    # Set random seed
    torch.manual_seed(10)
    
    # Setup device
    # device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
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
    # test_A = DataSet(data_path=data_path, name="A", label="test")
    
    # # Dataset B
    # train_B = DataSet(data_path=data_path, name="B", label="train")
    # test_B = DataSet(data_path=data_path, name="B", label="test")
    
    # # Dataset C
    # train_C = DataSet(data_path=data_path, name="C", label="train")
    # test_C = DataSet(data_path=data_path, name="C", label="test")
    
    # Initialize CoOp model
    coop_model = GDinoCoop(base_model, prompt_length=args.context_length).to(device)

    # Quick smoke test to ensure forward works with the learnable prompt
    print("\nRunning smoke test forward (synthetic data)...")
    smoke_test_forward_minimal(coop_model, processor, device)

    # Train on Dataset A (one epoch by default for a light run)
    print("\nTraining on Dataset A...")
    train_model(coop_model, processor, train_A, args)
    
    # Save the trained prompt
    save_model(coop_model, save_path)

if __name__ == "__main__":
    main()