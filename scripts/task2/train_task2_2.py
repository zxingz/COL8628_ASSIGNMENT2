#!/usr/bin/env python3
"""
Task 2.2: CoCoOp Training Script

This script trains CoCoOp (Conditional Context Optimization) prompts on labeled datasets.

Usage:
    python train_task2_2.py [--dataset {A,B,C}] [--epochs <num>] [--lr <rate>]
                            [--batch_size <n>] [--context_length <tokens>]
                            [--meta_hidden_dim <dim>] [--device {cuda,cpu}]
                            [--save_path <file>] [--load_path <file>] [--seed <value>]

Example:
    python train_task2_2.py --dataset B --epochs 10 --lr 5e-4 --batch_size 4 --save_path models/cocoop_model.pth
"""

import argparse
import json
import os
import sys
from datetime import datetime
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


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_SAVE_PATH = DEFAULT_MODELS_DIR / "cocoop_model.pth"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "task2-2" / "train"

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

        # Trainable prompt embeddings used for CoOp
        self.prompt_length = getattr(config, "coop_prompt_len", 16)
        prompt_init = torch.randn(1, self.prompt_length, config.hidden_size)
        self.prompt_embeddings = nn.Parameter(prompt_init)
        self.prompt_embeddings.requires_grad_(True)

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
        if output_attentions is None:
            output_attentions = self.config.output_attentions
        if output_hidden_states is None:
            output_hidden_states = self.config.output_hidden_states
        if return_dict is None:
            return_dict = self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if inputs_embeds is not None:
            raise ValueError("This BertModel manages its own prompt embeddings; please provide input_ids only.")

        if input_ids is None:
            raise ValueError("You must provide input_ids for BertModel forward")

        self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
        input_shape = input_ids.size()

        batch_size, seq_length = input_shape
        device = input_ids.device
        prompt_len = self.prompt_length

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

        # Offset text position ids so they follow the prompt tokens
        if position_ids is None:
            position_ids = torch.arange(prompt_len, prompt_len + seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        else:
            if position_ids.size(1) != seq_length:
                position_ids = position_ids[:, -seq_length:]
            position_ids = position_ids + prompt_len

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        # Prepare prompt embeddings (expand to batch, add positional & token type context)
        prompt_embeddings = self.prompt_embeddings.expand(batch_size, -1, -1)

        if self.embeddings.position_embeddings is not None:
            prompt_position_ids = torch.arange(0, prompt_len, dtype=torch.long, device=device)
            prompt_position_ids = prompt_position_ids.unsqueeze(0).expand(batch_size, -1)
            prompt_position_embeddings = self.embeddings.position_embeddings(prompt_position_ids)
        else:
            prompt_position_embeddings = 0

        prompt_token_type_ids = torch.zeros((batch_size, prompt_len), dtype=torch.long, device=device)
        prompt_token_type_embeddings = self.embeddings.token_type_embeddings(prompt_token_type_ids)

        prompt_embeddings = prompt_embeddings + prompt_position_embeddings + prompt_token_type_embeddings

        conditional_delta = getattr(self, "_conditional_prompt_delta", None)
        if conditional_delta is not None:
            if conditional_delta.size(0) != batch_size:
                raise ValueError(
                    f"Conditional prompt delta batch size {conditional_delta.size(0)} does not match inputs batch size {batch_size}."
                )
            expected_shape = (batch_size, prompt_len, self.config.hidden_size)
            if conditional_delta.shape != expected_shape:
                raise ValueError("Conditional prompt delta must have shape (batch, prompt_length, hidden_size).")
            prompt_embeddings = prompt_embeddings + conditional_delta.to(prompt_embeddings.dtype)
            delattr(self, "_conditional_prompt_delta")

        prompt_embeddings = self.embeddings.LayerNorm(prompt_embeddings)
        prompt_embeddings = self.embeddings.dropout(prompt_embeddings)

        # Concatenate prompt tokens with text tokens
        embedding_output = torch.cat([prompt_embeddings, embedding_output], dim=1).contiguous()
        seq_length = seq_length + prompt_len
        input_shape = (batch_size, seq_length)

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length + past_key_values_length), device=device)
        else:
            if attention_mask.dim() == 2:
                prompt_attention_mask = attention_mask.new_ones((batch_size, prompt_len))
                attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)
            elif attention_mask.dim() == 3:
                new_mask = attention_mask.new_zeros((batch_size, seq_length, seq_length))
                orig_len = attention_mask.size(1)
                new_mask[:, prompt_len : prompt_len + orig_len, prompt_len : prompt_len + orig_len] = attention_mask
                attention_mask = new_mask

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
        seq_without_prompt = max(encoder_outputs[0].size(1) - prompt_len, 0)
        if seq_without_prompt > 0:
            sequence_output = encoder_outputs[0].narrow(1, prompt_len, seq_without_prompt).contiguous()
        else:
            sequence_output = encoder_outputs[0].new_zeros((batch_size, 0, encoder_outputs[0].size(-1)))

        if sequence_output.size(0) != batch_size:
            raise RuntimeError(
                f"Unexpected batch dimension after prompt trimming: got {sequence_output.size(0)}, expected {batch_size}."
            )

        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            outputs = (sequence_output, pooled_output)
            if len(encoder_outputs) > 1:
                trimmed = []
                for item in encoder_outputs[1:]:
                    if isinstance(item, tuple) and len(item) > 0 and item[0] is not None and item[0].dim() == 3:
                        trimmed_tuple = []
                        for state in item:
                            if state is None:
                                trimmed_tuple.append(None)
                            else:
                                if seq_without_prompt > 0:
                                    trimmed_slice = state.narrow(1, prompt_len, seq_without_prompt).contiguous()
                                else:
                                    trimmed_slice = state.new_zeros((batch_size, 0, state.size(-1)))
                                trimmed_tuple.append(trimmed_slice)
                        trimmed.append(tuple(trimmed_tuple))
                    else:
                        trimmed.append(item)
                outputs += tuple(trimmed)
            return outputs

        hidden_states = encoder_outputs.hidden_states
        if hidden_states is not None:
            trimmed_states = []
            for state in hidden_states:
                if state is None:
                    trimmed_states.append(None)
                else:
                    if seq_without_prompt > 0:
                        trimmed_slice = state.narrow(1, prompt_len, seq_without_prompt).contiguous()
                    else:
                        trimmed_slice = state.new_zeros((batch_size, 0, state.size(-1)))
                    trimmed_states.append(trimmed_slice)
            hidden_states = tuple(trimmed_states)

        cross_attentions = getattr(encoder_outputs, "cross_attentions", None)

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=cross_attentions,
        )
    
class ConditionalPromptLearner(nn.Module):
    """Produces instance-conditional prompt adjustments for CoCoOp."""

    def __init__(self, prompt_length: int, hidden_size: int, meta_hidden_dim: int = 256):
        super().__init__()
        self.prompt_length = prompt_length
        self.hidden_size = hidden_size
        self.input_dim = 6  # mean + std for each RGB channel

        self.meta_net = nn.Sequential(
            nn.Linear(self.input_dim, meta_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(meta_hidden_dim, prompt_length * hidden_size),
        )

        nn.init.normal_(self.meta_net[-1].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.meta_net[-1].bias)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.dim() != 4:
            raise ValueError("pixel_values must be (batch, channels, height, width) for CoCoOp conditioning")

        mean = pixel_values.mean(dim=(2, 3))
        std = pixel_values.std(dim=(2, 3), unbiased=False)
        descriptor = torch.cat([mean, std], dim=-1)

        delta = self.meta_net(descriptor)
        return delta.view(-1, self.prompt_length, self.hidden_size)


class GDinoCoCoOp(nn.Module):
    def __init__(self, model, prompt_length: int = 16, meta_hidden_dim: int = 256):
        super().__init__()
        self.model = model

        if not hasattr(model, "model") or not hasattr(model.model, "text_backbone"):
            raise RuntimeError("Expected underlying Grounding DINO model to expose a text_backbone")

        text_backbone = model.model.text_backbone

        if text_backbone.prompt_length != prompt_length:
            new_prompt = torch.randn(
                1,
                prompt_length,
                text_backbone.config.hidden_size,
                device=text_backbone.prompt_embeddings.device,
            )
            text_backbone.prompt_embeddings = nn.Parameter(new_prompt)
            text_backbone.prompt_length = prompt_length

        for name, param in self.model.named_parameters():
            if "prompt_embeddings" in name:
                continue
            param.requires_grad = False

        text_backbone.prompt_embeddings.requires_grad_(True)
        self.prompt_length = prompt_length
        self.hidden_size = text_backbone.config.hidden_size
        self.conditional_prompt = ConditionalPromptLearner(prompt_length, self.hidden_size, meta_hidden_dim)

    def _inject_conditional_prompt(self, pixel_values: torch.Tensor) -> None:
        delta = self.conditional_prompt(pixel_values)
        text_backbone = self.model.model.text_backbone
        text_backbone._conditional_prompt_delta = delta.to(text_backbone.prompt_embeddings.device)

    def forward(self, pixel_values: torch.Tensor, **kwargs):
        if pixel_values is None:
            raise ValueError("pixel_values must be provided for CoCoOp conditioning")

        self._inject_conditional_prompt(pixel_values)
        try:
            return self.model(pixel_values=pixel_values, **kwargs)
        finally:
            text_backbone = self.model.model.text_backbone
            if hasattr(text_backbone, "_conditional_prompt_delta"):
                delattr(text_backbone, "_conditional_prompt_delta")

def load_model(device):
    """Loads the Grounding-DINO model and processor."""
    print("Loading Grounding-DINO model... (This may take a moment)")
    
    load_path = "models/grounding_dino_tiny.pth"
    state = torch.load(load_path, map_location=device)
    cfg = GroundingDinoConfig()
    
    model = GroundingDinoForObjectDetection(cfg)
    bert_model = BertModel(cfg.text_config)
    model.model.text_backbone = bert_model
    
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"Warning: missing keys when loading weights: {missing}")
    if unexpected:
        print(f"Warning: unexpected keys when loading weights: {unexpected}")
    model.to(device)
    
    model_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_id)
    
    # model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    
    print("Model loaded successfully.")
    return model, processor

def get_batches(dataset, batch_size=8):
    dataset.df = dataset.df.reset_index(drop=True)
    for i in range(0, len(dataset.df), batch_size):
        batch_df = dataset.df.iloc[i : i + batch_size]

        batch_prompts = []
        batch_labels = batch_df["pathology"].tolist()
        batch_boxes = []
        for _, row in batch_df.iterrows():
            label_text = str(row["pathology"]).strip().lower()
            if label_text == "nan":
                label_text = "unknown"
            prompt_text = f"a mammogram with {label_text} findings"
            batch_prompts.append([prompt_text])

            if row.isnull().any():
                batch_boxes.append([])
            else:
                batch_boxes.append([(row["xmin"], row["ymin"], row["xmax"], row["ymax"])])

        images = batch_df.apply(lambda x: dataset.load_image(x["image_name"]), axis=1).tolist()

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
    parser = argparse.ArgumentParser(description='Train CoCoOp prompts on labeled datasets')

    parser.add_argument('--dataset', type=str, choices=['A', 'B', 'C'], default='A',
                        help='Dataset identifier to train on (default: A)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of training epochs (default: 1)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for AdamW optimizer (default: 1e-3)')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for prompt optimization (default: 2)')
    parser.add_argument('--context_length', type=int, default=16,
                        help='Number of learnable prompt tokens (default: 16)')
    parser.add_argument('--meta_hidden_dim', type=int, default=256,
                        help='Hidden size of the conditional meta-network (default: 256)')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda',
                        help='Device to use for computation (default: cuda)')
    parser.add_argument('--seed', type=int, default=10,
                        help='Random seed for reproducibility (default: 10)')
    parser.add_argument('--save_path', type=str, default=str(DEFAULT_SAVE_PATH),
                        help='Path to persist trained prompt weights (default: models/cocoop_model.pth)')
    parser.add_argument('--load_path', type=str, default=None,
                        help='Optional path to existing prompt/meta weights to warm-start training')

    return parser.parse_args()

def load_training_data(train_path):
    """Load training dataset from the specified path."""
    # TODO: Implement data loading logic
    print(f"Loading training data from: {train_path}")
    pass

def train_model(model, processor, train_data, args):
    """Train the CoCoOp model with optional validation splits and history tracking."""
    print("Starting CoCoOp training...")
    device = next(model.parameters()).device

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    model.train()

    val_split_target = 0
    if args.batch_size > 10:
        val_split_target = max(1, int(args.batch_size * 0.1))

    def forward_batch(batch_prompts, batch_boxes, batch_images):
        if not batch_prompts:
            return None

        labels = []
        for boxes in batch_boxes:
            if len(boxes) > 0:
                labels.append({
                    "class_labels": torch.zeros(len(boxes), dtype=torch.long, device=device),
                    "boxes": torch.tensor(boxes, dtype=torch.float, device=device),
                })
            else:
                labels.append({
                    "class_labels": torch.tensor([], dtype=torch.long, device=device),
                    "boxes": torch.tensor([], dtype=torch.float, device=device).reshape(-1, 4),
                })

        inputs = processor(images=batch_images, text=batch_prompts, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        return model(**inputs, labels=labels, return_dict=True).loss

    history = []

    for epoch in range(args.epochs):
        total_loss = 0.0
        num_batches = 0
        batch_losses = []
        val_losses = []

        for batch_prompts, _, batch_boxes, batch_pil_images in get_batches(train_data, batch_size=args.batch_size):
            optimizer.zero_grad()

            val_size = 0
            if val_split_target > 0 and len(batch_prompts) > val_split_target:
                val_size = val_split_target

            train_end = len(batch_prompts) - val_size
            train_prompts = batch_prompts[:train_end]
            train_boxes = batch_boxes[:train_end]
            train_images = batch_pil_images[:train_end]
            val_prompts = batch_prompts[train_end:]
            val_boxes = batch_boxes[train_end:]
            val_images = batch_pil_images[train_end:]

            loss = forward_batch(train_prompts, train_boxes, train_images)
            if loss is None:
                continue

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            batch_losses.append(loss.item())
            print(f"Epoch {epoch+1}, Batch {num_batches}, Loss: {loss.item():.4f}")

            if val_size > 0:
                with torch.no_grad():
                    val_loss = forward_batch(val_prompts, val_boxes, val_images)
                if val_loss is not None:
                    val_losses.append(val_loss.item())
                    print(f"Epoch {epoch+1}, Batch {num_batches}, Validation Loss: {val_loss.item():.4f}")

        if num_batches == 0:
            print("No batches processed; please check dataset and batch size.")
            break

        avg_loss = total_loss / num_batches
        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else None
        if avg_val_loss is not None:
            print(f"Epoch {epoch+1}/{args.epochs}, Average Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{args.epochs}, Average Loss: {avg_loss:.4f}")

        history.append({
            "epoch": epoch + 1,
            "average_loss": avg_loss,
            "num_batches": num_batches,
            "batch_losses": batch_losses,
            "validation_average_loss": avg_val_loss,
            "validation_batch_losses": val_losses,
        })

    print("Training completed!")
    return history

def save_model(model, save_path):
    """Persist the trained prompt and meta-network weights."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    state_dict = {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
        if "prompt" in k or "conditional_prompt" in k
    }
    torch.save(state_dict, save_path)
    print(f"Saved CoCoOp weights to: {save_path}")

def load_prompt_weights(model, load_path, device):
    """Load prompt/meta weights if a checkpoint path is provided."""
    if load_path is None:
        return

    load_path = Path(load_path)
    if not load_path.exists():
        print(f"Load path {load_path} does not exist; skipping warm start.")
        return

    state = torch.load(load_path, map_location=device)
    filtered_state = {
        k: v
        for k, v in state.items()
        if "prompt" in k or "conditional_prompt" in k
    }
    if not filtered_state:
        print(f"No CoCoOp weights found in {load_path}; skipping warm start.")
        return

    load_result = model.load_state_dict(filtered_state, strict=False)
    print(f"Loaded prompt/meta weights from {load_path}.")
    if load_result.unexpected_keys:
        print(f"Warning: unexpected keys while loading: {load_result.unexpected_keys}")

def resolve_device(device_pref: str) -> torch.device:
    """Return requested device, falling back to CPU if CUDA unavailable."""
    device_pref = device_pref.lower()
    if device_pref == 'cuda':
        if torch.cuda.is_available():
            return torch.device('cuda')
        print("CUDA requested but not available. Falling back to CPU.")
        return torch.device('cpu')
    return torch.device('cpu')

def save_training_summary(dataset_name: str, checkpoint_path: Path, history, args, device: torch.device):
    """Persist training metadata and history to results/task2-2/train folder."""
    results_dir = DEFAULT_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset": dataset_name,
        "checkpoint_path": str(checkpoint_path),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "hyperparameters": {
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "context_length": args.context_length,
            "meta_hidden_dim": args.meta_hidden_dim,
            "device": str(device),
            "seed": args.seed,
        },
        "history": history or [],
    }

    summary_path = results_dir / f"dataset_{dataset_name}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved training summary to: {summary_path}")

def main():
    """Main execution function."""
    args = parse_arguments()
    
    print("=" * 50)
    print("Task 2.2: CoCoOp Training")
    print("=" * 50)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    data_path = PROJECT_ROOT / "data"
    dataset_name = args.dataset.upper()
    save_path = Path(args.save_path)
    base_model, processor = load_model(device)

    train_dataset = DataSet(data_path=data_path, name=dataset_name, label="train")
    coop_model = GDinoCoCoOp(
        base_model,
        prompt_length=args.context_length,
        meta_hidden_dim=args.meta_hidden_dim,
    ).to(device)

    load_prompt_weights(coop_model, args.load_path, device)

    print(f"\nTraining on Dataset {dataset_name}...")
    history = train_model(coop_model, processor, train_dataset, args)

    checkpoint_path = save_path.with_name(f"{save_path.stem}_{dataset_name}{save_path.suffix}")
    save_model(coop_model, checkpoint_path)
    save_training_summary(dataset_name, checkpoint_path, history, args, device)

if __name__ == "__main__":
    main()