#!/usr/bin/env python3
"""
Task 3: Semi-Supervised CoOp Training Script

This script adapts FixMatch for prompt tuning with Grounding DINO. Only the textual prompt
embeddings are optimized, while the vision backbone remains frozen. You can train on dataset
pairs (A-B, B-C, A-C) by selecting which dataset supplies the primary labeled pool and which
provides the partially labeled + unlabeled splits.

Usage:
    python train_task3.py --dataset_pair A-B [--epochs <num>] [--lr <rate>]
                          [--batch_size <n>] [--context_length <tokens>]
                          [--device {cuda,cpu}] [--save_path <file>] [--load_path <file>]

Example:
    python train_task3.py --dataset_pair B-C --epochs 15 --lr 5e-4 --batch_size 8 \
                          --save_path models/fm_coop_model.pth
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection import MeanAveragePrecision
from torchvision import transforms

sys.path.insert(0, os.path.join(str(Path(__file__).resolve().parent.parent.parent)))

from transformers import AutoProcessor
from transformers.models.bert.modeling_bert import (  # type: ignore
    BertEmbeddings,
    BertEncoder,
    BertPooler,
    BertPreTrainedModel,
    Cache,
)
from transformers.models.grounding_dino.configuration_grounding_dino import GroundingDinoConfig
from transformers.models.grounding_dino.modeling_grounding_dino import GroundingDinoForObjectDetection
from transformers.modeling_attn_mask_utils import (  # type: ignore
    _prepare_4d_attention_mask_for_sdpa,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.utils.auto_docstring import auto_docstring

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_SAVE_PATH = DEFAULT_MODELS_DIR / "fm_coop_model.pth"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "task3" / "train"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
SUPPORTED_DATASET_PAIRS = ("A-B", "B-C", "A-C")

# Single-class detection prompt; benign cases simply contribute no boxes.
PROMPT_TEXT = "a mammogram containing a malignant mass"
PROMPT_TOKEN_CLASS = 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_pref: str) -> torch.device:
    """Return requested device, falling back to CPU if CUDA unavailable."""
    device_pref = device_pref.lower()
    if device_pref == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device("cpu")


def load_dataset_frame(root: Path, dataset_id: str, split: str) -> pd.DataFrame:
    csv_path = root / f"dataset_{dataset_id}" / split / f"{split}.csv"
    frame = pd.read_csv(csv_path)
    frame["dataset"] = dataset_id
    frame["split"] = split
    return frame


def split_labeled_unlabeled(df: pd.DataFrame, labeled_fraction: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    labeled_fraction = max(0.0, min(1.0, labeled_fraction))
    if labeled_fraction <= 0.0:
        return df.iloc[0:0].copy(), df.copy()
    labeled_df = df.sample(frac=labeled_fraction, random_state=seed)
    unlabeled_df = df.drop(labeled_df.index)
    return labeled_df.reset_index(drop=True), unlabeled_df.reset_index(drop=True)


def build_weak_augmentation() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomAutocontrast(p=0.2),
        ]
    )


def build_strong_augmentation() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=0.3),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.5, 1.5)),
        ]
    )


def apply_transform(image: Image.Image, transform: Optional[transforms.Compose]) -> Image.Image:
    if transform is None:
        return image
    return transform(image.copy())


def xyxy_to_cxcywh(box: Sequence[float], width: float, height: float) -> List[float]:
    xmin, ymin, xmax, ymax = box
    cx = ((xmin + xmax) / 2.0) / width
    cy = ((ymin + ymax) / 2.0) / height
    w = (xmax - xmin) / width
    h = (ymax - ymin) / height
    return [cx, cy, w, h]


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x0 = cx - 0.5 * w
    y0 = cy - 0.5 * h
    x1 = cx + 0.5 * w
    y1 = cy + 0.5 * h
    return torch.stack((x0, y0, x1, y1), dim=-1)


@dataclass
class LabeledSample:
    image: Image.Image
    target: Dict[str, torch.Tensor]
    gt_boxes_xyxy: torch.Tensor
    gt_labels: torch.Tensor
    size: Tuple[int, int]


class MammoLabeledDataset(Dataset[LabeledSample]):
    def __init__(
        self,
        frame: pd.DataFrame,
        root: Path,
        subset: str,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.root = root
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> LabeledSample:
        row = self.frame.iloc[idx]
        image_path = self.root / f"dataset_{row['dataset']}" / self.subset / row["image_name"]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        if self.transform is not None:
            image = apply_transform(image, self.transform)

        has_box = not pd.isna(row["xmin"]) and not pd.isna(row["ymin"]) and not pd.isna(row["xmax"]) and not pd.isna(row["ymax"])
        if has_box:
            xyxy_box = [row["xmin"], row["ymin"], row["xmax"], row["ymax"]]
            gt_boxes_xyxy = torch.tensor([xyxy_box], dtype=torch.float32)
            class_labels = torch.tensor([PROMPT_TOKEN_CLASS], dtype=torch.long)
            cxcywh = torch.tensor([xyxy_to_cxcywh(xyxy_box, width, height)], dtype=torch.float32)
        else:
            gt_boxes_xyxy = torch.zeros((0, 4), dtype=torch.float32)
            class_labels = torch.zeros((0,), dtype=torch.long)
            cxcywh = torch.zeros((0, 4), dtype=torch.float32)

        target = {"class_labels": class_labels, "boxes": cxcywh}
        return LabeledSample(image=image, target=target, gt_boxes_xyxy=gt_boxes_xyxy, gt_labels=class_labels, size=(height, width))


@dataclass
class UnlabeledSample:
    weak_image: Image.Image
    strong_image: Image.Image
    size: Tuple[int, int]


class MammoUnlabeledDataset(Dataset[UnlabeledSample]):
    def __init__(
        self,
        frame: pd.DataFrame,
        root: Path,
        subset: str,
        weak_transform: Optional[transforms.Compose],
        strong_transform: Optional[transforms.Compose],
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.root = root
        self.subset = subset
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> UnlabeledSample:
        row = self.frame.iloc[idx]
        image_path = self.root / f"dataset_{row['dataset']}" / self.subset / row["image_name"]
        image = Image.open(image_path).convert("RGB")
        height, width = image.size[1], image.size[0]

        weak_image = apply_transform(image, self.weak_transform)
        strong_image = apply_transform(image, self.strong_transform)
        return UnlabeledSample(weak_image=weak_image, strong_image=strong_image, size=(height, width))


def collate_labeled(batch: Sequence[LabeledSample]):
    images = [item.image for item in batch]
    targets = []
    gt_info = []
    sizes = []
    for item in batch:
        targets.append(item.target)
        gt_info.append({"boxes": item.gt_boxes_xyxy, "labels": item.gt_labels})
        sizes.append(item.size)
    return images, targets, gt_info, sizes


def collate_unlabeled(batch: Sequence[UnlabeledSample]):
    weak_images = [item.weak_image for item in batch]
    strong_images = [item.strong_image for item in batch]
    sizes = [item.size for item in batch]
    return weak_images, strong_images, sizes


class BertModel(BertPreTrainedModel):
    _no_split_modules = ["BertEmbeddings", "BertLayer"]

    def __init__(self, config, add_pooling_layer: bool = False):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.attn_implementation = config._attn_implementation
        self.position_embedding_type = config.position_embedding_type

        self.prompt_length = getattr(config, "coop_prompt_len", 16)
        prompt_init = torch.randn(1, self.prompt_length, config.hidden_size)
        self.prompt_embeddings = nn.Parameter(prompt_init)
        self.prompt_embeddings.requires_grad_(True)

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
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
    ) -> BaseModelOutputWithPoolingAndCrossAttentions:
        if output_attentions is None:
            output_attentions = self.config.output_attentions
        if output_hidden_states is None:
            output_hidden_states = self.config.output_hidden_states
        if return_dict is None:
            return_dict = self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = self.config.use_cache if use_cache is None else use_cache
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
                buffered = self.embeddings.token_type_ids[:, :seq_length]
                token_type_ids = buffered.expand(batch_size, seq_length)
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

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
        prompt_embeddings = self.embeddings.LayerNorm(prompt_embeddings)
        prompt_embeddings = self.embeddings.dropout(prompt_embeddings)

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

        if use_sdpa_attention_masks and attention_mask.dim() == 2:
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
            extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if use_sdpa_attention_masks and encoder_attention_mask.dim() == 2:
                encoder_extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    encoder_attention_mask, embedding_output.dtype, tgt_len=seq_length
                )
            else:
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

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
        sequence_output = encoder_outputs[0].narrow(1, prompt_len, seq_without_prompt).contiguous()
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        hidden_states = encoder_outputs.hidden_states
        if hidden_states is not None:
            trimmed_states = []
            for state in hidden_states:
                if state is None:
                    trimmed_states.append(None)
                else:
                    trimmed_states.append(state.narrow(1, prompt_len, seq_without_prompt).contiguous())
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


class GDinoCoop(nn.Module):
    def __init__(self, model: GroundingDinoForObjectDetection, prompt_length: int = 16):
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

    def forward(self, **kwargs):
        return self.model(**kwargs)


def prompt_parameters(module: nn.Module) -> Iterable[nn.Parameter]:
    for name, param in module.named_parameters():
        if param.requires_grad:
            yield param


def save_model(model: GDinoCoop, save_path: Path) -> None:
    """Persist prompt-related parameters to disk."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_state = {k: v.detach().cpu() for k, v in model.state_dict().items() if "prompt" in k}
    torch.save(prompt_state, save_path)
    print(f"Saved prompt weights to: {save_path}")


def load_prompt_weights(model: GDinoCoop, load_path: Optional[str], device: torch.device) -> None:
    """Warm start prompts/meta-network weights if available."""
    if not load_path:
        return

    candidate = Path(load_path)
    if not candidate.exists():
        print(f"Load path {candidate} does not exist; skipping warm start.")
        return

    state = torch.load(candidate, map_location=device)
    filtered_state = {k: v for k, v in state.items() if "prompt" in k}
    if not filtered_state:
        print(f"No prompt weights found in {candidate}; skipping warm start.")
        return

    load_result = model.load_state_dict(filtered_state, strict=False)
    print(f"Loaded prompt weights from {candidate} ({len(filtered_state)} tensors).")
    if load_result.unexpected_keys:
        print(f"Warning: unexpected keys when loading prompts: {load_result.unexpected_keys}")


def save_training_summary(
    dataset_pair: str,
    checkpoint_path: Path,
    history,
    args: argparse.Namespace,
    device: torch.device,
    best_map: float,
) -> None:
    results_dir = DEFAULT_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset_primary, dataset_secondary = dataset_pair.split("-")
    summary = {
        "dataset_pair": dataset_pair,
        "checkpoint_path": str(checkpoint_path),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "best_map": best_map,
        "hyperparameters": {
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "unlabeled_batch_size": args.unlabeled_batch_size,
            "context_length": args.context_length,
            "device": str(device),
            "seed": args.seed,
            "lambda_u": args.lambda_u,
            "threshold": args.threshold,
            "eval_threshold": args.eval_threshold,
            "ema_decay": args.ema_decay,
            "b_labeled_fraction": args.b_labeled_fraction,
            "dataset_primary": dataset_primary,
            "dataset_secondary": dataset_secondary,
        },
        "history": history or [],
    }

    summary_path = results_dir / f"dataset_{dataset_pair}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved training summary to: {summary_path}")


def load_frozen_grounding_dino(weight_path: Path, device: torch.device, prompt_length: int) -> GDinoCoop:
    cfg = GroundingDinoConfig()
    base_model = GroundingDinoForObjectDetection(cfg)
    text_backbone = BertModel(cfg.text_config)
    base_model.model.text_backbone = text_backbone

    state = torch.load(weight_path, map_location=device)
    missing, unexpected = base_model.load_state_dict(state, strict=False)
    if missing:
        print(f"[Warning] Missing keys when loading weights: {missing}")
    if unexpected:
        print(f"[Warning] Unexpected keys when loading weights: {unexpected}")

    base_model.to(device)
    return GDinoCoop(base_model, prompt_length=prompt_length).to(device)


def prepare_inputs(
    processor: AutoProcessor,
    images: Sequence[Image.Image],
    device: torch.device,
    text_prompts: Optional[Sequence[str]] = None,
) -> Dict[str, torch.Tensor]:
    if text_prompts is None:
        text_prompts = [PROMPT_TEXT] * len(images)
    inputs = processor(images=list(images), text=list(text_prompts), return_tensors="pt", padding=True)
    return {key: value.to(device) for key, value in inputs.items()}


def build_pseudo_labels(
    outputs,
    threshold: float,
    device: torch.device,
) -> List[Dict[str, torch.Tensor]]:
    logits = outputs.logits.sigmoid().detach()
    pred_boxes = outputs.pred_boxes.detach()

    pseudo_labels: List[Dict[str, torch.Tensor]] = []
    for box_scores, boxes in zip(logits, pred_boxes):
        if box_scores.ndim == 2:
            scores, _ = box_scores.max(dim=-1)
        else:
            scores = box_scores.squeeze(-1)

        mask = scores >= threshold
        selected_boxes = boxes[mask]

        if selected_boxes.numel() == 0:
            pseudo_labels.append(
                {
                    "class_labels": torch.zeros((0,), dtype=torch.long, device=device),
                    "boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
                }
            )
        else:
            pseudo_labels.append(
                {
                    "class_labels": torch.full(
                        (selected_boxes.size(0),), PROMPT_TOKEN_CLASS, dtype=torch.long, device=device
                    ),
                    "boxes": selected_boxes.to(device),
                }
            )
    return pseudo_labels


def update_ema(student: GDinoCoop, teacher: GDinoCoop, decay: float) -> None:
    student_state = dict(student.named_parameters())
    teacher_state = dict(teacher.named_parameters())
    for name, param in teacher_state.items():
        if name not in student_state:
            continue
        param.data.mul_(decay).add_(student_state[name].data, alpha=1.0 - decay)


def move_targets_to_device(targets: List[Dict[str, torch.Tensor]], device: torch.device) -> List[Dict[str, torch.Tensor]]:
    moved = []
    for target in targets:
        moved.append({key: value.to(device) for key, value in target.items()})
    return moved


def evaluate_map(
    model: GDinoCoop,
    dataloader: DataLoader,
    processor: AutoProcessor,
    device: torch.device,
    conf_threshold: float,
) -> float:
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox")

    with torch.no_grad():
        for images, _, gt_info, sizes in dataloader:
            inputs = prepare_inputs(processor, images, device)
            outputs = model(**inputs)
            logits = outputs.logits.sigmoid()
            pred_boxes = outputs.pred_boxes

            for idx in range(len(images)):
                if logits[idx].ndim == 2:
                    scores = logits[idx].max(dim=-1).values
                else:
                    scores = logits[idx].squeeze(-1)
                labels = torch.full(scores.shape, PROMPT_TOKEN_CLASS, dtype=torch.long, device=scores.device)

                mask = scores >= conf_threshold
                scores = scores[mask]
                labels = labels[mask]
                boxes = pred_boxes[idx][mask]

                if boxes.numel() == 0:
                    pred = {"boxes": torch.zeros((0, 4)), "scores": torch.zeros((0,)), "labels": torch.zeros((0,), dtype=torch.long)}
                else:
                    boxes_xyxy = cxcywh_to_xyxy(boxes)
                    height, width = sizes[idx]
                    boxes_xyxy[:, [0, 2]] *= width
                    boxes_xyxy[:, [1, 3]] *= height
                    pred = {"boxes": boxes_xyxy.cpu(), "scores": scores.cpu(), "labels": labels.cpu()}

                target_boxes = gt_info[idx]["boxes"]
                target_labels = gt_info[idx]["labels"]
                target = {"boxes": target_boxes.cpu(), "labels": target_labels.cpu()}
                metric.update([pred], [target])

    result = metric.compute()
    return float(result.get("map", torch.tensor(0.0)))


def train(args: argparse.Namespace, device: torch.device):
    set_seed(args.seed)

    data_root = Path(args.data_root)
    weight_path = Path(args.model_path)
    base_save_path = Path(args.save_path)
    dataset_primary, dataset_secondary = args.dataset_pair.split("-")
    dataset_pair_label = f"{dataset_primary}-{dataset_secondary}"
    checkpoint_path = base_save_path.with_name(f"{base_save_path.stem}_{dataset_pair_label}{base_save_path.suffix}")

    print(f"Loading processor {args.processor_name}...")
    processor = AutoProcessor.from_pretrained(args.processor_name)

    print("Loading Grounding DINO weights (frozen except prompts)...")
    student = load_frozen_grounding_dino(weight_path, device, args.context_length)
    load_prompt_weights(student, args.load_path, device)
    teacher = deepcopy(student)
    teacher.eval()

    df_primary_train = load_dataset_frame(data_root, dataset_primary, "train")
    df_secondary_train = load_dataset_frame(data_root, dataset_secondary, "train")
    df_secondary_labeled, df_secondary_unlabeled = split_labeled_unlabeled(
        df_secondary_train, args.b_labeled_fraction, args.seed
    )
    labeled_frame = pd.concat([df_primary_train, df_secondary_labeled], ignore_index=True)

    df_secondary_test = load_dataset_frame(data_root, dataset_secondary, "test")

    weak_aug = build_weak_augmentation()
    strong_aug = build_strong_augmentation()

    labeled_dataset = MammoLabeledDataset(labeled_frame, data_root, "train", transform=None)
    labeled_loader = DataLoader(
        labeled_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_labeled,
        drop_last=False,
    )

    unlabeled_dataset = MammoUnlabeledDataset(
        df_secondary_unlabeled,
        data_root,
        "train",
        weak_transform=weak_aug,
        strong_transform=strong_aug,
    )
    if len(unlabeled_dataset) == 0:
        unlabeled_loader = None
        print("[Info] No unlabeled samples available; proceeding with supervised training only.")
    else:
        drop_last = len(unlabeled_dataset) >= args.unlabeled_batch_size
        unlabeled_loader = DataLoader(
            unlabeled_dataset,
            batch_size=args.unlabeled_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_unlabeled,
            drop_last=drop_last,
        )

    eval_dataset = MammoLabeledDataset(df_secondary_test, data_root, "test", transform=None)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_labeled,
        drop_last=False,
    )

    optimizer = optim.AdamW(prompt_parameters(student), lr=args.lr, weight_decay=args.weight_decay)

    best_map = -1.0
    best_checkpoint_path = checkpoint_path
    history = []
    val_split_target = 0
    if args.batch_size > 10:
        val_split_target = max(1, int(args.batch_size * 0.1))

    for epoch in range(1, args.epochs + 1):
        student.train()
        labeled_iter = iter(labeled_loader)
        has_unlabeled = unlabeled_loader is not None and len(unlabeled_loader) > 0
        unlabeled_iter = iter(unlabeled_loader) if has_unlabeled else None

        running_sup = 0.0
        running_unsup = 0.0
        running_total = 0.0
        batches = 0
        epoch_batch_metrics = []
        val_losses = []

        len_unlabeled = len(unlabeled_loader) if has_unlabeled else 0
        max_steps = max(len(labeled_loader), len_unlabeled) if max(len(labeled_loader), len_unlabeled) > 0 else 0

        for step in range(max_steps):
            try:
                labeled_images, labeled_targets, _, _ = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                labeled_images, labeled_targets, _, _ = next(labeled_iter)

            if has_unlabeled:
                try:
                    weak_images, strong_images, _ = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(unlabeled_loader)
                    weak_images, strong_images, _ = next(unlabeled_iter)
            else:
                weak_images = None
                strong_images = None

            val_size = 0
            if val_split_target > 0 and len(labeled_images) > val_split_target:
                val_size = val_split_target
            train_len = len(labeled_images) - val_size
            if train_len <= 0:
                val_size = 0
                train_len = len(labeled_images)

            train_images = labeled_images[:train_len]
            train_targets = labeled_targets[:train_len]
            val_images = labeled_images[train_len:]
            val_targets = labeled_targets[train_len:]

            optimizer.zero_grad()

            inputs_labeled = prepare_inputs(processor, train_images, device)
            targets_labeled = move_targets_to_device(train_targets, device)
            outputs_labeled = student(**inputs_labeled, labels=targets_labeled)
            supervised_loss = outputs_labeled.loss

            if has_unlabeled and weak_images is not None and strong_images is not None:
                with torch.no_grad():
                    teacher_inputs = prepare_inputs(processor, weak_images, device)
                    teacher_outputs = teacher(**teacher_inputs)
                    pseudo_labels = build_pseudo_labels(teacher_outputs, args.threshold, device)

                strong_inputs = prepare_inputs(processor, strong_images, device)
                if any(label["boxes"].numel() > 0 for label in pseudo_labels):
                    outputs_unlabeled = student(**strong_inputs, labels=pseudo_labels)
                    unsupervised_loss = outputs_unlabeled.loss
                else:
                    unsupervised_loss = torch.tensor(0.0, device=device)
            else:
                unsupervised_loss = torch.tensor(0.0, device=device)

            total_loss = supervised_loss + args.lambda_u * unsupervised_loss
            total_loss.backward()
            optimizer.step()

            update_ema(student, teacher, args.ema_decay)

            running_sup += float(supervised_loss.detach())
            running_unsup += float(unsupervised_loss.detach())
            running_total += float(total_loss.detach())
            batches += 1

            val_loss_value = None
            if val_size > 0 and len(val_images) > 0:
                with torch.no_grad():
                    val_inputs = prepare_inputs(processor, val_images, device)
                    val_targets_device = move_targets_to_device(val_targets, device)
                    val_outputs = student(**val_inputs, labels=val_targets_device)
                    val_loss_value = float(val_outputs.loss.detach())
                    val_losses.append(val_loss_value)

            epoch_batch_metrics.append(
                {
                    "step": step + 1,
                    "supervised_loss": float(supervised_loss.detach()),
                    "unsupervised_loss": float(unsupervised_loss.detach()),
                    "total_loss": float(total_loss.detach()),
                    "validation_loss": val_loss_value,
                }
            )

        avg_sup = running_sup / max(batches, 1)
        avg_unsup = running_unsup / max(batches, 1)
        avg_total = running_total / max(batches, 1)
        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else None

        eval_map = evaluate_map(student, eval_loader, processor, device, args.eval_threshold)
        if avg_val_loss is not None:
            print(
                f"Epoch {epoch:02d} | loss={avg_total:.4f} (sup={avg_sup:.4f}, unsup={avg_unsup:.4f}) | "
                f"val={avg_val_loss:.4f} | {dataset_secondary}-test mAP={eval_map:.4f}"
            )
        else:
            print(
                f"Epoch {epoch:02d} | loss={avg_total:.4f} (sup={avg_sup:.4f}, unsup={avg_unsup:.4f}) | "
                f"{dataset_secondary}-test mAP={eval_map:.4f}"
            )

        history.append(
            {
                "epoch": epoch,
                "average_supervised_loss": avg_sup,
                "average_unsupervised_loss": avg_unsup,
                "average_total_loss": avg_total,
                "validation_average_loss": avg_val_loss,
                "num_batches": batches,
                "batch_metrics": epoch_batch_metrics,
                "eval_map": eval_map,
            }
        )

        if eval_map > best_map:
            best_map = eval_map
            save_model(student, checkpoint_path)
            best_checkpoint_path = checkpoint_path
            print(f"  Saved improved prompts to {checkpoint_path}")

    if best_map < 0:
        save_model(student, checkpoint_path)
        best_checkpoint_path = checkpoint_path

    print(f"Best validation mAP: {best_map:.4f}")
    return history, best_checkpoint_path, best_map


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FixMatch-style semi-supervised CoOp training")
    parser.add_argument(
        "--dataset_pair",
        type=str,
        choices=SUPPORTED_DATASET_PAIRS,
        default="A-B",
        help="Dataset pair to train on in <PRIMARY>-<SECONDARY> format",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=str(DEFAULT_DATA_ROOT),
        help="Root directory containing dataset folders",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/grounding_dino_tiny.pth",
        help="Path to the frozen Grounding DINO checkpoint",
    )
    parser.add_argument(
        "--processor_name",
        type=str,
        default="IDEA-Research/grounding-dino-tiny",
        help="Identifier of the Hugging Face processor",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=str(DEFAULT_SAVE_PATH),
        help="Base path for saving prompt weights (dataset suffix will be added)",
    )
    parser.add_argument(
        "--load_path",
        type=str,
        default=None,
        help="Optional checkpoint to warm-start prompt weights",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--unlabeled_batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--context_length", type=int, default=16)
    parser.add_argument("--lambda_u", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--eval_threshold", type=float, default=0.3)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--b_labeled_fraction", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (defaults to CUDA if available)")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    print("=" * 50)
    print("Task 3: Semi-Supervised CoOp Training")
    print("=" * 50)

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    history, checkpoint_path, best_map = train(args, device)
    save_training_summary(args.dataset_pair, checkpoint_path, history, args, device, best_map)


if __name__ == "__main__":
    main()
