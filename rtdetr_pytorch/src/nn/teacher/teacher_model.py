
import os
import pickle
import torch
import sys
sys.path.append(os.path.expanduser('/data/halyusuf/github/test/H-Deformable-DETR'))
from models import build_model

def get_teacher(
    teacher_args_path: str,
    teacher_pretrained_model_path: str,
    teacher_pretrained_backbone_path: str
):
    """
    Load and build the teacher model using the provided arguments and pretrained backbone.
    Args:
        teacher_args_path (str): Path to the file containing the teacher model arguments.
        teacher_pretrained_backbone_path (str): Path to the pretrained backbone model.
    Returns:
        dict: A dictionary containing the teacher model, criterion, and postprocessors.
    """
    
    teacher = {}
    with open(teacher_args_path, 'rb') as f:
        loaded_args = pickle.load(f)

    loaded_args.pretrained_backbone_path = teacher_pretrained_backbone_path
    loaded_args.frozen_weights = teacher_pretrained_model_path
    
    teacher['model'], teacher['criterion'], teacher['postprocessors'] = build_model(loaded_args)

    return teacher


def get_teacher_topk_indices(indices, topk_indices):
    # allowed_target_indices: a Python set or tensor of allowed indices
    allowed_target_indices = set(topk_indices)  # example

    filtered_indices = []
    for pred_idx, tgt_idx in indices:
        # tgt_idx is a tensor of indices
        mask = torch.tensor([i.item() in allowed_target_indices for i in tgt_idx], device=tgt_idx.device)
        # Filter both pred_idx and tgt_idx using the mask
        filtered_pred_idx = pred_idx[mask]
        filtered_tgt_idx = tgt_idx[mask]
        filtered_indices.append((filtered_pred_idx, filtered_tgt_idx))

    return filtered_indices