
import os
import pickle
import sys
sys.path.append(os.path.expanduser('/data/halyusuf/github/test/H-Deformable-DETR'))
from models import build_model

def get_teacher_model(
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