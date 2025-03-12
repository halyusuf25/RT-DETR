
import os
import torch
# import argparse
# import datetime
# import random
# import time
# from pathlib import Path
import pickle
# import mmcv
# import numpy as np
# from torch.utils.data import DataLoader

HDEFORMABLE_SWIN_L_900_ARGS = '/data/shared/pretrained_models/hdeformable/args900.pkl'
# HDEFORMABLE_SWIN_L_900_ARGS_NoDevice = 'args_hdeformable_SwinLarg_900q_no_device.pkl'

# import necessary function to run buil_model from H-DEFORMABLE-DETR repo
# import importlib
import sys
sys.path.append(os.path.expanduser('/data/halyusuf/github/test/H-Deformable-DETR'))
# import util.misc as utils
# import custom_datasets.samplers as samplers
# from custom_datasets import build_dataset, get_coco_api_from_dataset
# from engine import evaluate, train_one_epoch
from models import build_model
# from main import get_args_parser
import pprint as pp


# Load the args object from the file
with open(HDEFORMABLE_SWIN_L_900_ARGS, 'rb') as f:
    loaded_args = pickle.load(f)

loaded_args.pretrained_backbone_path = '/data/shared/pretrained_backbones/swin/swin_large_patch4_window7_224_22k.pth'

teacher_model, teacher_criterion, teacher_postprocessors = build_model(loaded_args)
print(f"loaded arguments: {loaded_args}")
print(f"type(teacher_model): {type(teacher_model)}")
print(f"type(teacher_criterion): {type(teacher_criterion)}")
print(f"type(teacher_postprocessors): {type(teacher_postprocessors)}")
print(f" is teacher model is torch.nn.Module: {isinstance(teacher_model, torch.nn.Module)}")
pp.pprint(loaded_args)
