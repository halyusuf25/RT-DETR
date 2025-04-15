""""by lyuwenyu
"""

import torch
import torch.utils.data

import torchvision
torchvision.disable_beta_transforms_warning()

import PIL 

__all__ = ['show_sample']

def show_sample(sample):
    """for coco dataset/dataloader
    """
    import matplotlib.pyplot as plt
    from torchvision.transforms.v2 import functional as F
    from torchvision.utils import draw_bounding_boxes

    image, target = sample
    if isinstance(image, PIL.Image.Image):
        image = F.to_image_tensor(image)

    image = F.convert_dtype(image, torch.uint8)
    annotated_image = draw_bounding_boxes(image, target["boxes"], colors="yellow", width=3)

    fig, ax = plt.subplots()
    ax.imshow(annotated_image.permute(1, 2, 0).numpy())
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fig.tight_layout()
    fig.show()
    plt.show()


import json
import matplotlib.pyplot as plt

def plot_training_metrics(log_file_path):
    # Read the log file
    with open(log_file_path, 'r') as f:
        lines = f.readlines()
    
    # Parse each line as JSON
    data = [json.loads(line) for line in lines]
    
    # Extract epochs and metrics
    epochs = [entry['epoch'] for entry in data]
    train_losses = [entry['train_loss'] for entry in data]
    
    # Extract mAP metrics
    map_overall = [entry['test_coco_eval_bbox'][0] for entry in data]  # first value is mAP@[0.5:0.95]
    map_small = [entry['test_coco_eval_bbox'][3] for entry in data]    # fourth value is mAP for small objects
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot 1: Training Loss
    ax1.plot(epochs, train_losses, 'b-o', linewidth=2)
    ax1.set_title('Training Loss Over Epochs', fontsize=14)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Add annotations for loss values
    for i, loss in enumerate(train_losses):
        ax1.annotate(f'{loss:.2f}', 
                    (epochs[i], train_losses[i]), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
    
    # Plot 2: mAP Metrics
    ax2.plot(epochs, map_overall, 'g-o', linewidth=2, label='mAP@[0.5:0.95]')
    ax2.plot(epochs, map_small, 'r-^', linewidth=2, label='mAP Small Objects')
    ax2.set_title('COCO mAP Metrics Over Epochs', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('mAP', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    # Add annotations for mAP values
    for i, (map_o, map_s) in enumerate(zip(map_overall, map_small)):
        ax2.annotate(f'{map_o:.3f}', 
                    (epochs[i], map_overall[i]), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
        ax2.annotate(f'{map_s:.3f}', 
                    (epochs[i], map_small[i]), 
                    textcoords="offset points", 
                    xytext=(0,-15), 
                    ha='center')
    
    # Format x-axis ticks
    plt.xticks(epochs)
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()
    
    # Print the actual values
    print(f"Epochs: {epochs}")
    print(f"Training Loss values: {train_losses}")
    print(f"mAP@[0.5:0.95] values: {map_overall}")
    print(f"mAP Small Objects values: {map_small}")
