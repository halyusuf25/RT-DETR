import torch
import torch.distributed as dist
import os

# Initialize the process group
dist.init_process_group(backend="nccl")

# Get the rank of the process
rank = dist.get_rank()
torch.cuda.set_device(rank)

print(f"Rank {rank} is working on GPU {torch.cuda.current_device()}")
