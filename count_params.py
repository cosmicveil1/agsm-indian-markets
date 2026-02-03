import torch
from models.agsm_lob import AGSMNetLOB

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# User's current config: dim=16, state_dim=16
model = AGSMNetLOB(
    in_channels=1, 
    dim=16, 
    state_dim=16, 
    num_classes=3,
    num_blocks=2 # default
)

total_params = count_parameters(model)
print(f"Total Trainable Parameters: {total_params:,}")
print(f"Model Configuration: dim=16, state_dim=16, blocks=2")
