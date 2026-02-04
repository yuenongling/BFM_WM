import parsing_wm
from src.wall_model import WallModel
import sys
import torch

# Get the config file name from command line arguments
config_file = sys.argv[1]
config = parsing_wm.parse_toml_config('./inputfiles/' + config_file)
gpu_num = config.get('general', {}).get('GpuNum', -1)
if gpu_num >= 0 and torch.cuda.is_available():
    device = torch.device(f'cuda:{gpu_num}') 
else:
    raise "device is incorrect"

# Restore from a checkpoint if provided
checkpoint = sys.argv[2] if len(sys.argv) > 2 else None
if checkpoint:
    # NOTE: Here for continual training, we assume that we are working on a new dataset using EWC
    # So by default, we also need to read in FIM and using it to add the EWC loss
    checkpoint_path = './models/' + checkpoint
    training_instance = WallModel.load(checkpoint_path, device=device)
    training_instance.override_config(config, checkpoint)
else:
    # Training from scratch
    training_instance = WallModel(config)

# Load and train
training_instance.load_data()
training_instance.train()

# Save testing results
r2_train, r2_valid = training_instance.test(save_path='./results')
