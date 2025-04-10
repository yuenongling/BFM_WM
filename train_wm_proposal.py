import parsing_wm
from src.wall_model import WallModel
import sys

# Get the config file name from command line arguments
config_file = sys.argv[1]
config = parsing_wm.parse_toml_config('./inputfiles/' + config_file)

# Training and saving routine
training_instance = WallModel(config)
training_instance.load_data()
training_instance.train()
r2_train, r2_valid = training_instance.test(save_path='./results')
