"""
Example usage of the redesigned wall model classes
"""
import sys
from proposal.wall_model import WallModel
import parsing_wm

def train_new_model(input_file=None):
    """Example of training a new wall model"""
    # Define configuration
    if input_file:
        # Load configuration from input file
        config = parsing_wm.parse_toml_config('./proposal/inputfiles/' + input_file)
    else:
        raise ValueError("Input file is required for training a new model.")
    
    # Create wall model
    wall_model = WallModel(config)
    
    # Load data
    wall_model.load_data()
    
    # Visualize input distribution
    wall_model.visualize_inputs(save_path='./results')
    
    # Create and train model
    wall_model.train()
    
    # Test model
    r2_train, r2_valid = wall_model.test(save_path='./results')
    print(f"Training R²: {r2_train:.4f}, Validation R²: {r2_valid:.4f}")
    
    # Visualize loss history
    wall_model.visualize_loss_history(save_path='./results')
    
    return wall_model

def load_and_test_model(model_path):
    """Example of loading and testing an existing model"""
    # Load model
    wall_model = WallModel.load_compact(model_path, device="cpu")
    
    # Test on various datasets
    test_datasets = [
        'CH',                # Channel flow
    ]
    
    # Test each dataset
    for dataset in test_datasets:
        print(f"\nTesting on {dataset}...")
        results = wall_model.test_external_dataset(
            dataset_key=dataset,
            tauw=True,
            mask_threshold=2e-4,
            save_path='./results/testing'
        )
        
        # Check fixed-height profiles for certain datasets
        if dataset in ['naca_0012', 'apg_m13n']:
            for height in [0.025, 0.05, 0.1]:
                wall_model.test_external_dataset(
                    dataset_key=dataset,
                    fixed_height=height,
                    save_path='./results/testing'
                )
    
    return wall_model

if __name__ == "__main__":
    
    # Check if a model path is provided
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        print(f"Training a new wall model with input file: {input_file}...")
        print(50 * "-")
        wall_model = train_new_model(input_file)
    else:
        print("Training a new wall model...")
        wall_model = train_new_model()

    # wall_model = load_and_test_model('./models/NN_wm_CH1_G0_S0_tn210_vn53_fds0_lds0_inputs2/NN_wm_CH1_G0_S0_tn210_vn53_fds0_lds0_inputs2_final_ep1000_tl0.00131264_vl0.00142734.pth')

    print("\nDone!")
