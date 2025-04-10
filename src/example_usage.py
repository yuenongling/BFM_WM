"""
Example usage of the redesigned wall model classes
"""
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.wall_model import WallModel
from src.baseline_models import LogLawPredictor

def train_new_model():
    """Example of training a new wall model"""
    # Define configuration
    config = {
        'general': {
            'Verbose': 1,
            'UseWandb': False,
            'Save': True,
            'SaveDir': './models',
            'GpuNum': 0
        },
        'data': {
            'CH': 1,       # Use channel flow data
            'SYN': 0,      # Don't use synthetic data
            'TBL': [5, 10, 15],  # Use TBL data with these angles
            'partition': {
                'TrainRatio': 0.8,
                'RandomSplitWM': True,
            },
            'upy': 0.2     # Upper y value for input
        },
        'model': {
            'InputDim': 2,
            'HiddenLayers': [64, 64, 32],
            'OutputDim': 1,
            'Activation': 'tanh',
            'inputs': {
                'InputScaling': 2  # Two-point velocity scaling
            },
            'outputs': {
                'OutputScaling': 1  # Standard output scaling
            },
            'LDS': {
                'lds': 0    # No label distribution smoothing
            },
            'FDS': {
                'fds': 0    # No feature distribution smoothing
            },
            'weights': {
                'custom': 0  # No custom weights
            }
        },
        'training': {
            'optimizer': {
                'Type': 'Adam',
                'LearningRate': 1e-3,
                'WeightDecay': 1e-5
            },
            'scheduler': {
                'Type': 'plateau',
                'Patience': 10,
                'Factor': 0.5,
                'Epochs': 1000,
                'PrintInterval': 10,
                'SaveInterval': 100
            }
        }
    }
    
    # Create wall model
    wall_model = WallModel(config)
    
    # Load data
    wall_model.load_data()
    
    # Visualize input distribution
    wall_model.visualize_inputs(save_path='./results')
    
    # Create and train model
    wall_model.train(save_dir='./models')
    
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
        'naca_0012',         # NACA 0012 airfoil
        'apg_m13n',          # APG m13n case
        '10',                # TBL 10°
        'bub_K'              # Separation bubble K
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
    # Create results directory
    os.makedirs('./results', exist_ok=True)
    os.makedirs('./results/testing', exist_ok=True)
    
    # Check if a model path is provided
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        print(f"Loading and testing model from: {model_path}")
        wall_model = load_and_test_model(model_path)
    else:
        print("Training a new wall model...")
        wall_model = train_new_model()
        
    print("\nDone!")
