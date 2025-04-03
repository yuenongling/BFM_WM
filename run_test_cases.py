#!/usr/bin/env python3
"""
Script to run test cases for the wall model
"""
import os
import argparse
from proposal.wall_model import WallModel

def run_test_cases(model_path, datasets=None, save_path='./results/testing'):
    """
    Run test cases for the wall model
    
    Args:
        model_path: Path to the model to test
        datasets: List of datasets to test on. If None, use default datasets
        save_path: Path to save results
    """
    # Create save directory if it doesn't exist
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
    
    # Load model
    print(f"Loading model from {model_path}...")
    wall_model = WallModel.load_compact(model_path, device="cpu")
    
    # Default datasets if none provided
    if datasets is None:
        datasets = [
            'CH',                # Channel flow
            'SYN',               # Synthetic data
        ]
    
    # Test each dataset
    for dataset in datasets:
        print(f"\nTesting on {dataset}...")
        
        # Test with default parameters
        results = wall_model.test_external_dataset(
            dataset_key=dataset,
            tauw=True,
            mask_threshold=None,
            save_path=save_path
        )
        
        # Print results
        if 'metrics' in results:
            print(f"Mean relative error: {results['metrics']['model']['mean_rel_error']:.2f}%")
            print(f"Std relative error: {results['metrics']['model']['std_rel_error']:.2f}%")
            
            if 'loglaw' in results['metrics']:
                print(f"Log law mean relative error: {results['metrics']['loglaw']['mean_rel_error']:.2f}%")
                print(f"Log law std relative error: {results['metrics']['loglaw']['std_rel_error']:.2f}%")
        
        # Check fixed-height profiles for certain datasets
        # if dataset in ['naca_0012', 'apg_m13n', 'bub_K']:
        #     for height in [0.025, 0.05, 0.1]:
        #         print(f"Testing fixed height {height} for {dataset}...")
        #         wall_model.test_external_dataset(
        #             dataset_key=dataset,
        #             fixed_height=height,
        #             save_path=save_path
        #         )
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run test cases for the wall model")
    parser.add_argument("--model", type=str, required=True, help="Path to the model to test")
    parser.add_argument("--datasets", type=str, nargs="+", help="Datasets to test on")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save results")
    
    args = parser.parse_args()
    
    run_test_cases(args.model, args.datasets, args.save_path) 
