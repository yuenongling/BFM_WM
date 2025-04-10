"""
Main wall model module that integrates all components
"""
import os
import torch
import torch.nn as nn
import numpy as np
import pickle as pkl
from typing import Dict, Optional, Tuple, List, Union, Any
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import re

# Import our modules
from src.wall_model_base import WallModelBase
from src.data_handler import WallModelDataHandler
from src.visualization import WallModelVisualization
from src.trainer import WallModelTrainer
from src.baseline_models import LogLawPredictor, WallFunctionPredictor, EqWallModelPredictor
from wall_model_cases import STATION

class WallModel(WallModelBase):
    """
    Main wall model class that integrates data handling, training, testing, and visualization
    """
    
    def __init__(self, config: Dict):
        """
        Initialize wall model with configuration
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Initialize components
        self.data_handler = WallModelDataHandler(config)
        self.visualizer = WallModelVisualization(self._get_dataset_labels())
        
        # Set up dataset constants
        self.dataset_constants = self._load_dataset_constants()
        
        # Initialize log law predictor
        self.log_law_predictor = LogLawPredictor()
        
        # Status tracking
        self.is_trained = False
    
    def _get_dataset_labels(self) -> Dict[str, str]:
        """Get human-readable labels for datasets"""
        return {
            'CH': 'Channel Flow',
            'SYN': 'Synthetic Data',
            'gaussian_2M': 'Gaussian Bump',
            'bub_K': 'Separation Bubble K',
            'bub_A': 'Separation Bubble A',
            'bub_B': 'Separation Bubble B',
            'bub_C': 'Separation Bubble C',
            'naca_0012': 'NACA 0012 Airfoil',
            'naca_4412_4': 'NACA 4412 Airfoil (Re=400K)',
            'naca_4412_10': 'NACA 4412 Airfoil (Re=1M)',
            'apg_m13n': 'APG m13n',
            'apg_m16n': 'APG m16n',
            'apg_m18n': 'APG m18n',
            'apg_b1n': 'APG b1n',
            'apg_b2n': 'APG b2n',
            'aairfoil_10M': 'Airfoil 10M',
            'aairfoil_2M': 'Airfoil 2M',
            '-4': 'TBL -4°',
            '-3': 'TBL -3°',
            '-2': 'TBL -2°',
            '-1': 'TBL -1°',
            '5': 'TBL 5°',
            '10': 'TBL 10°',
            '15': 'TBL 15°',
            '20': 'TBL 20°',
            'curve_pg': 'Curved Pressure Gradient'
        }
    
    def _load_dataset_constants(self) -> Dict[str, Any]:
        """Load constants related to datasets"""
        # These could be loaded from a file, but for now hardcoded
        return {
            'LOG_SCALE': {
                'CH': False,
                'gaussian': True,
                'gaussian_2M': True,
                'bub': True,
                'bub_K': True,
                'bub_A': True,
                'bub_B': True,
                'bub_C': True,
                'syn': False,
                'SYN': False,
                'naca_0012': True,
                'naca_4412': True,
                'apg': True,
                'aairfoil': True,
                'TBL': True,
                'curve_pg': True
            },
            'BIN_LOCATIONS': {
                'apg': (0.4, 0.1, 0.5, 0.35),
                '20': (0.5, -0.05, 0.5, 0.35),
                'gaussian': (0.0, 0.0, 0.5, 0.35),
                'default': (0.5, 0.6, 0.5, 0.35)
            }
        }
    
    def _create_model(self) -> nn.Module:
        """Create neural network model based on configuration"""
        # Get input scaling to determine correct input dimension
        input_scaling = self.config.get('model', {}).get('inputs', {}).get('InputScaling', 1)
        
        # Determine input dimension based on input_scaling
        if input_scaling == 0:  # One-point velocity
            input_dim = 1
        elif input_scaling == 1:  # One-point velocity + up
            input_dim = 2
        elif input_scaling == 2:  # Two-points velocity + up
            input_dim = 3
        elif input_scaling == 3:  # Three-points velocity + up
            input_dim = 4
        elif input_scaling == 4:  # Two-points velocity
            input_dim = 2
        elif input_scaling == 5:  # Three-point velocity
            input_dim = 3
        else:
            # Use provided or default (2)
            input_dim = self.config.get('model', {}).get('InputDim', 2)
        
        print(f"Creating model with input dimension: {input_dim}")
        
        hidden_layers = self.config.get('model',{}).get('HiddenLayers', [32, 32])
        output_dim = self.config.get('model',{}).get('OutputDim', 1)
        activation = self.config.get('model',{}).get('Activation', 'relu')
        
        # Convert activation string to function
        if activation.lower() == 'relu':
            act_fn = nn.ReLU()
        elif activation.lower() == 'leakyrelu':
            act_fn = nn.LeakyReLU()
        elif activation.lower() == 'sigmoid':
            act_fn = nn.Sigmoid()
        else:  # Default to tanh
            act_fn = nn.Tanh()
        
        # Create layers
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(act_fn)
            prev_dim = h_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        # Print model summary
        print(50 * "-")
        print(f"Model summary:")
        print(f"  Input dimension: {input_dim}")
        print(f"  Hidden layers: {hidden_layers}")
        print(f"  Output dimension: {output_dim}")
        print(f"  Activation function: {activation}")
        print(f"  Model layers: {layers}")
        print(f"  Model device: {self.device}")
        print(f"  Model type: {type(self.model)}")
        print(50 * "-")
        
        # Create model
        self.model = nn.Sequential(*layers).to(self.device)
        return self.model
    
    def load_data(self) -> None:
        """Load and preprocess data based on configuration"""
        # Read data
        self.input, self.output = self.data_handler.read_data()
        self.input_dim = self.input.shape[1]
        
        # Store in data handler for consistency
        self.data_handler.input = self.input
        self.data_handler.output = self.output
        # self.data_handler.flow_type = self.flow_type
        self.data_handler.input_dim = self.input_dim
        
        # Preprocess data
        self.input, self.output = self.data_handler.preprocess_data()
        
        # Update preprocessing parameters
        self.input_mean = self.data_handler.input_mean
        self.input_std = self.data_handler.input_std
        
        # Get nested config values properly
        partition_config = self.config.get('data', {}).get('partition', {})
        train_ratio = partition_config.get('TrainRatio', 0.8)
        random_split = partition_config.get('RandomSplitWM', True)
        
        print(f"Splitting data with ratio {train_ratio}, random_split={random_split}")
        
        self.input_train, self.output_train, self.input_valid, self.output_valid = \
            self.data_handler.split_data(train_ratio=train_ratio, random_split=random_split)
        
        # Store the split indices
        self.train_index = self.data_handler.train_index
        self.valid_index = self.data_handler.valid_index
    
    def train(self) -> None:
        """
        Train the wall model
        
        Args:
            save_dir: Optional directory to save models
        """
        # Create model if not already created
        if self.model is None:
            self._create_model()
        
        # Convert data to tensors
        input_train_tensor, output_train_tensor, input_valid_tensor, output_valid_tensor = \
            self.data_handler.convert_to_tensors(self.device)
        
        # Create trainer
        trainer = WallModelTrainer(self.model, self.config, self.device)
        
        # Store sample counts for model naming
        trainer.n_train = len(self.train_index)
        trainer.n_valid = len(self.valid_index)
        
        # Train model
        self.model = trainer.train(
            input_train_tensor,
            output_train_tensor,
            input_valid_tensor,
            output_valid_tensor,
            data_handler=self.data_handler,
        )
        
        # Store loss history
        self.train_loss_history = trainer.train_loss_history
        self.valid_loss_history = trainer.valid_loss_history
        
        # Mark as trained
        self.is_trained = True
    
    def test(self, plot: bool = True, save_path: Optional[str] = None) -> Tuple[float, float]:
        """
        Test the model on training and validation data
        
        Args:
            plot: Whether to plot the results
            save_path: Optional path to save plots
            
        Returns:
            Tuple of (r2_train, r2_valid)
        """
        if not self.is_trained and self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Make predictions
        self.model.eval()
        
        with torch.no_grad():
            # Convert inputs to tensors if needed
            if not torch.is_tensor(self.input_train):
                input_train_tensor = torch.from_numpy(self.input_train).float().to(self.device)
                input_valid_tensor = torch.from_numpy(self.input_valid).float().to(self.device)
            else:
                input_train_tensor = self.input_train
                input_valid_tensor = self.input_valid
            
            # Compute predictions
            outputs_train_predict = self.model(input_train_tensor).squeeze()
            outputs_valid_predict = self.model(input_valid_tensor).squeeze()
        
        # Convert to numpy for metrics
        if torch.is_tensor(outputs_train_predict):
            outputs_train_predict = outputs_train_predict.cpu().detach().numpy()
            outputs_valid_predict = outputs_valid_predict.cpu().detach().numpy()
        
        # Convert targets to numpy if needed
        if torch.is_tensor(self.output_train):
            output_train_np = self.output_train.cpu().detach().numpy()
            output_valid_np = self.output_valid.cpu().detach().numpy()
        else:
            output_train_np = self.output_train
            output_valid_np = self.output_valid
        
        # Calculate R² scores
        r2_train = r2_score(output_train_np, outputs_train_predict)
        r2_valid = r2_score(output_valid_np, outputs_valid_predict)
        
        # Plot results if requested
        if plot:
            self.visualizer.plot_training_results(
                output_train_np,
                outputs_train_predict,
                output_valid_np,
                outputs_valid_predict,
                save_path=save_path
            )
            
            # Plot loss history if available
            if hasattr(self, 'train_loss_history') and hasattr(self, 'valid_loss_history'):
                self.visualizer.plot_loss_history(
                    np.array(self.train_loss_history),
                    np.array(self.valid_loss_history),
                    save_path=save_path
                )
        
        return r2_train, r2_valid
    
    def test_external_dataset(self, 
                            dataset_key: str,
                            log_scale: bool = False,
                            tauw: bool = True,
                            mask_threshold: Optional[float] = None,
                            fixed_height: Optional[float] = None,
                            abs_inputs: bool = False,
                            weighted_utau: bool = False,
                            abs_err: bool = False,
                            save_path: Optional[str] = None,
                            compare_with_loglaw: bool = True) -> Dict[str, Any]:
        """
        Test the model on an external dataset
        
        Args:
            dataset_key: Key for the dataset to test on
            log_scale: Whether to use log scale for plots
            tauw: Whether to plot wall shear stress
            mask_threshold: Optional threshold for masking near-zero values
            fixed_height: Optional fixed height ratio to test at
            abs_inputs: Whether to use absolute values of inputs
            weighted_utau: Whether to weight results by friction velocity
            abs_err: Whether to use absolute error instead of relative
            save_path: Optional path to save plots
            compare_with_loglaw: Whether to compare with log law baseline
            
        Returns:
            Dictionary of test results
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        # Check whether the data set is station specific
        # If so, we need to filter the data set to only include the station
        if 'station' in dataset_key:
            match = re.search(r'(.+)_station_(\d+)$', dataset_key) # Get the case name and station number
            if match:
                case = match.group(1) # Get the case name
                station_num = int(match.group(2)) # Get the station number
                station_x   = STATION[case][station_num]
        else:
            case = dataset_key
        
        # Load and prepare external dataset
        inputs, outputs, unnormalized_inputs, flow_type = self.data_handler.load_external_dataset(case)

        if 'station' in dataset_key:
            x = flow_type[:, 2] # Get the x-coordinates of the data set
            x_closest_to_station = np.abs(x - station_x).argmin() # Get the index of the closest x-coordinate to the station
            if np.abs(x[x_closest_to_station] - station_x) > 1e-3:
                raise ValueError(f"No data points found for station {station_num} in case {case}. Closest x-coordinate is {x[x_closest_to_station]} compared to {station_x}.")
            else:
                idx_closest_to_station = np.where(np.abs(x - x[x_closest_to_station]) < 1e-6)[0] # Get the index of the closest x-coordinate to the station
                print(idx_closest_to_station)
                inputs = inputs[idx_closest_to_station]
                outputs = outputs[idx_closest_to_station]
                unnormalized_inputs = unnormalized_inputs[idx_closest_to_station]
                flow_type = flow_type[idx_closest_to_station]


        # Flatten outputs for consistency
        outputs = outputs.flatten()

        # Print array shapes for debugging
        print("\nArray shapes:")
        print(f"inputs: {inputs.shape}")
        print(f"outputs: {outputs.shape}")
        print(f"unnormalized_inputs: {unnormalized_inputs.shape}")
        print(f"flow_type: {flow_type.shape}")
        
        # Determine bin location for this dataset
        bin_loc = self.dataset_constants['BIN_LOCATIONS'].get(
            dataset_key, 
            self.dataset_constants['BIN_LOCATIONS']['default']
        )
        
        # Default log scale setting for this dataset type
        if not log_scale:
            # Check if dataset key has a default log scale setting
            for key, value in self.dataset_constants['LOG_SCALE'].items():
                if key in dataset_key:
                    log_scale = value
                    break
        
        # Use absolute values of inputs if requested
        if abs_inputs:
            inputs = np.abs(inputs)
        
        # Standardize inputs for model prediction
        if self.input_mean is None or self.input_std is None:
            print("Warning: No preprocessing parameters found. Using raw inputs.")
            inputs_norm = inputs
        else:
            inputs_norm = (inputs - self.input_mean) / self.input_std
            inputs_norm = np.nan_to_num(inputs_norm, nan=0)
            inputs_norm[abs(inputs_norm) > 1e20] = 0
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            inputs_tensor = torch.from_numpy(inputs_norm).float().to(self.device)
            outputs_predict = self.model(inputs_tensor).squeeze().cpu().detach().numpy()

        # Convert nondimensionalized outputs to dimensionalized outputs
        for i in range(len(outputs_predict)):
            y = unnormalized_inputs[i, 0]
            nu = unnormalized_inputs[i, 2]
            u = abs(unnormalized_inputs[i, 1])
            outputs[i] = outputs[i] * nu / y
            outputs_predict[i] = outputs_predict[i] * nu / y
        
        print(f"outputs_predict: {outputs_predict.shape}")
        
        # Initialize results dictionary
        results = {
            'dataset': dataset_key,
            'log_scale': log_scale,
            'mask_threshold': mask_threshold,
            'fixed_height': fixed_height,
            'metrics': {
                'model': {
                    'mean_rel_error': 0.0,
                    'std_rel_error': 0.0,
                    'mean_abs_error': 0.0,
                    'std_abs_error': 0.0
                }
            }
        }
        
        # Test at fixed height if requested
        if fixed_height is not None:
            self.visualizer.plot_results_fixed_height(
                fixed_height=fixed_height,
                output_pred=outputs_predict,
                output_true=outputs,
                dataset=dataset_key,
                unnormalized_inputs=unnormalized_inputs,
                flow_type=flow_type,
                save_path=save_path,
                abs_err=abs_err
            )
            return results
        
        # Plot error scatter
        max_err, mean_abs_err, std_abs_err, mean_rel_err, std_rel_err = self.visualizer.plot_results_scatter_error(
            inputs=inputs,
            output=outputs_predict,
            output_true=outputs,
            dataset=dataset_key,
            log_scale=log_scale,
            abs_err=abs_err,
            bin_loc=bin_loc,
            unnormalized_inputs=unnormalized_inputs,
            flow_type=flow_type,
            weighted_utau=weighted_utau,
            tauw=tauw,
            save_path=save_path,
            mask_threshold=mask_threshold
        )
        
        # Store metrics
        results['metrics']['model'].update({
            'mean_rel_error': mean_rel_err,
            'std_rel_error': std_rel_err,
            'mean_abs_error': mean_abs_err,
            'std_abs_error': std_abs_err,
            'max_error': max_err
        })

        # Flatten outputs for comparison
        outputs = outputs.flatten()
        
        # Compare with log law if requested
        if compare_with_loglaw:
            # Calculate log law predictions
            log_predictions = np.zeros_like(outputs)
            for idx in range(len(outputs)):
                y = unnormalized_inputs[idx, 0]
                nu = unnormalized_inputs[idx, 2]
                u = abs(unnormalized_inputs[idx, 1])
                log_predictions[idx] = self._eqwm_solve(y, nu, u)
            
            # Plot log law error scatter
            mean_abs_err_log, std_abs_err_log, mean_rel_err_log, std_rel_err_log = self.visualizer.plot_results_scatter_error_loglaw(
                inputs=inputs,
                output_true=outputs,
                log_predictions=log_predictions,
                dataset=dataset_key,
                log_scale=log_scale,
                bin_loc=bin_loc,
                unnormalized_inputs=unnormalized_inputs,
                flow_type=flow_type,
                save_path=save_path,
                mask_threshold=mask_threshold,
                max_model_err=max_err
            )
            
            # Store log law metrics
            results['metrics']['loglaw'] = {
                'mean_rel_error': mean_rel_err_log,
                'std_rel_error': std_rel_err_log,
                'mean_abs_error': mean_abs_err_log,
                'std_abs_error': std_abs_err_log,
                # 'max_error': max_err_log
            }
            
            # Plot comparison bar chart
            self.visualizer.plot_comparison_bar_chart(
                model_mean=mean_rel_err,
                model_std=std_rel_err,
                loglaw_mean=mean_rel_err_log,
                loglaw_std=std_rel_err_log,
                dataset=dataset_key,
                save_path=save_path
            )
        
        return results
    
    def _eqwm_solve(self, y: float, nu: float, u: float) -> float:
        """
        Solve equilibrium wall model equation using LogLawPredictor
        
        Args:
            y: Wall distance
            nu: Kinematic viscosity
            u: Velocity
            
        Returns:
            Friction velocity
        """
        # Create input array for the predictor
        unnormalized_inputs = np.array([[y, u, nu]])
        
        # Get prediction from log law predictor
        tau_w = self.log_law_predictor.predict(unnormalized_inputs)

        # Return friction velocity (u_tau = sqrt(tau_w))
        return tau_w[0]

    def visualize_inputs(self, save_path: Optional[str] = None) -> None:
        """
        Visualize the input distribution
        
        Args:
            save_path: Optional path to save the plot
        """
        self.visualizer.plot_input_distribution(
            self.input,
            self.flow_type,
            save_path=save_path
        )
    
    def visualize_loss_history(self, save_path: Optional[str] = None) -> None:
        """
        Visualize the training loss history
        
        Args:
            save_path: Optional path to save the plot
        """
        if not hasattr(self, 'train_loss_history') or not hasattr(self, 'valid_loss_history'):
            raise ValueError("No loss history available")
        
        self.visualizer.plot_loss_history(
            np.array(self.train_loss_history),
            np.array(self.valid_loss_history),
            save_path=save_path
        )
    
    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> 'WallModel':
        """
        Load a model from the given path
        
        Args:
            path: Path to the saved model
            device: Optional device to load the model on
            
        Returns:
            Loaded wall model
        """
        return super().load(path, device)
    
    @classmethod
    def load_compact(cls, path: str, device: Optional[str] = None) -> 'WallModel':
        """
        Load a model in compact format (just what's needed for inference)
        
        Args:
            path: Path to the saved model
            device: Optional device to load the model on
            
        Returns:
            Loaded wall model with minimal components
        """
        if device is None:
            checkpoint = torch.load(path)
        else:
            checkpoint = torch.load(path, map_location=device)
        
        # Create instance with minimal config
        config = checkpoint.get('config', {})
        instance = cls(config)
        
        # Set device if specified
        if device is not None:
            instance.device = torch.device(device)
        
        # Set preprocessing parameters
        instance.input_mean = checkpoint['input_mean']
        instance.input_std = checkpoint['input_std']
        
        # Create and load model
        instance._create_model()
        instance.model.load_state_dict(checkpoint['model_state_dict'])
        instance.model.to(instance.device)
        instance.model.eval()
        
        # Mark as trained
        instance.is_trained = True
        
        return instance
