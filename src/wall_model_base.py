"""
Base class for wall model with core functionality
"""
import torch
import torch.nn as nn
import numpy as np
import os
from typing import Dict, Optional, Tuple, List, Union
import pickle as pkl

class WallModelBase:
    """Base class for wall model with core functionality"""
    
    def __init__(self, config: Dict):
        """Initialize the wall model with the given configuration"""
        self.config = config
        self.model = None
        self.device = self._get_device()
        
        # Data-related attributes
        self.input = None
        self.output = None
        self.flow_type = None
        self.input_dim = None
        
        # Preprocessing
        self.input_mean = None
        self.input_std = None
        
        # For training
        self.train_index = None
        self.valid_index = None
        self.input_train = None
        self.output_train = None
        self.input_valid = None
        self.output_valid = None
        
    def _get_device(self) -> torch.device:
        """Get the device for the model (CPU or GPU)"""
        gpu_num = self.config.get('general', {}).get('GpuNum', -1)
        if gpu_num >= 0 and torch.cuda.is_available():
            return torch.device(f'cuda:{gpu_num}')
        return torch.device('cpu')
    
    def _create_model(self) -> nn.Module:
        """Create the neural network model based on configuration"""
        # Implementation would be specific to the model architecture
        raise NotImplementedError("Subclasses must implement _create_model")
    
    def load_data(self) -> None:
        """Load data from sources as specified in the configuration"""
        raise NotImplementedError("Subclasses must implement load_data")
    
    def preprocess_data(self) -> None:
        """Preprocess data (standardization, etc.)"""
        raise NotImplementedError("Subclasses must implement preprocess_data")
    
    def train(self) -> None:
        """Train the model"""
        raise NotImplementedError("Subclasses must implement train")
    
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Make predictions with the model"""
        if self.model is None:
            raise ValueError("Model has not been created or loaded")
        
        # Convert to tensor if needed
        if not torch.is_tensor(inputs):
            inputs = torch.from_numpy(inputs).float().to(self.device)
        
        # Get predictions
        with torch.no_grad():
            predictions = self.model(inputs).squeeze().cpu().numpy()
        
        return predictions
    
    def save(self, path: str) -> None:
        """Save the model and necessary data for inference"""
        if self.model is None:
            raise ValueError("Model has not been created or loaded")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_mean': self.input_mean,
            'input_std': self.input_std,
            'config': self.config
        }, path)
    
    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> 'WallModelBase':
        """Load a model from the given path"""
        if device is None:
            checkpoint = torch.load(path)
        else:
            checkpoint = torch.load(path, map_location=device)
        
        # Create instance
        instance = cls(checkpoint['config'])
        
        # Set device if specified
        if device is not None:
            instance.device = torch.device(device)
        
        # Set preprocessing parameters
        # instance.input_mean = checkpoint['input_mean']
        # instance.input_std = checkpoint['input_std']
        
        # Create and load model
        instance._create_model()
        instance.model.load_state_dict(checkpoint['model_state_dict'])
        instance.model.to(instance.device)
        instance.model.eval()
        
        return instance
