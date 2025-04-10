# Wall Model Redesign

This folder contains a modular redesign of the wall model class to improve maintainability and extensibility. The redesign breaks down the monolithic `TrainingWallModelClass` into smaller, more focused components with clear responsibilities.

## Components

### 1. `wall_model_base.py`
- Base class with core functionality
- Handles device selection, model saving/loading, and prediction interface
- Defines required abstract methods for subclasses

### 2. `data_handler.py`
- Handles all data-related operations
- Data loading, preprocessing, partitioning
- Tensor conversion and standardization

### 3. `visualization.py`
- All visualization functionality is isolated here
- Plot generation for training results, error distributions, profiles
- Comparison visualizations for baseline models

### 4. `trainer.py`
- Manages the training process
- Optimizers, loss functions, learning rate scheduling
- Training loop and checkpoint management
- Wandb integration

### 5. `baseline_models.py`
- Contains baseline models for comparison
- Log law, wall function, and equilibrium wall model implementations
- Consistent interface for all baseline models

### 6. `wall_model.py`
- Main class that integrates all components
- High-level methods for training, testing, and visualization
- User-friendly interface

### 7. `example_usage.py`
- Example script showing how to use the redesigned classes
- Training and testing workflows

## Key Benefits of Redesign

1. **Modularity**: Each component has a clear responsibility
2. **Extensibility**: Easy to add new features or modify existing ones without affecting other components
3. **Testability**: Components can be tested independently
4. **Maintainability**: Easier to understand and maintain smaller components
5. **Flexibility**: New visualization options can be added without modifying the core functionality

## Usage Examples

### Training a new model:

```python
from proposal.wall_model import WallModel

# Define configuration
config = {
    'general': {'Verbose': 1, 'UseWandb': False, 'Save': True, 'GpuNum': 0},
    'data': {
        'CH': 1,
        'TBL': [5, 10, 15],
        'partition': {'TrainRatio': 0.8, 'RandomSplitWM': True},
    },
    'model': {
        'InputDim': 2,
        'HiddenLayers': [64, 64, 32],
        'OutputDim': 1,
        'Activation': 'tanh',
        'inputs': {'InputScaling': 2},
        'outputs': {'OutputScaling': 1}
    },
    'training': {
        'optimizer': {'Type': 'Adam', 'LearningRate': 1e-3},
        'scheduler': {'Type': 'plateau', 'Epochs': 1000}
    }
}

# Create, train and test model
wall_model = WallModel(config)
wall_model.load_data()
wall_model.train(save_dir='./models')
r2_train, r2_valid = wall_model.test(save_path='./results')
```

### Testing an existing model:

```python
from proposal.wall_model import WallModel

# Load model
wall_model = WallModel.load_compact('models/my_model.pth', device="cpu")

# Test on various datasets
wall_model.test_external_dataset(
    dataset_key='naca_0012',
    tauw=True,
    mask_threshold=2e-4,
    save_path='./results/testing'
)

# Test at fixed heights
wall_model.test_external_dataset(
    dataset_key='apg_m13n',
    fixed_height=0.05,
    save_path='./results/testing'
)
```

## Migration Guide

To migrate from the old monolithic class to the new modular design:

1. Replace imports:
   ```python
   # Old
   from wall_model_class import TrainingWallModelClass
   
   # New
   from proposal.wall_model import WallModel
   ```

2. Loading models:
   ```python
   # Old
   model = TrainingWallModelClass.load_model_compact(model_path, device="cpu")
   
   # New
   model = WallModel.load_compact(model_path, device="cpu")
   ```

3. Testing:
   ```python
   # Old
   model.test_model_external_dataset(dataset='naca_0012', log_scale=True)
   
   # New
   model.test_external_dataset(dataset_key='naca_0012', log_scale=True)
   ```

## Future Improvements

1. Additional visualization options
2. More baseline model implementations
3. Support for different neural network architectures
4. Better error handling and user feedback
5. More comprehensive documentation