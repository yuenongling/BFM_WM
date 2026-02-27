import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
import os
import warnings
from typing import Optional

# Assume cti_ffp corresponds to float32
DTYPE = torch.float32
NP_DTYPE = np.float32


def threshold_small_values(
    data: np.ndarray, 
    threshold: float = 1e-10
) -> np.ndarray:
    """
    Set values with absolute value smaller than threshold to zero.
    This helps avoid potential overflow/underflow issues.
    
    Args:
        data: Input numpy array
        threshold: Values with |value| < threshold are set to 0
        
    Returns:
        Array with small values zeroed out
    """
    result = data.copy()
    mask = np.abs(result) < threshold
    n_zeroed = np.sum(mask)
    if n_zeroed > 0:
        result[mask] = 0.0
        print(f"    Zeroed {n_zeroed} values with |value| < {threshold}")
    return result


def format_array_cpp_definition(
    name: str, 
    data: np.ndarray, 
    dtype_str: str = "cti_ffp", 
    elements_per_line: int = 10
) -> str:
    """Formats a numpy array into a C++ constant array definition (for .cu file)."""
    s = f"const {dtype_str} {name}[] = {{\n    "
    for i, val in enumerate(data):
        s += f"{val:.8e}f,"
        if (i + 1) % elements_per_line == 0 and (i + 1) < len(data):
            s += "\n    "
    
    # Remove trailing comma
    s = s.rstrip()
    if s.endswith(","):
        s = s[:-1]
    
    s += "\n};\n"
    return s


def extract_linear_layers(model: nn.Module) -> list[nn.Linear]:
    """Extract all Linear layers from a model in order."""
    return [m for m in model.modules() if isinstance(m, nn.Linear)]


def infer_model_structure(linear_layers: list[nn.Linear]) -> dict:
    """
    Infer the model structure from the linear layers.
    
    Returns:
        dict with keys:
            - n_inputs: number of input features
            - n_outputs: number of output features  
            - n_neurons: number of neurons in hidden layers (assumes constant)
            - n_layers: total number of linear layers
            - n_hidden_layers: number of hidden layers (excluding input and output)
    """
    if len(linear_layers) < 2:
        raise ValueError(f"Model must have at least 2 linear layers (input and output), found {len(linear_layers)}")
    
    n_inputs = linear_layers[0].in_features
    n_outputs = linear_layers[-1].out_features
    n_neurons = linear_layers[0].out_features  # Assume constant neuron size
    n_layers = len(linear_layers)
    n_hidden_layers = n_layers - 2  # Exclude input and output layers
    
    # Validate constant neuron assumption
    for i, layer in enumerate(linear_layers[:-1]):  # All except output
        if layer.out_features != n_neurons:
            warnings.warn(
                f"Layer {i} has {layer.out_features} outputs, expected {n_neurons}. "
                "Neuron count may not be constant."
            )
    
    for i, layer in enumerate(linear_layers[1:], start=1):  # All except input
        if i < len(linear_layers) - 1:  # Hidden layers
            if layer.in_features != n_neurons:
                warnings.warn(
                    f"Layer {i} has {layer.in_features} inputs, expected {n_neurons}. "
                    "Neuron count may not be constant."
                )
    
    return {
        'n_inputs': n_inputs,
        'n_outputs': n_outputs,
        'n_neurons': n_neurons,
        'n_layers': n_layers,
        'n_hidden_layers': n_hidden_layers,
    }


def load_model_from_checkpoint(checkpoint_path: str, model_loader=None) -> nn.Module:
    """
    Load a model from a checkpoint file.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model_loader: Optional callable that takes checkpoint_path and returns a model.
                     If None, attempts to load state_dict directly.
    
    Returns:
        The loaded model in eval mode
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    if model_loader is not None:
        model = model_loader(checkpoint_path)
    else:
        # Try to load directly - user must provide model_loader for custom formats
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        if isinstance(checkpoint, nn.Module):
            model = checkpoint
        else:
            raise ValueError(
                "Checkpoint is not a direct model. Please provide a model_loader function "
                "that takes the checkpoint path and returns the loaded model."
            )
    
    return model.eval()


def write_mlp_params_split(
    checkpoint_path: str,
    header_filename: str,
    source_filename: str,
    input_gains: list[float],
    input_offsets: list[float],
    output_gain: float,
    output_offset: float,
    dtype_str: str = "cti_ffp",
    model_loader=None,
    act_fn_id: int = 0,  # 0 for ReLU, can be customized
    zero_threshold: float = 1e-10,  # Values with |val| < threshold are set to 0
):
    """
    Loads an MLP model from a checkpoint, extracts weights/biases, and writes
    declarations to a header (.hpp) and definitions to a source (.cu) file.
    
    This version automatically infers the model structure and handles any number
    of layers with constant neuron size.

    Args:
        checkpoint_path: Path to the PyTorch model checkpoint file (.pth).
        header_filename: The path to the output header file (e.g., "mlp_params.hpp").
        source_filename: The path to the output source file (e.g., "mlp_params.cu").
        input_gains: List or array containing the gain for each input.
        input_offsets: List or array containing the offset for each input.
        output_gain: The gain value for the output.
        output_offset: The offset value for the output.
        dtype_str: The C++ data type string to use for declarations/definitions.
        model_loader: Optional callable(checkpoint_path) -> nn.Module. 
                     Use this for custom model loading logic.
        act_fn_id: Activation function identifier (0 for ReLU by default).
        zero_threshold: Values with absolute value smaller than this are set to 0.
                       Helps avoid overflow/underflow issues. Default: 1e-10.

    Raises:
        ValueError: If the model structure doesn't match expectations or input param
                    sizes are wrong.
        FileNotFoundError: If the checkpoint file doesn't exist.
    """
    print(f"Loading model from {checkpoint_path}...")
    
    # Load the model
    model = load_model_from_checkpoint(checkpoint_path, model_loader)
    print("Model loaded successfully.")
    
    # Extract linear layers
    linear_layers = extract_linear_layers(model)
    print(f"Found {len(linear_layers)} linear layers.")
    
    # Infer structure
    structure = infer_model_structure(linear_layers)
    n_inputs = structure['n_inputs']
    n_outputs = structure['n_outputs']
    n_neurons = structure['n_neurons']
    n_layers = structure['n_layers']
    n_hidden_layers = structure['n_hidden_layers']
    
    print(f"Inferred structure:")
    print(f"  - Inputs: {n_inputs}")
    print(f"  - Outputs: {n_outputs}")
    print(f"  - Neurons per hidden layer: {n_neurons}")
    print(f"  - Total linear layers: {n_layers}")
    print(f"  - Hidden layers: {n_hidden_layers}")
    
    # Validate input parameters
    if len(input_gains) != n_inputs:
        raise ValueError(f"input_gains must have length {n_inputs}, got {len(input_gains)}")
    if len(input_offsets) != n_inputs:
        raise ValueError(f"input_offsets must have length {n_inputs}, got {len(input_offsets)}")
    
    # Calculate sizes
    size_input_w = n_inputs * n_neurons
    size_hidden_w = n_hidden_layers * n_neurons * n_neurons
    # Biases: one per neuron in each layer except output has n_outputs biases
    size_bias = n_hidden_layers * n_neurons + n_neurons + n_outputs  # hidden + input layer + output
    size_output_w_b = n_neurons * n_outputs + n_outputs  # weights + biases
    
    print(f"\nCalculated array sizes:")
    print(f"  - input_layer_wall: {size_input_w}")
    print(f"  - hidden_layers_wall: {size_hidden_w}")
    print(f"  - bias_wall: {size_bias}")
    print(f"  - output_layer_wall: {size_output_w_b}")
    
    # Extract weights and biases
    with torch.no_grad():
        # Input layer
        input_layer = linear_layers[0]
        input_w = input_layer.weight.detach().to(DTYPE).cpu().numpy().flatten()
        input_b = input_layer.bias.detach().to(DTYPE).cpu().numpy()
        
        # Hidden layers
        hidden_weights = []
        hidden_biases = []
        for i in range(1, n_layers - 1):  # Skip input and output
            layer = linear_layers[i]
            hidden_weights.append(layer.weight.detach().to(DTYPE).cpu().numpy().flatten())
            hidden_biases.append(layer.bias.detach().to(DTYPE).cpu().numpy())
        
        # Output layer
        output_layer = linear_layers[-1]
        output_w = output_layer.weight.detach().to(DTYPE).cpu().numpy().flatten()
        output_b = output_layer.bias.detach().to(DTYPE).cpu().numpy()
        
        # Apply threshold to avoid overflow/underflow
        print(f"\nApplying zero threshold ({zero_threshold}) to weights and biases...")
        print("  Input layer weights:")
        input_w = threshold_small_values(input_w, zero_threshold)
        print("  Input layer biases:")
        input_b = threshold_small_values(input_b, zero_threshold)
        
        for i in range(len(hidden_weights)):
            print(f"  Hidden layer {i+1} weights:")
            hidden_weights[i] = threshold_small_values(hidden_weights[i], zero_threshold)
        for i in range(len(hidden_biases)):
            print(f"  Hidden layer {i+1} biases:")
            hidden_biases[i] = threshold_small_values(hidden_biases[i], zero_threshold)
        
        print("  Output layer weights:")
        output_w = threshold_small_values(output_w, zero_threshold)
        print("  Output layer biases:")
        output_b = threshold_small_values(output_b, zero_threshold)
        
        # Concatenate
        if hidden_weights:
            hidden_layers_wall_data = np.concatenate(hidden_weights)
        else:
            hidden_layers_wall_data = np.array([], dtype=NP_DTYPE)
        
        # All biases concatenated: input, hidden..., output
        all_biases = [input_b] + hidden_biases + [output_b]
        bias_wall_data = np.concatenate(all_biases)
        
        # Output layer: weights + bias
        output_layer_wall_data = np.concatenate([output_w, output_b])
    
    # Prepare standardization parameters
    input_gains_wall_data = np.array(input_gains, dtype=NP_DTYPE)
    input_offsets_wall_data = np.array(input_offsets, dtype=NP_DTYPE)
    output_gain_wall_data = np.array([output_gain], dtype=NP_DTYPE)
    output_offset_wall_data = np.array([output_offset], dtype=NP_DTYPE)
    
    # Verify sizes
    actual_sizes = {
        'input_w': len(input_w),
        'hidden_w': len(hidden_layers_wall_data),
        'bias': len(bias_wall_data),
        'output': len(output_layer_wall_data),
    }
    expected_sizes = {
        'input_w': size_input_w,
        'hidden_w': size_hidden_w,
        'bias': size_bias,
        'output': size_output_w_b,
    }
    
    for key in actual_sizes:
        if actual_sizes[key] != expected_sizes[key]:
            warnings.warn(
                f"{key} size mismatch: expected {expected_sizes[key]}, got {actual_sizes[key]}"
            )
    
    # Write header file
    print(f"\nWriting header file: {header_filename}...")
    os.makedirs(os.path.dirname(header_filename) or '.', exist_ok=True)
    
    with open(header_filename, 'w') as f:
        guard_name = os.path.basename(header_filename).upper().replace('.', '_').replace('-', '_')
        f.write(f"#ifndef {guard_name}\n")
        f.write(f"#define {guard_name}\n\n")
        
        f.write("// Auto-generated by write_mlp_params_generic.py\n")
        f.write(f"// Source Checkpoint: {os.path.basename(checkpoint_path)}\n")
        f.write(f"// Model structure: Linear({n_inputs}->{n_neurons})+Act -> "
                f"{n_hidden_layers}x[Linear({n_neurons}->{n_neurons})+Act] -> "
                f"Linear({n_neurons}->{n_outputs})\n\n")
        
        f.write("#include <cti_utils_gpu.hpp> // Assuming this defines cti_ffp\n")
        f.write("// Or define cti_ffp if not included elsewhere:\n")
        f.write("// using cti_ffp = float; \n\n")
        
        f.write("// --- Network Architecture Constants ---\n")
        f.write(f"const int Nlayers_wall = {n_layers};\n")
        f.write(f"const int Nneurons_wall = {n_neurons};\n")
        f.write(f"const int Ninputs_wall = {n_inputs};\n")
        f.write(f"const int Noutputs_wall = {n_outputs};\n")
        f.write(f"const int act_fn_wall = {act_fn_id}; // Activation function ID\n\n")
        
        f.write("// --- Extern Declarations for Weights, Biases, and Parameters ---\n\n")
        
        f.write(f"extern const {dtype_str} input_layer_wall[{size_input_w}];      "
                f"// Input weights ({n_inputs}x{n_neurons})\n")
        if size_hidden_w > 0:
            f.write(f"extern const {dtype_str} hidden_layers_wall[{size_hidden_w}];  "
                    f"// Hidden weights ({n_hidden_layers}x {n_neurons}x{n_neurons})\n")
        f.write(f"extern const {dtype_str} bias_wall[{size_bias}];           "
                f"// All biases\n")
        f.write(f"extern const {dtype_str} output_layer_wall[{size_output_w_b}];   "
                f"// Output weights + bias ({n_neurons}x{n_outputs}+{n_outputs})\n")
        f.write("\n")
        f.write(f"extern const {dtype_str} input_gains_wall[{n_inputs}];     // Input gains\n")
        f.write(f"extern const {dtype_str} input_offsets_wall[{n_inputs}];   // Input offsets\n")
        f.write(f"extern const {dtype_str} output_gain_wall[{n_outputs}];    // Output gain\n")
        f.write(f"extern const {dtype_str} output_offset_wall[{n_outputs}];  // Output offset\n")
        f.write("\n")
        
        f.write(f"#endif // {guard_name}\n")
    
    print("Header file written successfully.")
    
    # Write source file
    print(f"Writing source file: {source_filename}...")
    os.makedirs(os.path.dirname(source_filename) or '.', exist_ok=True)
    
    with open(source_filename, 'w') as f:
        f.write("// Auto-generated by write_mlp_params_generic.py\n")
        f.write(f"// Source Checkpoint: {os.path.basename(checkpoint_path)}\n")
        f.write(f"// Contains definitions for MLP parameters declared in {os.path.basename(header_filename)}\n\n")
        
        f.write(f"#include \"{os.path.basename(header_filename)}\"\n\n")
        
        f.write(f"// --- Input Layer Weights ({n_inputs} inputs -> {n_neurons} neurons) ---\n")
        f.write(format_array_cpp_definition("input_layer_wall", input_w, dtype_str))
        f.write("\n")
        
        if size_hidden_w > 0:
            f.write(f"// --- Hidden Layer Weights ({n_hidden_layers} layers concatenated) ---\n")
            f.write(format_array_cpp_definition("hidden_layers_wall", hidden_layers_wall_data, dtype_str))
            f.write("\n")
        
        f.write("// --- Biases (All layers concatenated) ---\n")
        f.write(format_array_cpp_definition("bias_wall", bias_wall_data, dtype_str))
        f.write("\n")
        
        f.write(f"// --- Output Layer ({n_neurons} weights x {n_outputs} outputs + {n_outputs} bias) ---\n")
        f.write(format_array_cpp_definition("output_layer_wall", output_layer_wall_data, dtype_str))
        f.write("\n")
        
        f.write("// --- Standardization Parameters --- \n")
        f.write(format_array_cpp_definition("input_gains_wall", input_gains_wall_data, dtype_str))
        f.write(format_array_cpp_definition("input_offsets_wall", input_offsets_wall_data, dtype_str))
        f.write(format_array_cpp_definition("output_gain_wall", output_gain_wall_data, dtype_str))
        f.write(format_array_cpp_definition("output_offset_wall", output_offset_wall_data, dtype_str))
        f.write("\n")
    
    print("Source file written successfully.")
    
    return structure  # Return the inferred structure for reference


# ================== Command Line Interface ==================
if __name__ == "__main__":
    import argparse
    
    def wall_model_loader(checkpoint_path):
        """Custom loader for WallModel checkpoints."""
        from src.wall_model import WallModel
        wall_model = WallModel.load_compact(checkpoint_path, device="cpu")
        return wall_model.model
    
    parser = argparse.ArgumentParser(
        description="Export PyTorch MLP weights to C++ header and source files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to the PyTorch model checkpoint file (.pth)"
    )
    parser.add_argument(
        "output",
        type=str,
        help="Output base filename (without extension). Will generate <output>.hpp and <output>.cu"
    )
    
    # Optional arguments
    parser.add_argument(
        "--input-gains",
        type=float,
        nargs="+",
        default=None,
        help="Input gain values (space-separated). If not provided, defaults to 1.0 for each input."
    )
    parser.add_argument(
        "--input-offsets",
        type=float,
        nargs="+",
        default=None,
        help="Input offset values (space-separated). If not provided, defaults to 0.0 for each input."
    )
    parser.add_argument(
        "--output-gain",
        type=float,
        default=1.0,
        help="Output gain value"
    )
    parser.add_argument(
        "--output-offset",
        type=float,
        default=0.0,
        help="Output offset value"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="cti_ffp",
        help="C++ data type string to use in generated files"
    )
    parser.add_argument(
        "--act-fn",
        type=int,
        default=0,
        help="Activation function ID (0=ReLU)"
    )
    parser.add_argument(
        "--header-ext",
        type=str,
        default=".hpp",
        help="Header file extension"
    )
    parser.add_argument(
        "--source-ext",
        type=str,
        default=".cu",
        help="Source file extension"
    )
    parser.add_argument(
        "--zero-threshold",
        type=float,
        default=1e-10,
        help="Values with |value| < threshold are set to 0 (avoids overflow/underflow)"
    )
    
    args = parser.parse_args()
    
    # Generate output filenames
    output_base = args.output
    # Remove extension if user provided one
    for ext in ['.hpp', '.h', '.cu', '.cpp', '.cxx']:
        if output_base.endswith(ext):
            output_base = output_base[:-len(ext)]
            break
    
    header_file = output_base + args.header_ext
    source_file = output_base + args.source_ext
    
    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        exit(1)
    
    # First, load model to infer input size for default gains/offsets
    print(f"Loading model to infer structure...")
    try:
        model = load_model_from_checkpoint(args.checkpoint, wall_model_loader)
        linear_layers = extract_linear_layers(model)
        structure = infer_model_structure(linear_layers)
        n_inputs = structure['n_inputs']
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)
    
    # Set default gains/offsets based on inferred input size
    input_gains = args.input_gains if args.input_gains is not None else [1.0] * n_inputs
    input_offsets = args.input_offsets if args.input_offsets is not None else [0.0] * n_inputs
    
    # Validate input gains/offsets length
    if len(input_gains) != n_inputs:
        print(f"Error: --input-gains must have {n_inputs} values (model has {n_inputs} inputs), got {len(input_gains)}")
        exit(1)
    if len(input_offsets) != n_inputs:
        print(f"Error: --input-offsets must have {n_inputs} values (model has {n_inputs} inputs), got {len(input_offsets)}")
        exit(1)
    
    # Run export
    try:
        structure = write_mlp_params_split(
            checkpoint_path=args.checkpoint,
            header_filename=header_file,
            source_filename=source_file,
            input_gains=input_gains,
            input_offsets=input_offsets,
            output_gain=args.output_gain,
            output_offset=args.output_offset,
            dtype_str=args.dtype,
            model_loader=wall_model_loader,
            act_fn_id=args.act_fn,
            zero_threshold=args.zero_threshold,
        )
        print(f"\n{'='*60}")
        print("Export complete!")
        print(f"  Header: {header_file}")
        print(f"  Source: {source_file}")
        print(f"  Structure: {structure}")
        print(f"{'='*60}")
    except Exception as e:
        print(f"Error during export: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
