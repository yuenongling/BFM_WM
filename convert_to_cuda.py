import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
import os
import warnings
from src.wall_model import WallModel

# Assume cti_ffp corresponds to float32
DTYPE = torch.float32
NP_DTYPE = np.float32

def format_array_cpp_definition(name: str, data: np.ndarray, dtype_str: str = "cti_ffp", elements_per_line: int = 10) -> str:
    """Formats a numpy array into a C++ constant array definition (for .cu file)."""
    # Definition doesn't need size in brackets if initializer is present
    s = f"const {dtype_str} {name}[] = {{\n    "
    lines = []
    for i, val in enumerate(data):
        # Use standard C float representation, potentially adjust precision
        s += f"{val:.8e}f,"
        if (i + 1) % elements_per_line == 0 and (i + 1) < len(data):
            s += "\n    "
     # Remove trailing comma and potentially trailing newline/spaces
    if s.endswith(","):
        s = s[:-1]
    elif s.endswith(",\n    "):
        s = s.rstrip().rstrip(',') # More robust removal

    s += "\n};\n"
    return s

def write_mlp_params_split(
    checkpoint_path: str,
    # model_structure: nn.Module, # Pass the model structure instance
    header_filename: str,       # e.g., "mlp_params.hpp"
    source_filename: str,       # e.g., "mlp_params.cu"
    input_gains: list[float],   # Expecting list/array of size 3
    input_offsets: list[float], # Expecting list/array of size 3
    output_gain: float,         # Expecting single float
    output_offset: float,       # Expecting single float
    dtype_str: str = "cti_ffp", # The C++ type name (e.g., float, double, cti_ffp)
):
    """
    Loads an MLP model from a checkpoint, extracts weights/biases, and writes
    declarations to a header (.hpp) and definitions to a source (.cu) file.

    Args:
        checkpoint_path: Path to the PyTorch model checkpoint file (.pth).
        model_structure: An instance of the nn.Module defining the MLP architecture.
                         Expected: Linear(3->40)+ReLU -> 3x[Linear(40->40)+ReLU] -> Linear(40->1).
                         Weights will be loaded into this instance.
        header_filename: The path to the output header file (e.g., "mlp_params.hpp").
        source_filename: The path to the output source file (e.g., "mlp_params.cu").
        input_gains: List or array containing the gain for each of the 3 inputs.
        input_offsets: List or array containing the offset for each of the 3 inputs.
        output_gain: The gain value for the output.
        output_offset: The offset value for the output.
        dtype_str: The C++ data type string to use for declarations/definitions.

    Raises:
        ValueError: If the model structure doesn't match expectations, input param
                    sizes are wrong, or checkpoint loading fails.
        TypeError: If model layers are not nn.Linear where expected.
        FileNotFoundError: If the checkpoint file doesn't exist.
    """
    print(f"Loading model from {checkpoint_path}...")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Load checkpoint - adjust map_location if necessary
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    wall_model = WallModel.load_compact(checkpoint_path, device="cpu")
    model_structure = wall_model.model

    # Load state dict into the provided model structure
    # Handle potential variations in checkpoint saving (e.g., nested dicts)
    if 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
         model_state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, OrderedDict):
         model_state_dict = checkpoint
    else:
        raise ValueError("Could not find model state_dict in the checkpoint. "
                         "Please check the checkpoint structure.")

    model_structure.load_state_dict(model_state_dict)
    model = model_structure.eval() # Set model to evaluation mode
    print("Model loaded successfully.")

    # --- Validation (same as before) ---
    if not isinstance(model, nn.Module):
        raise TypeError("Input 'model_structure' must be a PyTorch nn.Module.")
    if len(input_gains) != 3 or len(input_offsets) != 3:
        raise ValueError("input_gains and input_offsets must have length 3.")

    # --- Extract Layers ---
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    # if len(linear_layers) != 5:
    #     raise ValueError(f"Expected 5 Linear layers, found {len(linear_layers)}. "
    #                      "Model structure might be different than expected (3->40->40->40->40->1).")

    l_input = linear_layers[0]
    l_hidden1 = linear_layers[1]
    l_hidden2 = linear_layers[2]
    l_hidden3 = linear_layers[3]
    l_hidden4 = linear_layers[4]
    l_output = linear_layers[5]

    # --- Verify Layer Dimensions (same as before) ---
    # if l_input.in_features != 3 or l_input.out_features != 40:
    #     raise ValueError(f"Input layer shape mismatch: expected (3, 40), got ({l_input.in_features}, {l_input.out_features})")
    # # ... (add checks for hidden1, hidden2, hidden3, output as before if desired) ...
    # if l_output.in_features != 40 or l_output.out_features != 1:
    #      raise ValueError(f"Output layer shape mismatch: expected (40, 1), got ({l_output.in_features}, {l_output.out_features})")


    # --- Network Parameters ---
    Nneurons_wall = 20
    Nlayers_wall = 6  # Total number of weight matrices/layers
    act_fn_wall = 0   # 0 for ReLU (ensure kernel matches this convention)

    # --- Calculate Correct Sizes ---
    size_input_w = 3 * Nneurons_wall                # 120
    size_hidden_w = (Nlayers_wall - 2) * Nneurons_wall * Nneurons_wall # 3 * 40 * 40 = 4800
    size_bias = (Nlayers_wall - 1) * Nneurons_wall + 1 # 4 * 40 + 1 = 161
    size_output_w_b = Nneurons_wall + 1             # 40 + 1 = 41
    size_input_std = 3
    size_output_std = 1

    # --- Prepare Weights and Biases (match expected CUDA kernel order) ---
    with torch.no_grad():
        # Extract and flatten weights/biases (same logic as before)
        input_w = l_input.weight.detach().to(DTYPE).cpu().numpy().flatten()
        input_b = l_input.bias.detach().to(DTYPE).cpu().numpy()
        hidden1_w = l_hidden1.weight.detach().to(DTYPE).cpu().numpy().flatten()
        hidden1_b = l_hidden1.bias.detach().to(DTYPE).cpu().numpy()
        hidden2_w = l_hidden2.weight.detach().to(DTYPE).cpu().numpy().flatten()
        hidden2_b = l_hidden2.bias.detach().to(DTYPE).cpu().numpy()
        hidden3_w = l_hidden3.weight.detach().to(DTYPE).cpu().numpy().flatten()
        hidden3_b = l_hidden3.bias.detach().to(DTYPE).cpu().numpy()
        hidden4_w = l_hidden4.weight.detach().to(DTYPE).cpu().numpy().flatten()
        hidden4_b = l_hidden4.bias.detach().to(DTYPE).cpu().numpy()
        output_w = l_output.weight.detach().to(DTYPE).cpu().numpy().flatten()
        output_b = l_output.bias.detach().to(DTYPE).cpu().numpy()

        hidden_layers_wall_data = np.concatenate([hidden1_w, hidden2_w, hidden3_w, hidden4_w])
        bias_wall_data = np.concatenate([input_b, hidden1_b, hidden2_b, hidden3_b, hidden4_b, output_b])
        output_layer_wall_data = np.concatenate([output_w, output_b])

    # --- Prepare Standardization Parameters ---
    input_gains_wall_data = np.array(input_gains, dtype=NP_DTYPE)
    input_offsets_wall_data = np.array(input_offsets, dtype=NP_DTYPE)
    # Store single output values as arrays of size 1 for consistency
    output_gain_wall_data = np.array([output_gain], dtype=NP_DTYPE)
    output_offset_wall_data = np.array([output_offset], dtype=NP_DTYPE)

    # --- Check Extracted Sizes vs Calculated Sizes ---
    if len(input_w) != size_input_w: warnings.warn(f"Input weight size mismatch: expected {size_input_w}, got {len(input_w)}")
    if len(hidden_layers_wall_data) != size_hidden_w: warnings.warn(f"Hidden weight size mismatch: expected {size_hidden_w}, got {len(hidden_layers_wall_data)}")
    if len(bias_wall_data) != size_bias: warnings.warn(f"Bias size mismatch: expected {size_bias}, got {len(bias_wall_data)}")
    if len(output_layer_wall_data) != size_output_w_b: warnings.warn(f"Output layer size mismatch: expected {size_output_w_b}, got {len(output_layer_wall_data)}")


    # --- Give Warning about Original Header Discrepancy ---
    print("\n" + "="*60)
    print(">>> IMPORTANT WARNING <<<")
    print("The original header file screenshot showed declarations with INCORRECT sizes:")
    print("  - input_layer_wall[40]    -> Should be [{}]".format(size_input_w))
    print("  - hidden_layers_wall[11200] -> Should be [{}]".format(size_hidden_w))
    print("  - bias_wall[320]          -> Should be [{}]".format(size_bias))
    print("The generated header file '{}' uses the CORRECT sizes.".format(header_filename))
    print("Ensure your CUDA kernel code uses these correct sizes and indexing.")
    print("The generated source file '{}' contains the definitions.".format(source_filename))
    print("="*60 + "\n")

    # --- Write Header File (.hpp) ---
    print(f"Writing header file: {header_filename}...")
    os.makedirs(os.path.dirname(header_filename) or '.', exist_ok=True)
    with open(header_filename, 'w') as f:
        guard_name = os.path.basename(header_filename).upper().replace('.', '_')
        f.write(f"#ifndef {guard_name}\n")
        f.write(f"#define {guard_name}\n\n")

        f.write("// Auto-generated by write_mlp_params_split.py\n")
        f.write(f"// Source Checkpoint: {os.path.basename(checkpoint_path)}\n")
        f.write("// Model structure: Linear(3->40)+ReLU -> 3x[Linear(40->40)+ReLU] -> Linear(40->1)\n\n")

        f.write("#include <cti_utils_gpu.hpp> // Assuming this defines cti_ffp\n")
        f.write("// Or define cti_ffp if not included elsewhere:\n")
        f.write("// using cti_ffp = float; \n\n")

        f.write("// --- Network Architecture Constants ---\n")
        f.write(f"const int Nlayers_wall = {Nlayers_wall};\n")
        f.write(f"const int Nneurons_wall = {Nneurons_wall};\n")
        f.write(f"const int act_fn_wall = {act_fn_wall}; // 0 assumed for ReLU\n\n")

        f.write("// --- Extern Declarations for Weights, Biases, and Parameters ---\n")
        f.write(f"// NOTE: Array sizes corrected based on model architecture\n\n")

        f.write(f"extern const {dtype_str} input_layer_wall[{size_input_w}];      // Input weights (3x40)\n")
        f.write(f"extern const {dtype_str} hidden_layers_wall[{size_hidden_w}];  // Hidden weights (3x 40x40)\n")
        f.write(f"extern const {dtype_str} bias_wall[{size_bias}];           // All biases (40+40+40+40+1)\n")
        f.write(f"extern const {dtype_str} output_layer_wall[{size_output_w_b}];   // Output weights + bias (40+1)\n")
        f.write("\n")
        f.write(f"extern const {dtype_str} input_gains_wall[{size_input_std}];     // Input gains (size 3)\n")
        f.write(f"extern const {dtype_str} input_offsets_wall[{size_input_std}];   // Input offsets (size 3)\n")
        f.write(f"extern const {dtype_str} output_gain_wall[{size_output_std}];    // Output gain (size 1)\n")
        f.write(f"extern const {dtype_str} output_offset_wall[{size_output_std}];  // Output offset (size 1)\n")
        f.write("\n")

        f.write(f"#endif // {guard_name}\n")
    print("Header file written successfully.")

    # --- Write Source File (.cu) ---
    print(f"Writing source file: {source_filename}...")
    os.makedirs(os.path.dirname(source_filename) or '.', exist_ok=True)
    with open(source_filename, 'w') as f:
        f.write("// Auto-generated by write_mlp_params_split.py\n")
        f.write(f"// Source Checkpoint: {os.path.basename(checkpoint_path)}\n")
        f.write(f"// Contains definitions for MLP parameters declared in {os.path.basename(header_filename)}\n\n")

        f.write(f"#include \"{os.path.basename(header_filename)}\"\n\n") # Include the header

        # Define arrays (definitions provide storage)
        f.write("// --- Input Layer Weights (3 inputs -> 40 neurons) ---\n")
        f.write(format_array_cpp_definition("input_layer_wall", input_w, dtype_str))
        f.write("\n")

        f.write("// --- Hidden Layer Weights (Layers 1, 2, 3 concatenated) ---\n")
        f.write(format_array_cpp_definition("hidden_layers_wall", hidden_layers_wall_data, dtype_str))
        f.write("\n")

        f.write("// --- Biases (Input, H1, H2, H3, Output layers concatenated) ---\n")
        f.write(format_array_cpp_definition("bias_wall", bias_wall_data, dtype_str))
        f.write("\n")

        f.write("// --- Output Layer (40 weights + 1 bias) ---\n")
        f.write(format_array_cpp_definition("output_layer_wall", output_layer_wall_data, dtype_str))
        f.write("\n")

        # Define standardization parameters
        f.write("// --- Standardization Parameters --- \n")
        f.write(format_array_cpp_definition("input_gains_wall", input_gains_wall_data, dtype_str))
        f.write(format_array_cpp_definition("input_offsets_wall", input_offsets_wall_data, dtype_str))
        f.write(format_array_cpp_definition("output_gain_wall", output_gain_wall_data, dtype_str))
        f.write(format_array_cpp_definition("output_offset_wall", output_offset_wall_data, dtype_str))
        f.write("\n")
    print("Source file written successfully.")


# ================== Example Usage ==================
if __name__ == "__main__":

    # Specify the path to your trained model checkpoint
    # <<< --- USER INPUT NEEDED --- >>>
    checkpoint_file = "./models/cleaned_up_20251208_ymax_0_15_ymin_0_0025.pth" # IMPORTANT: Update this path

    # 3. Define your standardization parameters (replace with actual values)
    # <<< --- USER INPUT NEEDED --- >>>
    example_input_gains = [1.0, 1.0, 1.0]      # Placeholder - Use actual gains
    example_input_offsets = [0.0, 0.0, 0.0]    # Placeholder - Use actual offsets
    example_output_gain = 1.0                  # Placeholder - Use actual gain
    example_output_offset = 0.0                # Placeholder - Use actual offset

    # 4. Specify the output filenames
    output_header_file = "mlp_params_1208.hpp"
    output_source_file = "mlp_params_1208.cu" # Or .cpp if not directly compiled by nvcc

    # 5. Run the export function
    try:
        # Check if placeholder checkpoint path exists before running
        if not os.path.exists(checkpoint_file) or checkpoint_file == "path/to/your/model_checkpoint.pth":
             print("="*70)
             print(">>> ERROR: Please update the 'checkpoint_file' variable in the script <<<")
             print("           with the actual path to your trained PyTorch model (.pth).")
             print("="*70)
        else:
            write_mlp_params_split(
                checkpoint_path=checkpoint_file,
                # model_structure=model_definition,
                header_filename=output_header_file,
                source_filename=output_source_file,
                input_gains=example_input_gains,
                input_offsets=example_input_offsets,
                output_gain=example_output_gain,
                output_offset=example_output_offset,
                dtype_str="cti_ffp" # Specify the C++ type name
            )
    except (ValueError, TypeError, FileNotFoundError, KeyError) as e:
        print(f"Error during export: {e}")
        print("Please ensure the checkpoint path is correct, the file contains a "
              "valid state_dict, and the model structure matches the checkpoint.")
