# Filename: export_split_cpp.py
import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
import os
import warnings

import sys
sys.path.append("../src/")
from wall_model import WallModel

# Assume cti_ffp corresponds to float32
DTYPE = torch.float32
NP_DTYPE = np.float32

def format_array_cpp_definition(name: str, data: np.ndarray, dtype_str: str = "cti_ffp", elements_per_line: int = 10) -> str:
    """Formats a numpy array into a C++ constant array definition (for .cpp file)."""
    s = f"const {dtype_str} {name}[] = {{\n    "
    lines = []
    for i, val in enumerate(data):
        s += f"{val:.8e}f,"
        if (i + 1) % elements_per_line == 0 and (i + 1) < len(data):
            s += "\n    "
    if s.endswith(","): s = s[:-1]
    elif s.endswith(",\n    "): s = s.rstrip().rstrip(',')
    s += "\n};\n"
    return s

# (Keep the write_mlp_params_split function exactly as in the previous answer,
#  just rename the source_filename argument conceptually to expect a .cpp extension,
#  the function itself doesn't care about the extension)

def write_mlp_params_split(
    checkpoint_path: str,
    # model_structure: nn.Module,
    header_filename: str,       # e.g., "mlp_params.hpp"
    source_filename: str,       # <<< Now expecting e.g., "mlp_params.cpp" >>>
    input_gains: list[float],
    input_offsets: list[float],
    output_gain: float,
    output_offset: float,
    dtype_str: str = "cti_ffp",
):
    """
    Loads MLP, extracts params, writes declarations to .hpp and definitions to .cpp.
    (Code identical to previous 'write_mlp_params_split', just clarifying source_filename use)
    """
    print(f"Loading model from {checkpoint_path}...")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    # ... (rest of model loading, validation, extraction is identical to previous version) ...
    if 'model_state_dict' in checkpoint: model_state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint: model_state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, OrderedDict): model_state_dict = checkpoint
    else: raise ValueError("Could not find model state_dict in the checkpoint.")

    wall_model = WallModel.load_compact(checkpoint_path, device="cpu")
    model_structure = wall_model.model

    model_structure.load_state_dict(model_state_dict)
    model = model_structure.eval()


    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if len(linear_layers) != 5: raise ValueError(f"Expected 5 Linear layers, found {len(linear_layers)}")
    l_input, l_hidden1, l_hidden2, l_hidden3, l_output = linear_layers

    # --- Network Parameters ---
    Nneurons_wall = l_input.out_features # 40
    Nlayers_wall = len(linear_layers) # 5
    act_fn_wall = 0   # 0 for ReLU

    # --- Calculate Correct Sizes ---
    size_input_w = l_input.in_features * Nneurons_wall       # 3 * 40 = 120
    size_hidden_w = (Nlayers_wall - 2) * Nneurons_wall * Nneurons_wall # 3 * 40 * 40 = 4800
    size_bias = (Nlayers_wall - 1) * Nneurons_wall + l_output.out_features # 4 * 40 + 1 = 161
    size_output_w_b = Nneurons_wall * l_output.out_features + l_output.out_features # 40*1 + 1 = 41
    size_input_std = l_input.in_features # 3
    size_output_std = l_output.out_features # 1

    # --- Prepare Weights and Biases ---
    with torch.no_grad():
        input_w = l_input.weight.detach().to(DTYPE).cpu().numpy().flatten()
        input_b = l_input.bias.detach().to(DTYPE).cpu().numpy()
        hidden1_w = l_hidden1.weight.detach().to(DTYPE).cpu().numpy().flatten()
        hidden1_b = l_hidden1.bias.detach().to(DTYPE).cpu().numpy()
        hidden2_w = l_hidden2.weight.detach().to(DTYPE).cpu().numpy().flatten()
        hidden2_b = l_hidden2.bias.detach().to(DTYPE).cpu().numpy()
        hidden3_w = l_hidden3.weight.detach().to(DTYPE).cpu().numpy().flatten()
        hidden3_b = l_hidden3.bias.detach().to(DTYPE).cpu().numpy()
        output_w = l_output.weight.detach().to(DTYPE).cpu().numpy().flatten()
        output_b = l_output.bias.detach().to(DTYPE).cpu().numpy()
        hidden_layers_wall_data = np.concatenate([hidden1_w, hidden2_w, hidden3_w])
        bias_wall_data = np.concatenate([input_b, hidden1_b, hidden2_b, hidden3_b, output_b])
        output_layer_wall_data = np.concatenate([output_w, output_b])

    input_gains_wall_data = np.array(input_gains, dtype=NP_DTYPE)
    input_offsets_wall_data = np.array(input_offsets, dtype=NP_DTYPE)
    output_gain_wall_data = np.array([output_gain], dtype=NP_DTYPE)
    output_offset_wall_data = np.array([output_offset], dtype=NP_DTYPE)

    # --- Print Warnings (identical code) ---
    print("\n" + "="*60)
    print(">>> IMPORTANT WARNING <<<")
    print("The original header file screenshot showed declarations with INCORRECT sizes.")
    print("The generated header file uses the CORRECT sizes.")
    print("="*60 + "\n")

    # --- Write Header File (.hpp) (identical code) ---
    print(f"Writing header file: {header_filename}...")
    os.makedirs(os.path.dirname(header_filename) or '.', exist_ok=True)
    with open(header_filename, 'w') as f:
        guard_name = os.path.basename(header_filename).upper().replace('.', '_').replace('-', '_')
        f.write(f"#ifndef {guard_name}\n")
        f.write(f"#define {guard_name}\n\n")
        f.write("// Auto-generated by export script\n\n")
        # --- Define cti_ffp if not included ---
        f.write("// Define cti_ffp if not included via cti_utils_gpu.hpp\n")
        f.write(f"using {dtype_str} = float; // Assuming cti_ffp is float\n\n")
        f.write("// --- Network Architecture Constants ---\n")
        f.write(f"const int Nlayers_wall = {Nlayers_wall};\n")
        f.write(f"const int Nneurons_wall = {Nneurons_wall};\n")
        f.write(f"const int act_fn_wall = {act_fn_wall}; // 0 assumed for ReLU\n\n")
        f.write("// --- Extern Declarations for Weights, Biases, and Parameters ---\n")
        f.write(f"extern const {dtype_str} input_layer_wall[{size_input_w}];\n")
        f.write(f"extern const {dtype_str} hidden_layers_wall[{size_hidden_w}];\n")
        f.write(f"extern const {dtype_str} bias_wall[{size_bias}];\n")
        f.write(f"extern const {dtype_str} output_layer_wall[{size_output_w_b}];\n")
        f.write(f"extern const {dtype_str} input_gains_wall[{size_input_std}];\n")
        f.write(f"extern const {dtype_str} input_offsets_wall[{size_input_std}];\n")
        f.write(f"extern const {dtype_str} output_gain_wall[{size_output_std}];\n")
        f.write(f"extern const {dtype_str} output_offset_wall[{size_output_std}];\n\n")
        f.write(f"#endif // {guard_name}\n")
    print("Header file written successfully.")

    # --- Write Source File (.cpp) (identical code, writes to source_filename) ---
    print(f"Writing source file: {source_filename}...")
    os.makedirs(os.path.dirname(source_filename) or '.', exist_ok=True)
    with open(source_filename, 'w') as f:
        f.write("// Auto-generated by export script\n")
        f.write(f"// Source Checkpoint: {os.path.basename(checkpoint_path)}\n")
        f.write(f"// Contains definitions for MLP parameters declared in {os.path.basename(header_filename)}\n\n")
        f.write(f"#include \"{os.path.basename(header_filename)}\"\n\n") # Include the header
        f.write("// --- Definitions ---\n")
        f.write(format_array_cpp_definition("input_layer_wall", input_w, dtype_str))
        f.write(format_array_cpp_definition("hidden_layers_wall", hidden_layers_wall_data, dtype_str))
        f.write(format_array_cpp_definition("bias_wall", bias_wall_data, dtype_str))
        f.write(format_array_cpp_definition("output_layer_wall", output_layer_wall_data, dtype_str))
        f.write(format_array_cpp_definition("input_gains_wall", input_gains_wall_data, dtype_str))
        f.write(format_array_cpp_definition("input_offsets_wall", input_offsets_wall_data, dtype_str))
        f.write(format_array_cpp_definition("output_gain_wall", output_gain_wall_data, dtype_str))
        f.write(format_array_cpp_definition("output_offset_wall", output_offset_wall_data, dtype_str))
    print("Source file written successfully.")

# ================== Example Usage ==================
if __name__ == "__main__":

    # Specify the path to your trained model checkpoint
    # <<< --- USER INPUT NEEDED --- >>>
    checkpoint_file = "../models/NN_wm_CH1_G0_S1_TBL1_tn543759_vn135940_fds0_lds0_customw1_inputs2_final_ep8000_tl0.03098575_vl0.03911465.pth" # IMPORTANT: Update this path

    # 3. Define your standardization parameters (replace with actual values)
    # <<< --- USER INPUT NEEDED --- >>>
    example_input_gains = [1.0, 1.0, 1.0]      # Placeholder - Use actual gains
    example_input_offsets = [0.0, 0.0, 0.0]    # Placeholder - Use actual offsets
    example_output_gain = 1.0                  # Placeholder - Use actual gain
    example_output_offset = 0.0                # Placeholder - Use actual offset

    # 4. Specify the output filenames
    output_header_file = "mlp_params.hpp"
    output_source_file = "mlp_params.cpp" # Or .cpp if not directly compiled by nvcc

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
