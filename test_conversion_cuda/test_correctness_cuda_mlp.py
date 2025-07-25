import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
import os
import re  # Regular expressions for parsing
from src.wall_model import WallModel

# Assume cti_ffp corresponds to float32
DTYPE = torch.float32
NP_DTYPE = np.float32

# --- Function to Parse the .cu File ---
def parse_cpp_array_definition(line: str) -> tuple[str | None, np.ndarray | None]:
    """Parses a single line C++ array definition like 'const float name[] = { ... };'"""
    # Regex to find variable name and content within braces
    match = re.search(r'const\s+\w+\s+(\w+)\[\]\s*=\s*\{([^}]+)\};', line)
    if match:
        name = match.group(1)
        content = match.group(2)
        try:
            # Split by comma, strip whitespace and 'f' suffix, convert to float
            values = [float(v.strip().rstrip('f')) for v in content.split(',') if v.strip()]
            return name, np.array(values, dtype=NP_DTYPE)
        except ValueError as e:
            print(f"Warning: Could not parse values for array '{name}': {e}")
            return name, None
    return None, None

def load_params_from_cu(source_filename: str) -> dict[str, np.ndarray]:
    """Loads parameters by parsing the .cu source file."""
    if not os.path.exists(source_filename):
        raise FileNotFoundError(f"Source file not found: {source_filename}")

    params = {}
    print(f"Parsing parameters from {source_filename}...")
    with open(source_filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("const") and "{" in line and "}" in line:
                name, data = parse_cpp_array_definition(line)
                if name and data is not None:
                    print(f"  Loaded: {name} (shape: {data.shape})")
                    params[name] = data
    print("Parsing complete.")
    if not params:
         print(f"Warning: No parameters were loaded from {source_filename}. Check file content and format.")
    return params

# --- Manual Forward Pass Implementation ---
def manual_relu(x: np.ndarray) -> np.ndarray:
    """Manual ReLU activation."""
    return np.maximum(0, x)

def manual_forward_pass(x_input: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
    """
    Performs the forward pass manually using parameters loaded from the .cu file.

    Args:
        x_input: NumPy array of input data, shape (batch_size, 3).
        params: Dictionary containing the weights, biases, and std parameters
                loaded from the .cu file.

    Returns:
        NumPy array of output predictions.
    """
    # --- Retrieve parameters and reshape ---
    # Ensure parameter names match exactly those in the .cu file
    try:
        # Reshape weights: PyTorch Linear weights are (out_features, in_features)
        # Flattened row-major in .cu file matches kernel access W[row * N_cols + col]
        input_w = params['input_layer_wall'].reshape(40, 3)
        # Hidden weights were concatenated flattened [40, 40] matrices
        hidden_w_flat = params['hidden_layers_wall']
        hidden1_w = hidden_w_flat[0:1600].reshape(40, 40)
        hidden2_w = hidden_w_flat[1600:3200].reshape(40, 40)
        hidden3_w = hidden_w_flat[3200:4800].reshape(40, 40)
        # Biases were concatenated: input(40), h1(40), h2(40), h3(40), output(1)
        biases_flat = params['bias_wall']
        input_b = biases_flat[0:40]
        hidden1_b = biases_flat[40:80]
        hidden2_b = biases_flat[80:120]
        hidden3_b = biases_flat[120:160]
        output_b = biases_flat[160:161] # Keep as array of size 1
        # Output layer weights/bias array: weights(40), bias(1)
        output_layer = params['output_layer_wall']
        output_w = output_layer[0:40].reshape(1, 40) # Shape (1, 40)

        # Standardization parameters
        input_gains = params['input_gains_wall']     # Shape (3,)
        input_offsets = params['input_offsets_wall'] # Shape (3,)
        output_gain = params['output_gain_wall'][0]  # Get scalar value
        output_offset = params['output_offset_wall'][0] # Get scalar value

    except KeyError as e:
        raise KeyError(f"Missing parameter in loaded data: {e}. Check .cu file parsing.")
    except ValueError as e:
        raise ValueError(f"Error reshaping parameters: {e}. Check expected vs actual sizes.")


    # --- Standardization ---
    # IMPORTANT: This formula MUST match the standardization used in your CUDA kernel!
    # Example: (value - offset) * gain
    # Adjust if your kernel uses a different formula (e.g., involving means, stddevs, constants).
    print("DEBUG: Applying standardization: (x_input - input_offsets) * input_gains")
    x_std = (x_input - input_offsets) * input_gains # Element-wise if shapes broadcast

    x_std = x_input

    # --- Layer 1: Input Linear + ReLU ---
    # Linear: Z = X @ W.T + b
    z1 = x_std @ input_w.T + input_b
    a1 = manual_relu(z1)

    # --- Layer 2: Hidden Linear + ReLU ---
    z2 = a1 @ hidden1_w.T + hidden1_b
    a2 = manual_relu(z2)

    # --- Layer 3: Hidden Linear + ReLU ---
    z3 = a2 @ hidden2_w.T + hidden2_b
    a3 = manual_relu(z3)

    # --- Layer 4: Hidden Linear + ReLU ---
    z4 = a3 @ hidden3_w.T + hidden3_b
    a4 = manual_relu(z4) # Last hidden activation

    # --- Layer 5: Output Linear ---
    z_out = a4 @ output_w.T + output_b

    # --- Unstandardization ---
    # IMPORTANT: This formula MUST match the unstandardization in your CUDA kernel!
    # Example: (value / gain) + offset
    print("DEBUG: Applying unstandardization: (z_out / output_gain) + output_offset")
    output_final = output_gain

    return output_final

# ================== Test Execution ==================
if __name__ == "__main__":

    # 2. Specify paths (UPDATE THESE)
    # <<< --- USER INPUT NEEDED --- >>>
    checkpoint_file = "./models/NN_wm_CH1_G0_S1_TBL1_tn543759_vn135940_fds0_lds0_customw1_inputs2_final_ep8000_tl0.03098575_vl0.03911465.pth" # IMPORTANT: Update this path
    wall_model = WallModel.load_compact(checkpoint_file, device="cpu")
    model_definition = wall_model.model
    cu_source_file = "mlp_params.cu"                     # Path to generated .cu file

    # 3. Load PyTorch model
    print(f"Loading PyTorch model from: {checkpoint_file}")
    if not os.path.exists(checkpoint_file):
        print(f"ERROR: Checkpoint file not found at {checkpoint_file}")
        exit()
    try:
        checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
        # Adjust key if necessary based on how checkpoint was saved
        if 'model_state_dict' in checkpoint: model_state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint: model_state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, OrderedDict): model_state_dict = checkpoint
        else: raise ValueError("Cannot find state_dict in checkpoint")

        model_definition.load_state_dict(model_state_dict)
        pytorch_model = model_definition.eval() # Set to evaluation mode
        print("PyTorch model loaded.")
    except Exception as e:
        print(f"ERROR loading PyTorch model: {e}")
        exit()

    # 4. Load parameters from .cu file
    try:
        manual_params = load_params_from_cu(cu_source_file)
        if not manual_params:
             print(f"ERROR: No parameters loaded from {cu_source_file}. Cannot proceed.")
             exit()
    except Exception as e:
        print(f"ERROR loading parameters from {cu_source_file}: {e}")
        exit()

    # 5. Generate Test Data
    batch_size = 5
    # Generate random data in a typical range, or use realistic sample data
    np.random.seed(42) # for reproducibility
    test_input_np = np.random.rand(batch_size, 3).astype(NP_DTYPE) * 100 # Example range
    test_input_torch = torch.from_numpy(test_input_np).to(DTYPE)
    print(f"\nGenerated test input data (shape: {test_input_np.shape})")

    # 6. Run PyTorch Inference
    print("Running inference with PyTorch model...")
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input_torch).cpu().numpy()
    print("PyTorch inference complete.")

    # 7. Run Manual Inference
    print("Running inference with manual forward pass...")
    try:
        manual_output = manual_forward_pass(test_input_np, manual_params)
        print("Manual inference complete.")
    except Exception as e:
        print(f"ERROR during manual forward pass: {e}")
        print("Check parameter loading, reshaping, and manual calculation logic.")
        exit()

    # 8. Compare Results
    print("\n--- Comparison ---")
    print("PyTorch Output (first 5):")
    print(pytorch_output[:5])
    print("\nManual Output (first 5):")
    print(manual_output[:5])

    # Use numpy.allclose for comparison with tolerance
    # Adjust rtol (relative tolerance) and atol (absolute tolerance) as needed
    # Default: rtol=1e-05, atol=1e-08
    try:
        are_close = np.allclose(pytorch_output, manual_output, rtol=1e-5, atol=1e-7)
        print(f"\nOutputs are close (within tolerance): {are_close}")

        if not are_close:
            diff = np.abs(pytorch_output - manual_output)
            print(f"Maximum absolute difference: {np.max(diff)}")
            print(f"Mean absolute difference: {np.mean(diff)}")
            print("\nPotential issues:")
            print("- Check standardization/unstandardization formulas in manual_forward_pass.")
            print("- Verify weight/bias reshaping logic in manual_forward_pass.")
            print("- Ensure parameter parsing from .cu file is correct.")
            print("- Floating point precision differences between torch/numpy might require adjusting tolerance.")
    except Exception as e:
         print(f"\nERROR during comparison: {e}")
