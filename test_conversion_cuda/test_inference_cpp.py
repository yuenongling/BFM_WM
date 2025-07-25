# Filename: test_inference_cpp.py
import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
import os
import subprocess # To run C++ code
import sys
import shutil   # To find compiler
sys.path.append("../src/")
from wall_model import WallModel

# Assume cti_ffp corresponds to float32
DTYPE = torch.float32
NP_DTYPE = np.float32

# ================== Test Configuration ==================
# <<< --- USER INPUT NEEDED --- >>>
CHECKPOINT_FILE = "../models/NN_wm_CH1_G0_S1_TBL1_tn543759_vn135940_fds0_lds0_customw1_inputs2_final_ep8000_tl0.03098575_vl0.03911465.pth" # Path to PyTorch model
HEADER_FILE = "mlp_params.hpp"
SOURCE_FILE = "mlp_params.cpp"
CPP_INFERENCE_HEADER = "mlp_inference.hpp"
CPP_INFERENCE_SOURCE = "mlp_inference.cpp"
CPP_TEST_DRIVER = "test_mlp_cpu.cpp"
CPP_EXECUTABLE = "test_mlp_cpu" # Name for the compiled C++ test program
# Optional: Specify C++ compiler
# CXX = "g++"
# CXX = "clang++"
CXX = None # Set to None to auto-detect

# Define the EXACT SAME test data here as used in test_mlp_cpu.cpp
N_TEST_SAMPLES = 5
# np.random.seed(42) # Use the same seed!
# TEST_INPUT_NP = np.random.rand(N_TEST_SAMPLES, 3).astype(NP_DTYPE) * 100 # Must match C++ test

TEST_INPUT_NP =np.array([[14.038007, 59.122353, 74.136917],
                [ 4.220639, 68.483536, 43.300095],
                [ 8.197036, 87.021194, 76.834122],
                [13.760944, 26.059240, 17.501183],
                [59.526745, 96.465652, 64.080780]], dtype=NP_DTYPE)

# Tolerances for comparison
RTOL = 1e-5
ATOL = 1e-6 # Adjust ATOL slightly higher for C++ vs Python float differences if needed

# ================== Helper Functions ==================
def find_compiler(preferred=None):
    """Tries to find a C++ compiler."""
    compilers = [preferred] if preferred else ['g++', 'clang++']
    for compiler in compilers:
        if shutil.which(compiler):
            print(f"Using C++ compiler: {compiler}")
            return compiler
    return None

def compile_cpp_test(compiler, header_files, source_files, executable_name):
    """Compiles the C++ test executable."""
    print(f"Compiling C++ test code to '{executable_name}'...")
    # Include current directory for headers, link math library
    cmd = [compiler, '-std=c++17', '-o', executable_name] + \
          source_files + ['-I.', '-lm']
    print(f"Compile command: {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Compilation successful.")
        if proc.stdout: print("Compiler output:\n", proc.stdout)
        if proc.stderr: print("Compiler warnings/errors:\n", proc.stderr) # Should be empty if check=True passed
        return True
    except FileNotFoundError:
        print(f"Error: Compiler '{compiler}' not found. Check installation and PATH.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"Error: Compilation failed with exit code {e.returncode}")
        print("Compiler output (stdout):\n", e.stdout)
        print("Compiler output (stderr):\n", e.stderr)
        return False
    except Exception as e:
        print(f"An unexpected error occurred during compilation: {e}")
        return False

def run_cpp_test(executable_name):
    """Runs the compiled C++ test and captures output."""
    print(f"Running C++ test executable './{executable_name}'...")
    executable_path = './' + executable_name
    if sys.platform.startswith("win"): # Handle windows path if necessary
         executable_path = '.\\' + executable_name + '.exe' # May need .exe suffix

    if not os.path.exists(executable_path):
        print(f"Error: Compiled executable '{executable_path}' not found.")
        return None

    try:
        proc = subprocess.run([executable_path], capture_output=True, text=True, check=True)
        print("C++ execution successful.")
        return proc.stdout
    except FileNotFoundError:
         print(f"Error: Failed to run '{executable_path}'. Ensure it was compiled correctly.")
         return None
    except subprocess.CalledProcessError as e:
        print(f"Error: C++ execution failed with exit code {e.returncode}")
        print("Output (stdout):\n", e.stdout)
        print("Output (stderr):\n", e.stderr)
        return None
    except Exception as e:
        print(f"An unexpected error occurred during C++ execution: {e}")
        return None

def parse_cpp_output(output_str):
    """Parses the newline-separated float output from the C++ program."""
    try:
        lines = output_str.strip().splitlines()
        results = np.array([float(line) for line in lines], dtype=NP_DTYPE)
        # Reshape to (batch_size, 1) to match PyTorch output shape
        return results.reshape(-1, 1)
    except Exception as e:
        print(f"Error parsing C++ output: {e}\nOutput was:\n{output_str}")
        return None

# ================== Main Test Logic ==================
if __name__ == "__main__":
    # --- Preliminary Checks ---
    if not os.path.exists(CHECKPOINT_FILE) or CHECKPOINT_FILE == "path/to/your/model_checkpoint.pth":
        print(">>> ERROR: Please update 'CHECKPOINT_FILE' variable in the script <<<")
        sys.exit(1)
    if not os.path.exists(HEADER_FILE) or not os.path.exists(SOURCE_FILE):
         print(f"ERROR: Ensure parameter files '{HEADER_FILE}' and '{SOURCE_FILE}' exist.")
         print("       Run the export script first (export_split_cpp.py).")
         sys.exit(1)
    if not os.path.exists(CPP_INFERENCE_HEADER) or not os.path.exists(CPP_INFERENCE_SOURCE):
         print(f"ERROR: Ensure C++ inference files '{CPP_INFERENCE_HEADER}' and '{CPP_INFERENCE_SOURCE}' exist.")
         sys.exit(1)
    if not os.path.exists(CPP_TEST_DRIVER):
         print(f"ERROR: Ensure C++ test driver '{CPP_TEST_DRIVER}' exists.")
         sys.exit(1)

    selected_cxx = find_compiler(CXX)
    if not selected_cxx:
        print(">>> ERROR: No C++ compiler found. Please install g++ or clang++ <<<")
        sys.exit(1)

    # --- 1. Load PyTorch Model ---
    print(f"\n--- Loading PyTorch model from: {CHECKPOINT_FILE} ---")
    try:
        wall_model = WallModel.load_compact(CHECKPOINT_FILE, device="cpu")
        model_definition = wall_model.model
        checkpoint = torch.load(CHECKPOINT_FILE, map_location=torch.device('cpu'))
        if 'model_state_dict' in checkpoint: model_state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint: model_state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, OrderedDict): model_state_dict = checkpoint
        else: raise ValueError("Cannot find state_dict in checkpoint")
        model_definition.load_state_dict(model_state_dict)
        pytorch_model = model_definition.eval()
        print("PyTorch model loaded.")
    except Exception as e:
        print(f"ERROR loading PyTorch model: {e}")
        sys.exit(1)

    # --- 2. Generate PyTorch Results ---
    #
    activation_outputs = {}
                                    
    # Store intermediate activations
    def get_activation(name):
        # Dictionary to store the activations
        def hook(module, input_tensor, output_tensor):
            # Store the output tensor.
            # .detach() prevents gradients from flowing back here.
            # .cpu() moves it to CPU (optional, if you need it there).
            # .numpy() converts to numpy (optional).
            activation_outputs[name] = output_tensor.detach().cpu()
            # Use just output_tensor.detach() if you want to keep it on the device
            # Or just output_tensor if you need gradient info (less common for inspection)
        return hook

    hook_handles = [] # To store hook handles for later removal
    for layer_index, layer in enumerate(pytorch_model):
        # Check if the layer is an instance of nn.ReLU
        if isinstance(layer, nn.ReLU):
            # Create a meaningful name (e.g., 'relu_after_layer_0', 'relu_after_layer_2')
            layer_name = f"relu_after_layer_{layer_index-1}"
            print(f"Registering hook for: {layer_name} (index {layer_index})")
            # Register the forward hook and store the handle
            handle = layer.register_forward_hook(get_activation(layer_name))
            hook_handles.append(handle)
            break # Stop after the first ReLU layer

    print("\n--- Generating PyTorch Results ---")
    test_input_torch = torch.from_numpy(TEST_INPUT_NP).to(DTYPE)
    with torch.no_grad():
        pytorch_results_np = pytorch_model(test_input_torch).cpu().numpy()
    print("PyTorch inference complete.")
    print(f"PyTorch results shape: {pytorch_results_np.shape}")

    print("\n--- Captured Activations (Outputs of ReLU Layers) ---")
    for name, activations in activation_outputs.items():
        print(f"Layer '{name}':")
        print(f"  - Shape: {activations.shape}")
        print(f"  - Values (first sample):\n{activations[0]}")

    # --- 3. Compile C++ Test Code ---
    print("\n--- Compiling C++ Code ---")
    cpp_sources = [CPP_TEST_DRIVER, CPP_INFERENCE_SOURCE, SOURCE_FILE]
    cpp_headers = [HEADER_FILE, CPP_INFERENCE_HEADER] # For reference
    if not compile_cpp_test(selected_cxx, cpp_headers, cpp_sources, CPP_EXECUTABLE):
        sys.exit(1)

    # --- 4. Run C++ Test Code ---
    print("\n--- Running C++ Inference ---")
    cpp_output_str = run_cpp_test(CPP_EXECUTABLE)
    if cpp_output_str is None:
        sys.exit(1)

    # --- 5. Parse C++ Results ---
    print("\n--- Parsing C++ Results ---")
    cpp_results_np = parse_cpp_output(cpp_output_str)
    if cpp_results_np is None:
        sys.exit(1)
    print("C++ results parsed.")
    print(f"C++ results shape: {cpp_results_np.shape}")

    # --- 6. Compare Results ---
    print("\n--- Comparing PyTorch vs C++ Results ---")
    print(f"Test Input Data (first sample): {TEST_INPUT_NP[0]}")
    print(f"PyTorch Output (first sample): {pytorch_results_np[0][0]:.8f}")
    print(f"C++ Output (first sample):     {cpp_results_np[0][0]:.8f}")

    if pytorch_results_np.shape != cpp_results_np.shape:
         print("\nERROR: Output shapes differ!")
         print(f"  PyTorch shape: {pytorch_results_np.shape}")
         print(f"  C++ shape: {cpp_results_np.shape}")
         test_passed = False
    else:
        try:
            np.testing.assert_allclose(pytorch_results_np, cpp_results_np, rtol=RTOL, atol=ATOL)
            print(f"\nSUCCESS: Outputs match within tolerance (rtol={RTOL}, atol={ATOL}).")
            test_passed = True
        except AssertionError as e:
            print("\nFAILURE: Outputs do not match within tolerance!")
            print(e)
            # Calculate and print differences
            diff = np.abs(pytorch_results_np - cpp_results_np)
            max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
            print(f"Max absolute difference: {np.max(diff):.8e} at index {max_diff_idx}")
            print(f"Mean absolute difference: {np.mean(diff):.8e}")
            print("\nPotential issues:")
            print("- Check standardization/unstandardization logic match between C++ and Python/Kernel.")
            print("- Verify C++ forward pass logic (activations, matmuls, biases).")
            print("- Differences in floating point math (compiler flags, libraries). Consider adjusting ATOL.")
            print("- Ensure test data in C++ exactly matches Python.")
            test_passed = False

    # # --- Cleanup ---
    # print("\n--- Cleaning up ---")
    # try:
    #     executable_path = CPP_EXECUTABLE
    #     if sys.platform.startswith("win"): executable_path += ".exe"
    #     if os.path.exists(executable_path):
    #         os.remove(executable_path)
    #         print(f"Removed executable: {executable_path}")
    # except OSError as e:
    #     print(f"Warning: Could not remove executable '{executable_path}': {e}")

    print("\nTest finished.")
