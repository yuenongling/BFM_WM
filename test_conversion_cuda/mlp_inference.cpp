// Filename: mlp_inference.cpp
#include "mlp_inference.hpp"
#include <vector>
#include <cmath>     // For std::fmax, std::tanh, std::exp
#include <stdexcept> // For std::invalid_argument
#include <numeric>   // Potentially useful, e.g. std::inner_product if optimized
#include <iostream>  // For debugging output, can be removed in production

#define cti_ffp float
// Note: Parameters like input_layer_wall, Nneurons_wall etc. are accessed
// directly as they are declared extern const in mlp_params.hpp

cti_ffp mlp_inference_cpu(const std::vector<cti_ffp>& inputs) {
    // --- Input Validation ---
    if (inputs.size() != 3) {
        throw std::invalid_argument("Input vector must have size 3.");
    }

    // --- Standardization ---
    // IMPORTANT: Formula must match export script and intended kernel logic.
    // Example: (value - offset) * gain
    std::vector<cti_ffp> x_std(3);
    for (int i = 0; i < 3; ++i) {
        // Check if gain is near zero if dividing
        // Assuming gain is multiplicative factor here
        /*x_std[i] = (inputs[i] - input_offsets_wall[i]) * input_gains_wall[i];*/
        x_std[i] = inputs[i]; // No offset
        // If your standardization included a final '-1.0' like in the kernel template:
        // x_std[i] = (((inputs[i] - mean_placeholder)/stddev_placeholder) - offset_placeholder) * gain_placeholder - 1.0f;
    }

    /*std::cout << "Standardized Inputs: ";*/
    /*std::cout << x_std[0] << " " << x_std[1] << " " << x_std[2] << std::endl;*/

    // --- Layer Buffers ---
    std::vector<cti_ffp> activations(Nneurons_wall);
    std::vector<cti_ffp> next_activations(Nneurons_wall); // For hidden layers

    // --- Input Layer (Layer 0) ---
    for (int i = 0; i < Nneurons_wall; ++i) { // Output neuron index
        cti_ffp sum = 0.0f;
        for (int j = 0; j < 3; ++j) { // Input feature index
            // Accessing flattened input weights: W[neuron_out * n_in + idx_in]
            sum += input_layer_wall[i * 3 + j] * x_std[j];
        }
        /**/
        /*if (i == 0){*/
        /*  std::cout << "The full arithmetic for input layer neuron " << i << " is: ";*/
        /*  std::cout << "sum = ";*/
        /*  std::cout << input_layer_wall[i * 3 + 0] << " * " << x_std[0] << " + ";*/
        /*  std::cout << input_layer_wall[i * 3 + 1] << " * " << x_std[1] << " + ";*/
        /*  std::cout << input_layer_wall[i * 3 + 2] << " * " << x_std[2] << " + ";*/
        /*  std::cout << bias_wall[i] << " = ";*/
        /*  std::cout << sum << std::endl;*/
        /*}*/

        // Bias for input layer (assuming first Nneurons_wall elements of bias_wall)
        sum += bias_wall[i];

        // Activation
        if (act_fn_wall == 0) { // ReLU
            activations[i] = std::fmax(sum, 0.0f);
        } else { // Example: Tanh (adjust if different activation used)
            activations[i] = std::tanh(sum);
        }
    }

    /*std::cout << "Input Layer Activations: "; */
    /**/
    /*// Print all activations for debugging*/
    /*for (const auto& act : activations) {*/
    /*  std::cout << act << " ";*/
    /*}*/

    // --- Hidden Layers (Layers 1 to Nlayers_wall - 2) ---
    // Nlayers_wall includes input and output linear layers.
    // Hidden weight matrices correspond to layers 1, 2, ..., Nlayers_wall-2
    // Loop index 'l' goes from 0 to Nlayers_wall-3
    const int n_hidden_matrices = Nlayers_wall - 2; // = 3 for 5 total layers
    for (int l = 0; l < n_hidden_matrices; ++l) {
        for (int i = 0; i < Nneurons_wall; ++i) { // Output neuron for this layer
            cti_ffp sum = 0.0f;
            for (int j = 0; j < Nneurons_wall; ++j) { // Input neuron from previous layer
                // Accessing flattened hidden weights: hidden_W[matrix_idx][neuron_out][neuron_in]
                int weight_idx = l * Nneurons_wall * Nneurons_wall + i * Nneurons_wall + j;
                sum += hidden_layers_wall[weight_idx] * activations[j]; // Use previous layer's activations
            }
            // Accessing bias for this hidden layer (layer index l+1)
            // Bias array layout: [input_bias (40), h1_bias (40), h2_bias (40), h3_bias (40), out_bias (1)]
            int bias_idx = (l + 1) * Nneurons_wall + i;
            sum += bias_wall[bias_idx];

            // Activation
            if (act_fn_wall == 0) { // ReLU
                next_activations[i] = std::fmax(sum, 0.0f);
            } else { // Example: Tanh
                 next_activations[i] = std::tanh(sum);
            }
        }
        activations = next_activations; // Output activations become input for next layer
    } // End hidden layer loop
      
    /*std::cout << "Before output"; */
    /**/
    /*// Print all activations for debugging*/
    /*for (const auto& act : activations) {*/
    /*  std::cout << act << " ";*/
    /*}*/

    // --- Output Layer (Layer Nlayers_wall - 1) ---
    cti_ffp output_sum = 0.0f;
    for (int i = 0; i < Nneurons_wall; ++i) {
    /*    // output_layer_wall contains weights[0..Nneurons_wall-1] then bias[Nneurons_wall]*/
        output_sum += output_layer_wall[i] * activations[i]; // Use last hidden layer's activations
    }
    // Add final bias (last element of output_layer_wall)
    output_sum += output_layer_wall[Nneurons_wall];

    // --- Unstandardization ---
    // IMPORTANT: Formula must match export script and intended kernel logic.
    // Example: (value / gain) + offset
    // Check for division by zero if gain can be zero
    cti_ffp final_output = output_sum; // Start with raw output
    /*if (std::fabs(output_gain_wall[0]) > 1e-9f) { // Avoid division by zero*/
    /*     final_output = final_output / output_gain_wall[0];*/
    /*} else {*/
    /*    // Handle case of zero gain if necessary (e.g., return raw output, throw error)*/
    /*}*/
    /*final_output += output_offset_wall[0];*/

    // If your unstandardization used a different formula (e.g., involving *output_std_dev + output_mean) adapt here.

    return final_output;
}
