// Filename: test_mlp_cpu.cpp
#include "mlp_inference.hpp" // Includes mlp_params.hpp indirectly
#include <iostream>
#include <vector>
#include <iomanip> // For std::fixed, std::setprecision

#define cti_ffp float // Define cti_ffp as float for this example

int main() {
    // --- Define Test Inputs ---
    // Use the EXACT SAME values as generated in the Python test script
    // Example values - replace with actual generated values from Python test
    std::vector<std::vector<cti_ffp>> test_inputs = {
        // Batch size 5, copied from NumPy output in Python script
        {14.038007f, 59.122353f, 74.136917f},
        { 4.220639f, 68.483536f, 43.300095f},
        { 8.197036f, 87.021194f, 76.834122f},
        {13.760944f, 26.059240f, 17.501183f},
        {59.526745f, 96.465652f, 64.080780f}
    };

    std::cout << std::fixed << std::setprecision(8); // Set consistent output format

    // --- Run Inference for each input ---
    for (const auto& input_vec : test_inputs) {
        try {
            cti_ffp result = mlp_inference_cpu(input_vec);
            std::cout << result << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error during inference: " << e.what() << std::endl;
            return 1; // Indicate error
        }
    }

    return 0; // Indicate success
}
