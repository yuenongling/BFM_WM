// Filename: mlp_inference.hpp
#ifndef MLP_INFERENCE_HPP
#define MLP_INFERENCE_HPP

#include "mlp_params.hpp" // Provides cti_ffp type and parameter declarations
#include <vector>

/**
 * @brief Performs MLP inference using globally defined parameters.
 *
 * Assumes parameters (weights, biases, std params) are declared in
 * mlp_params.hpp and defined elsewhere (e.g., mlp_params.cpp).
 * Implements ReLU activation based on act_fn_wall.
 * Applies standardization/unstandardization matching export script assumptions.
 *
 * @param inputs Vector of 3 input features.
 * @return The single output value from the MLP.
 * @throws std::invalid_argument if input size is not 3.
 */
cti_ffp mlp_inference_cpu(const std::vector<cti_ffp>& inputs);

#endif // MLP_INFERENCE_HPP
