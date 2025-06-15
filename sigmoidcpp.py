import numpy as np
import dyson
from dyson import DysonRouter
import os

router = DysonRouter()
os.environ["dyson_api"] = "dyson_api_key_here"
# Define the C++ code with array operations
cpp_code = """
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <float.h>

extern "C" {
    // Vector operations that take arrays as input and return arrays
    
    // Apply sigmoid activation to an entire array
    void sigmoid_array(float* input, float* output, int size) {
        for (int i = 0; i < size; i++) {
            output[i] = 1.0f / (1.0f + exp(-input[i]));
        }
    }
    
    // Batch normalization (simplified version)
    void batch_norm(float* input, float* output, int size, float epsilon) {
        // Calculate mean
        float mean = 0.0f;
        for (int i = 0; i < size; i++) {
            mean += input[i];
        }
        mean /= size;
        
        // Calculate variance
        float variance = 0.0f;
        for (int i = 0; i < size; i++) {
            float diff = input[i] - mean;
            variance += diff * diff;
        }
        variance /= size;
        
        // Normalize
        for (int i = 0; i < size; i++) {
            output[i] = (input[i] - mean) / sqrtf(variance + epsilon);
        }
    }
    
    // Element-wise multiplication of two arrays
    void hadamard_product(float* a, float* b, float* output, int size) {
        for (int i = 0; i < size; i++) {
            output[i] = a[i] * b[i];
        }
    }
    
    // Convolution operation (1D)
    void conv1d(float* input, float* kernel, float* output, int input_size, int kernel_size) {
        int output_size = input_size - kernel_size + 1;
        
        for (int i = 0; i < output_size; i++) {
            output[i] = 0.0f;
            for (int j = 0; j < kernel_size; j++) {
                output[i] += input[i + j] * kernel[j];
            }
        }
    }
    
    // Feature transformation - combines multiple operations
    // This function demonstrates a more complex pipeline that:
    // 1. Applies convolution
    // 2. Normalizes the result
    // 3. Applies sigmoid activation
    float* feature_transform(float* input, float* kernel, 
                          int input_size, int kernel_size, float epsilon) {
        
        int output_size = input_size - kernel_size + 1;
        
        // Allocate temporary buffers
        float* conv_output = (float*)malloc(output_size * sizeof(float));
        float* norm_output = (float*)malloc(output_size * sizeof(float));
        float* output = (float*)malloc(output_size * sizeof(float));
        
        // Apply convolution
        conv1d(input, kernel, conv_output, input_size, kernel_size);
        
        // Apply batch normalization
        batch_norm(conv_output, norm_output, output_size, epsilon);
        
        // Apply sigmoid activation
        sigmoid_array(norm_output, output, output_size);
        
        // Free temporary buffers
        free(conv_output);
        free(norm_output);
        return output;
    }
}
"""

# Define the function you want to use (sigmoid_array as an example)
sigmoid_array_func = dyson.CppFunction(
    cpp_code=cpp_code, function_name="sigmoid_array", return_type="void"
)

# For a more complex example, use the feature_transform function

# Route hardware
hardware = router.route_hardware(cpp_code, mode="balanced")
print(f"Routed to: {hardware}")

# Compile the sigmoid_array function
compiled_sigmoid = dyson.run(
    sigmoid_array_func, target_device=hardware["hardware_type"]
)

# Create input and output arrays
input_size = 5
input_array = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
output_array = np.zeros(input_size, dtype=np.float32)

# Run the sigmoid function on the array
compiled_sigmoid(input_array, output_array, input_size)
print("Input array:", input_array)
print("Output after sigmoid:", output_array)
