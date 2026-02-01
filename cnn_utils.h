#ifndef CNN_UTILS_H
#define CNN_UTILS_H

#include "cnn_types.h"
#include <iostream>

// ReLU activation
inline data_t relu(acc_t x) {
    if (x > 127) return 127;   // Clamp to max 8-bit value
    if (x < 0) return 0;       // ReLU threshold
    return (data_t)x;
}

// Debug print for feature map statistics
inline void print_feature_map_stats(const char* layer_name, data_t* data, int size) {
#ifndef __SYNTHESIS__
    acc_t sum = 0;
    data_t min_val = 127;
    data_t max_val = -128;
    
    for (int i = 0; i < size; i++) {
        sum += data[i];
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }
    
    std::cout << "[" << layer_name << "] ";
    std::cout << "Size=" << size;
    std::cout << ", Min=" << (int)min_val;
    std::cout << ", Max=" << (int)max_val;
    std::cout << ", Avg=" << (sum / size) << std::endl;
#endif
}

// Calculate output size after convolution
inline int conv_out_size(int in_size, int kernel, int stride = 1, int padding = 0) {
    return ((in_size + 2 * padding - kernel) / stride) + 1;
}

// Calculate output size after pooling
inline int pool_out_size(int in_size, int pool_size, int stride) {
    return (in_size / stride);
}

#endif // CNN_UTILS_H
