#ifndef EMBEDDED_WEIGHT_LOADER_H
#define EMBEDDED_WEIGHT_LOADER_H

#include <iostream>
#include <cstdint>
#include "cnn_types.h"

// Forward declaration - this will be in ship_weights.h (generated)
extern const int8_t SHIP_DETECTOR_WEIGHTS[];
extern const uint8_t SHIP_DETECTOR_INPUT[];

// Load weights from embedded constant array (stored in ROM/BRAM)
class EmbeddedWeightLoader {
private:
    const int8_t* weights_ptr;
    size_t current_offset;
    
public:
    // Constructor takes pointer to embedded weights array
    EmbeddedWeightLoader(const int8_t* weights) 
        : weights_ptr(weights), current_offset(0) {
        std::cout << "✓ Using embedded weights (stored in ROM/BRAM)" << std::endl;
    }
    
    // Load CONV layer weights by copying from ROM
    template<int OUT_CH, int IN_CH, int K>
    void load_conv_weights(weight_t weights[OUT_CH][IN_CH][K][K]) {
        size_t num_weights = OUT_CH * IN_CH * K * K;
        
        std::cout << "  Loading CONV: " << OUT_CH << "×" << IN_CH << "×" << K << "×" << K 
                  << " = " << num_weights << " weights (offset " << current_offset << ")" << std::endl;
        
        for (int oc = 0; oc < OUT_CH; oc++) {
            for (int ic = 0; ic < IN_CH; ic++) {
                for (int kh = 0; kh < K; kh++) {
                    for (int kw = 0; kw < K; kw++) {
                        weights[oc][ic][kh][kw] = weights_ptr[current_offset++];
                    }
                }
            }
        }
    }
    
    // Alternative: Return pointer directly (no copying!)
    // This is better for FPGA - weights stay in ROM
    template<int OUT_CH, int IN_CH, int K>
    const int8_t* get_conv_weights_ptr() {
        size_t num_weights = OUT_CH * IN_CH * K * K;
        const int8_t* ptr = &weights_ptr[current_offset];
        current_offset += num_weights;
        
        std::cout << "  Mapped CONV: " << OUT_CH << "×" << IN_CH << "×" << K << "×" << K 
                  << " (offset " << (current_offset - num_weights) << ")" << std::endl;
        
        return ptr;
    }
    
    // Load FC layer weights
    template<int OUT_FEATURES, int IN_FEATURES>
    void load_fc_weights(weight_t weights[OUT_FEATURES][IN_FEATURES]) {
        size_t num_weights = OUT_FEATURES * IN_FEATURES;
        
        std::cout << "  Loading FC: " << OUT_FEATURES << "×" << IN_FEATURES 
                  << " = " << num_weights << " weights (offset " << current_offset << ")" << std::endl;
        
        for (int out = 0; out < OUT_FEATURES; out++) {
            for (int in = 0; in < IN_FEATURES; in++) {
                weights[out][in] = weights_ptr[current_offset++];
            }
        }
    }
    
    // Load biases
    template<int SIZE>
    void load_bias(acc_t bias[SIZE]) {
        std::cout << "  Loading BIAS: " << SIZE << " values (offset " 
                  << current_offset << ")" << std::endl;
        
        for (int i = 0; i < SIZE; i++) {
            bias[i] = weights_ptr[current_offset++];
        }
    }
    
    void skip(size_t bytes) {
        current_offset += bytes;
    }
    
    size_t get_offset() const { return current_offset; }
};

// Load embedded input image
bool load_embedded_input(
    const uint8_t* input_data,
    data_t input[CONV1_IN_CH][MAX_H][MAX_W],
    int H,
    int W
) {
    std::cout << "✓ Using embedded input image (stored in ROM)" << std::endl;
    
    // Convert from HWC to CHW format
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            for (int c = 0; c < 3; c++) {
                int idx = (h * W * 3) + (w * 3) + c;
                
                // Convert uint8 (0-255) to int8 (-128 to 127)
                input[c][h][w] = (data_t)(input_data[idx] - 128);
            }
        }
    }
    
    return true;
}

#endif // EMBEDDED_WEIGHT_LOADER_H
