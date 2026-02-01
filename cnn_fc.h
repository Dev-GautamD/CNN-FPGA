#ifndef CNN_FC_H
#define CNN_FC_H

#include "cnn_types.h"
#include "cnn_utils.h"

// Flatten operation
template<int CHANNELS, int H, int W>
void flatten(
    data_t input[CHANNELS][MAX_H][MAX_W],
    data_t output[CHANNELS * H * W]
) {
    int idx = 0;
    for (int c = 0; c < CHANNELS; c++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
#pragma HLS PIPELINE II=1
                output[idx++] = input[c][h][w];
            }
        }
    }
}

// Fully Connected Layer
template<int IN_FEATURES, int OUT_FEATURES>
void fc_layer(
    data_t input[IN_FEATURES],
    data_t output[OUT_FEATURES],
    weight_t weights[OUT_FEATURES][IN_FEATURES],
    acc_t bias[OUT_FEATURES],
    bool apply_relu = true
) {
    for (int out = 0; out < OUT_FEATURES; out++) {
#pragma HLS PIPELINE II=1
        
        acc_t sum = bias[out];
        
        for (int in = 0; in < IN_FEATURES; in++) {
            sum += input[in] * weights[out][in];
        }
        
        if (apply_relu) {
            output[out] = relu(sum);
        } else {
            // For final layer, just clamp without ReLU
            if (sum > 127) sum = 127;
            if (sum < -128) sum = -128;
            output[out] = (data_t)sum;
        }
    }
}

// Dropout layer (no-op during inference)
template<int FEATURES>
void dropout(
    data_t input[FEATURES],
    data_t output[FEATURES]
) {
    for (int i = 0; i < FEATURES; i++) {
#pragma HLS PIPELINE II=1
        output[i] = input[i];
    }
}

#endif // CNN_FC_H
