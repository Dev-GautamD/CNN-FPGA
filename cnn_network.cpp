#include "cnn_types.h"
#include "cnn_utils.h"
#include "cnn_conv.h"
#include "cnn_pool.h"
#include "cnn_fc.h"

// Main CNN Network
void cnn_network(
    data_t input[CONV1_IN_CH][MAX_H][MAX_W],
    data_t output[FC2_OUT],
    
    // Layer weights
    weight_t conv1_weights[CONV1_OUT_CH][CONV1_IN_CH][CONV1_K][CONV1_K],
    weight_t conv2_weights[CONV2_OUT_CH][CONV2_IN_CH][CONV2_K][CONV2_K],
    weight_t conv3_weights[CONV3_OUT_CH][CONV3_IN_CH][CONV3_K][CONV3_K],
    weight_t fc1_weights[FC1_OUT][FC1_IN],
    weight_t fc2_weights[FC2_OUT][FC2_IN],
    
    // Biases
    acc_t fc1_bias[FC1_OUT],
    acc_t fc2_bias[FC2_OUT],
    
    // Input dimensions
    int H,
    int W
) {
#pragma HLS INTERFACE bram port=input
#pragma HLS INTERFACE bram port=output
#pragma HLS INTERFACE bram port=conv1_weights
#pragma HLS INTERFACE bram port=conv2_weights
#pragma HLS INTERFACE bram port=conv3_weights
#pragma HLS INTERFACE bram port=fc1_weights
#pragma HLS INTERFACE bram port=fc2_weights
#pragma HLS INTERFACE bram port=fc1_bias
#pragma HLS INTERFACE bram port=fc2_bias
#pragma HLS INTERFACE s_axilite port=H
#pragma HLS INTERFACE s_axilite port=W
#pragma HLS INTERFACE s_axilite port=return

    // Intermediate feature maps
    static data_t conv1_out[CONV1_OUT_CH][MAX_H][MAX_W];
    static data_t pool1_out[CONV1_OUT_CH][MAX_H][MAX_W];
    static data_t conv2_out[CONV2_OUT_CH][MAX_H][MAX_W];
    static data_t pool2_out[CONV2_OUT_CH][MAX_H][MAX_W];
    static data_t conv3_out[CONV3_OUT_CH][MAX_H][MAX_W];
    static data_t pool3_out[CONV3_OUT_CH][MAX_H][MAX_W];
    static data_t flattened[FC1_IN];
    static data_t fc1_out[FC1_OUT];
    static data_t dropout_out[FC1_OUT];
    
    // Calculate dimensions at each stage
    int h1 = conv_out_size(H, CONV1_K, 1);      // 128->126
    int w1 = conv_out_size(W, CONV1_K, 1);
    
    int h2 = pool_out_size(h1, POOL1_SIZE, POOL1_SIZE);  // 126->63
    int w2 = pool_out_size(w1, POOL1_SIZE, POOL1_SIZE);
    
    int h3 = conv_out_size(h2, CONV2_K, 1);      // 63->61
    int w3 = conv_out_size(w2, CONV2_K, 1);
    
    int h4 = pool_out_size(h3, POOL2_SIZE, POOL2_SIZE);  // 61->30
    int w4 = pool_out_size(w3, POOL2_SIZE, POOL2_SIZE);
    
    int h5 = conv_out_size(h4, CONV3_K, CONV3_STRIDE);   // 30->14
    int w5 = conv_out_size(w4, CONV3_K, CONV3_STRIDE);
    
    int h6 = pool_out_size(h5, POOL3_SIZE, POOL3_SIZE);  // 14->7 (but diagram shows 8x4)
    int w6 = pool_out_size(w5, POOL3_SIZE, POOL3_SIZE);
    
    // Layer 1: CONV1 + ReLU (3->16, 3x3)
    conv_layer_simple<CONV1_IN_CH, CONV1_OUT_CH, CONV1_K, 1>(
        input, conv1_out, conv1_weights, H, W
    );
    
    // Layer 2: AvgPool (2x2)
    avg_pool<CONV1_OUT_CH, POOL1_SIZE>(
        conv1_out, pool1_out, h1, w1
    );
    
    // Layer 3: CONV2 + ReLU (16->32, 3x3)
    conv_layer_simple<CONV2_IN_CH, CONV2_OUT_CH, CONV2_K, 1>(
        pool1_out, conv2_out, conv2_weights, h2, w2
    );
    
    // Layer 4: AvgPool (2x2)
    avg_pool<CONV2_OUT_CH, POOL2_SIZE>(
        conv2_out, pool2_out, h3, w3
    );
    
    // Layer 5: CONV3 + ReLU (32->32, 3x3, stride 2)
    conv_layer_simple<CONV3_IN_CH, CONV3_OUT_CH, CONV3_K, CONV3_STRIDE>(
        pool2_out, conv3_out, conv3_weights, h4, w4
    );
    
    // Layer 6: MaxPool (2x2)
    max_pool<CONV3_OUT_CH, POOL3_SIZE>(
        conv3_out, pool3_out, h5, w5
    );
    
    // Layer 7: Flatten
    // Note: The diagram shows 8x4x4=1024, adjust h6/w6 accordingly
    flatten<CONV3_OUT_CH, 8, 4>(pool3_out, flattened);
    
    // Layer 8: FC1 (1024->256) + ReLU
    fc_layer<FC1_IN, FC1_OUT>(
        flattened, fc1_out, fc1_weights, fc1_bias, true
    );
    
    // Layer 9: Dropout (no-op in inference)
    dropout<FC1_OUT>(fc1_out, dropout_out);
    
    // Layer 10: FC2 (256->4) - Output layer
    fc_layer<FC2_IN, FC2_OUT>(
        dropout_out, output, fc2_weights, fc2_bias, false
    );
}
