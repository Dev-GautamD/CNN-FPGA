#ifndef CNN_CONV_H
#define CNN_CONV_H

#include "cnn_types.h"
#include "cnn_utils.h"

// Conv layer with line buffer (for streaming)
// Supports: 3x3 kernel, stride 1 or 2
template<int IN_CH, int OUT_CH, int K, int STRIDE>
void conv_layer_stream(
    hls::stream<data_t> &in,
    hls::stream<data_t> &out,
    weight_t weights[OUT_CH][IN_CH][K][K],
    int H,
    int W
) {
    // Line buffers for each input channel
    data_t linebuf[IN_CH][K-1][MAX_W];
#pragma HLS ARRAY_PARTITION variable=linebuf complete dim=1
#pragma HLS ARRAY_PARTITION variable=linebuf complete dim=2
    
    // Sliding window
    data_t window[IN_CH][K][K];
#pragma HLS ARRAY_PARTITION variable=window complete dim=0
    
    int out_h = conv_out_size(H, K, STRIDE);
    int out_w = conv_out_size(W, K, STRIDE);
    
    for (int ch = 0; ch < IN_CH; ch++) {
        for (int row = 0; row < H; row++) {
            for (int col = 0; col < W; col++) {
#pragma HLS PIPELINE II=1
                
                data_t pixel = in.read();
                
                // Shift line buffers
                for (int i = 0; i < K-2; i++) {
                    linebuf[ch][i][col] = linebuf[ch][i+1][col];
                }
                if (K > 1) {
                    linebuf[ch][K-2][col] = pixel;
                }
                
                // Check if we can compute convolution
                bool valid_row = (STRIDE == 1) ? (row >= K-1) : ((row >= K-1) && ((row - K + 1) % STRIDE == 0));
                bool valid_col = (STRIDE == 1) ? (col >= K-1) : ((col >= K-1) && ((col - K + 1) % STRIDE == 0));
                
                if (valid_row && valid_col) {
                    // Fill window
                    for (int i = 0; i < K-1; i++) {
                        for (int j = 0; j < K; j++) {
                            int offset = col - K + 1 + j;
                            window[ch][i][j] = linebuf[ch][i][offset];
                        }
                    }
                    for (int j = 0; j < K; j++) {
                        window[ch][K-1][j] = (j == K-1) ? pixel : linebuf[ch][K-2][col - K + 1 + j];
                    }
                    
                    // Process only when all channels are ready
                    if (ch == IN_CH - 1) {
                        // Compute all output channels
                        for (int oc = 0; oc < OUT_CH; oc++) {
#pragma HLS PIPELINE
                            acc_t sum = 0;
                            
                            for (int ic = 0; ic < IN_CH; ic++) {
                                for (int i = 0; i < K; i++) {
                                    for (int j = 0; j < K; j++) {
                                        sum += window[ic][i][j] * weights[oc][ic][i][j];
                                    }
                                }
                            }
                            
                            // ReLU activation
                            data_t result = relu(sum);
                            out.write(result);
                        }
                    }
                }
            }
        }
    }
}

// Simplified Conv layer (buffer-based, easier to debug)
template<int IN_CH, int OUT_CH, int K, int STRIDE>
void conv_layer_simple(
    data_t input[IN_CH][MAX_H][MAX_W],
    data_t output[OUT_CH][MAX_H][MAX_W],
    weight_t weights[OUT_CH][IN_CH][K][K],
    int H,
    int W
) {
    int out_h = conv_out_size(H, K, STRIDE);
    int out_w = conv_out_size(W, K, STRIDE);
    
    for (int oc = 0; oc < OUT_CH; oc++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
#pragma HLS PIPELINE II=1
                
                acc_t sum = 0;
                
                for (int ic = 0; ic < IN_CH; ic++) {
                    for (int kh = 0; kh < K; kh++) {
                        for (int kw = 0; kw < K; kw++) {
                            int ih = oh * STRIDE + kh;
                            int iw = ow * STRIDE + kw;
                            sum += input[ic][ih][iw] * weights[oc][ic][kh][kw];
                        }
                    }
                }
                
                output[oc][oh][ow] = relu(sum);
            }
        }
    }
}

#endif // CNN_CONV_H
