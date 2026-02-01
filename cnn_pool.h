#ifndef CNN_POOL_H
#define CNN_POOL_H

#include "cnn_types.h"

// Average Pooling (2x2, stride 2)
template<int CHANNELS, int POOL_SIZE>
void avg_pool(
    data_t input[CHANNELS][MAX_H][MAX_W],
    data_t output[CHANNELS][MAX_H][MAX_W],
    int H,
    int W
) {
    int out_h = H / POOL_SIZE;
    int out_w = W / POOL_SIZE;
    
    for (int c = 0; c < CHANNELS; c++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
#pragma HLS PIPELINE II=1
                
                acc_t sum = 0;
                
                for (int ph = 0; ph < POOL_SIZE; ph++) {
                    for (int pw = 0; pw < POOL_SIZE; pw++) {
                        int ih = oh * POOL_SIZE + ph;
                        int iw = ow * POOL_SIZE + pw;
                        sum += input[c][ih][iw];
                    }
                }
                
                // Average (divide by pool_size^2)
                output[c][oh][ow] = (data_t)(sum / (POOL_SIZE * POOL_SIZE));
            }
        }
    }
}

// Max Pooling (2x2, stride 2)
template<int CHANNELS, int POOL_SIZE>
void max_pool(
    data_t input[CHANNELS][MAX_H][MAX_W],
    data_t output[CHANNELS][MAX_H][MAX_W],
    int H,
    int W
) {
    int out_h = H / POOL_SIZE;
    int out_w = W / POOL_SIZE;
    
    for (int c = 0; c < CHANNELS; c++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
#pragma HLS PIPELINE II=1
                
                data_t max_val = -128;  // Min value for signed 8-bit
                
                for (int ph = 0; ph < POOL_SIZE; ph++) {
                    for (int pw = 0; pw < POOL_SIZE; pw++) {
                        int ih = oh * POOL_SIZE + ph;
                        int iw = ow * POOL_SIZE + pw;
                        data_t val = input[c][ih][iw];
                        if (val > max_val) {
                            max_val = val;
                        }
                    }
                }
                
                output[c][oh][ow] = max_val;
            }
        }
    }
}

#endif // CNN_POOL_H
