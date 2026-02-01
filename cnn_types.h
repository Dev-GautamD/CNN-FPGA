#ifndef CNN_TYPES_H
#define CNN_TYPES_H

#include <ap_int.h>
#include <hls_stream.h>

// Data type definitions
typedef ap_int<8>   data_t;    // 8-bit signed data
typedef ap_int<8>   weight_t;  // 8-bit signed weights
typedef ap_int<32>  acc_t;     // 32-bit accumulator

// Network architecture constants
#define MAX_H 128
#define MAX_W 128

// Layer 1: CONV1 + ReLU (3->16 channels, 3x3 kernel)
#define CONV1_IN_CH 3
#define CONV1_OUT_CH 16
#define CONV1_K 3

// Layer 2: AvgPool (2x2, stride 2)
#define POOL1_SIZE 2

// Layer 3: CONV2 + ReLU (16->32 channels, 3x3 kernel)
#define CONV2_IN_CH 16
#define CONV2_OUT_CH 32
#define CONV2_K 3

// Layer 4: AvgPool (2x2, stride 2)
#define POOL2_SIZE 2

// Layer 5: CONV3 + ReLU (32->32 channels, 3x3 kernel, stride 2)
#define CONV3_IN_CH 32
#define CONV3_OUT_CH 32
#define CONV3_K 3
#define CONV3_STRIDE 2

// Layer 6: MaxPool (2x2, stride 2)
#define POOL3_SIZE 2

// Layer 7: Flatten -> FC1 (1024->256)
#define FC1_IN 1024
#define FC1_OUT 256

// Layer 8: Dropout (keep for compatibility, p=0.5)
// No-op in inference

// Layer 9: FC2 (256->4)
#define FC2_IN 256
#define FC2_OUT 4

#endif // CNN_TYPES_H
