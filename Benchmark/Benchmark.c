#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

/* ---------------- Utility --- ------------- */

static inline float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

static inline double now_ms() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

/* ---------------- Layers ---------------- */

void conv2d(
    float *in, float *out,
    float *w, float *b,
    int Cin, int Hin, int Win,
    int Cout, int K, int stride, int pad
) {
    int Hout = (Hin + 2*pad - K) / stride + 1;
    int Wout = (Win + 2*pad - K) / stride + 1;

    for (int co = 0; co < Cout; co++) {
        for (int h = 0; h < Hout; h++) {
            for (int wo = 0; wo < Wout; wo++) {
                float sum = b[co];

                for (int ci = 0; ci < Cin; ci++) {
                    for (int kh = 0; kh < K; kh++) {
                        for (int kw = 0; kw < K; kw++) {
                            int ih = h * stride + kh - pad;
                            int iw = wo * stride + kw - pad;

                            if (ih >= 0 && ih < Hin && iw >= 0 && iw < Win) {
                                int in_idx = ci*Hin*Win + ih*Win + iw;
                                int w_idx  = co*Cin*K*K + ci*K*K + kh*K + kw;
                                sum += in[in_idx] * w[w_idx];
                            }
                        }
                    }
                }
                out[co*Hout*Wout + h*Wout + wo] = relu(sum);
            }
        }
    }
}

void avgpool(float *in, float *out, int C, int H, int W) {
    int H2 = H / 2, W2 = W / 2;

    for (int c = 0; c < C; c++)
        for (int h = 0; h < H2; h++)
            for (int w = 0; w < W2; w++) {
                float s = 0.0f;
                for (int i = 0; i < 2; i++)
                    for (int j = 0; j < 2; j++)
                        s += in[c*H*W + (2*h+i)*W + (2*w+j)];
                out[c*H2*W2 + h*W2 + w] = 0.25f * s;
            }
}

void maxpool(float *in, float *out, int C, int H, int W) {
    int H2 = H / 2, W2 = W / 2;

    for (int c = 0; c < C; c++)
        for (int h = 0; h < H2; h++)
            for (int w = 0; w < W2; w++) {
                float m = -1e9f;
                for (int i = 0; i < 2; i++)
                    for (int j = 0; j < 2; j++) {
                        float v = in[c*H*W + (2*h+i)*W + (2*w+j)];
                        if (v > m) m = v;
                    }
                out[c*H2*W2 + h*W2 + w] = m;
            }
}

void fc(float *in, float *out, float *w, float *b, int N, int M) {
    for (int o = 0; o < M; o++) {
        float s = b[o];
        for (int i = 0; i < N; i++)
            s += in[i] * w[o*N + i];
        out[o] = relu(s);
    }
}

/* ---------------- Helper Functions ---------------- */

long long calculate_conv2d_ops(int Cin, int Hin, int Win, int Cout, int K, int stride, int pad) {
    int Hout = (Hin + 2*pad - K) / stride + 1;
    int Wout = (Win + 2*pad - K) / stride + 1;
    // MACs per output element: Cin * K * K (multiply-accumulate operations)
    // Total outputs: Cout * Hout * Wout
    // Each MAC = 2 ops (multiply + add)
    return (long long)Cout * Hout * Wout * Cin * K * K * 2;
}

long long calculate_fc_ops(int N, int M) {
    // MACs: M * N, each MAC = 2 ops
    return (long long)M * N * 2;
}

long long calculate_pool_ops(int C, int H, int W) {
    int H2 = H / 2, W2 = W / 2;
    // For avgpool: 4 adds + 1 multiply per output
    // For maxpool: 3 comparisons per output
    return (long long)C * H2 * W2 * 4; // Conservative estimate
}

void print_memory_usage() {
    printf("\n=== MEMORY USAGE ===\n");
    printf("Input:           %zu bytes (%.2f KB)\n", 
           sizeof(float) * 3*128*128, 
           sizeof(float) * 3*128*128 / 1024.0);
    
    printf("\nWeights:\n");
    printf("  conv1_w:       %zu bytes (%.2f KB)\n", 
           sizeof(float) * 16*3*3*3, 
           sizeof(float) * 16*3*3*3 / 1024.0);
    printf("  conv1_b:       %zu bytes\n", sizeof(float) * 16);
    printf("  conv2_w:       %zu bytes (%.2f KB)\n", 
           sizeof(float) * 32*16*3*3, 
           sizeof(float) * 32*16*3*3 / 1024.0);
    printf("  conv2_b:       %zu bytes\n", sizeof(float) * 32);
    printf("  conv3_w:       %zu bytes (%.2f KB)\n", 
           sizeof(float) * 64*32*3*3, 
           sizeof(float) * 64*32*3*3 / 1024.0);
    printf("  conv3_b:       %zu bytes\n", sizeof(float) * 64);
    printf("  fc1_w:         %zu bytes (%.2f KB)\n", 
           sizeof(float) * 256*1024, 
           sizeof(float) * 256*1024 / 1024.0);
    printf("  fc1_b:         %zu bytes\n", sizeof(float) * 256);
    printf("  fc2_w:         %zu bytes (%.2f KB)\n", 
           sizeof(float) * 20*256, 
           sizeof(float) * 20*256 / 1024.0);
    printf("  fc2_b:         %zu bytes\n", sizeof(float) * 20);
    
    size_t total_weights = sizeof(float) * (16*3*3*3 + 16 + 32*16*3*3 + 32 + 
                                             64*32*3*3 + 64 + 256*1024 + 256 + 20*256 + 20);
    printf("  Total weights: %zu bytes (%.2f KB)\n", total_weights, total_weights / 1024.0);
    
    printf("\nIntermediate Buffers:\n");
    printf("  buf1:          %zu bytes (%.2f KB)\n", 
           sizeof(float) * 16*128*128, 
           sizeof(float) * 16*128*128 / 1024.0);
    printf("  buf2:          %zu bytes (%.2f KB)\n", 
           sizeof(float) * 16*64*64, 
           sizeof(float) * 16*64*64 / 1024.0);
    printf("  buf3:          %zu bytes (%.2f KB)\n", 
           sizeof(float) * 32*32*32, 
           sizeof(float) * 32*32*32 / 1024.0);
    printf("  buf4:          %zu bytes (%.2f KB)\n", 
           sizeof(float) * 32*16*16, 
           sizeof(float) * 32*16*16 / 1024.0);
    printf("  buf5:          %zu bytes (%.2f KB)\n", 
           sizeof(float) * 64*8*8, 
           sizeof(float) * 64*8*8 / 1024.0);
    printf("  buf6:          %zu bytes (%.2f KB)\n", 
           sizeof(float) * 64*4*4, 
           sizeof(float) * 64*4*4 / 1024.0);
    printf("  fc1_out:       %zu bytes\n", sizeof(float) * 256);
    printf("  output:        %zu bytes\n", sizeof(float) * 20);
    
    size_t total_buffers = sizeof(float) * (16*128*128 + 16*64*64 + 32*32*32 + 
                                             32*16*16 + 64*8*8 + 64*4*4 + 256 + 20);
    printf("  Total buffers: %zu bytes (%.2f KB)\n", total_buffers, total_buffers / 1024.0);
    
    size_t total_memory = sizeof(float) * 3*128*128 + total_weights + total_buffers;
    printf("\nTOTAL MEMORY:    %zu bytes (%.2f MB)\n", 
           total_memory, 
           total_memory / (1024.0 * 1024.0));
}

/* ---------------- Main ---------------- */

int main() {
    srand(42);

    printf("=== NEURAL NETWORK BENCHMARK (ARM/Pynq-Z2) ===\n");
    printf("Board: Pynq-Z2 (Zynq-7000 ARM Cortex-A9)\n");
    printf("Precision: FP32 (single precision float)\n\n");

    /* Input */
    static float input[3*128*128];

    /* Weights - FIX: Initialize bias arrays to zero */
    static float conv1_w[16*3*3*3], conv1_b[16];
    static float conv2_w[32*16*3*3], conv2_b[32];
    static float conv3_w[64*32*3*3], conv3_b[64];
    static float fc1_w[256*1024], fc1_b[256];
    static float fc2_w[20*256], fc2_b[20];

    /* Buffers */
    static float buf1[16*128*128];
    static float buf2[16*64*64];
    static float buf3[32*32*32];
    static float buf4[32*16*16];
    static float buf5[64*8*8];
    static float buf6[64*4*4];
    static float fc1_out[256];
    static float output[20];

    /* Random init */
    for (int i = 0; i < 3*128*128; i++)
        input[i] = (float)rand() / RAND_MAX;

    for (int i = 0; i < 16*3*3*3; i++) conv1_w[i] = ((float)rand()/RAND_MAX)-0.5f;
    for (int i = 0; i < 16; i++) conv1_b[i] = 0.0f; // FIX: Initialize biases
    
    for (int i = 0; i < 32*16*3*3; i++) conv2_w[i] = ((float)rand()/RAND_MAX)-0.5f;
    for (int i = 0; i < 32; i++) conv2_b[i] = 0.0f;
    
    for (int i = 0; i < 64*32*3*3; i++) conv3_w[i] = ((float)rand()/RAND_MAX)-0.5f;
    for (int i = 0; i < 64; i++) conv3_b[i] = 0.0f;
    
    for (int i = 0; i < 256*1024; i++) fc1_w[i] = ((float)rand()/RAND_MAX)-0.5f;
    for (int i = 0; i < 256; i++) fc1_b[i] = 0.0f;
    
    for (int i = 0; i < 20*256; i++) fc2_w[i] = ((float)rand()/RAND_MAX)-0.5f;
    for (int i = 0; i < 20; i++) fc2_b[i] = 0.0f;

    printf("=== NETWORK ARCHITECTURE ===\n");
    printf("Layer 1: Conv2D (3x128x128 -> 16x128x128, K=3, S=1, P=1) + ReLU\n");
    printf("Layer 2: AvgPool (16x128x128 -> 16x64x64, 2x2)\n");
    printf("Layer 3: Conv2D (16x64x64 -> 32x32x32, K=3, S=2, P=1) + ReLU\n");
    printf("Layer 4: AvgPool (32x32x32 -> 32x16x16, 2x2)\n");
    printf("Layer 5: Conv2D (32x16x16 -> 64x8x8, K=3, S=2, P=1) + ReLU\n");
    printf("Layer 6: MaxPool (64x8x8 -> 64x4x4, 2x2)\n");
    printf("Layer 7: FC (1024 -> 256) + ReLU\n");
    printf("Layer 8: FC (256 -> 20) + ReLU\n");

    // Calculate theoretical operations
    printf("\n=== THEORETICAL OPERATIONS ===\n");
    long long conv1_ops = calculate_conv2d_ops(3, 128, 128, 16, 3, 1, 1);
    long long pool1_ops = calculate_pool_ops(16, 128, 128);
    long long conv2_ops = calculate_conv2d_ops(16, 64, 64, 32, 3, 2, 1);
    long long pool2_ops = calculate_pool_ops(32, 32, 32);
    long long conv3_ops = calculate_conv2d_ops(32, 16, 16, 64, 3, 2, 1);
    long long pool3_ops = calculate_pool_ops(64, 8, 8);
    long long fc1_ops = calculate_fc_ops(1024, 256);
    long long fc2_ops = calculate_fc_ops(256, 20);
    
    printf("Conv1:    %15lld FLOPs\n", conv1_ops);
    printf("AvgPool1: %15lld FLOPs\n", pool1_ops);
    printf("Conv2:    %15lld FLOPs\n", conv2_ops);
    printf("AvgPool2: %15lld FLOPs\n", pool2_ops);
    printf("Conv3:    %15lld FLOPs\n", conv3_ops);
    printf("MaxPool:  %15lld FLOPs\n", pool3_ops);
    printf("FC1:      %15lld FLOPs\n", fc1_ops);
    printf("FC2:      %15lld FLOPs\n", fc2_ops);
    
    long long total_ops = conv1_ops + pool1_ops + conv2_ops + pool2_ops + 
                          conv3_ops + pool3_ops + fc1_ops + fc2_ops;
    printf("TOTAL:    %15lld FLOPs (%.2f MFLOPs)\n", 
           total_ops, total_ops / 1e6);

    print_memory_usage();

    printf("\n=== RUNNING INFERENCE ===\n");
    
    double t0, t1;
    double layer_times[8];

    // Layer 1: Conv2D
    t0 = now_ms();
    conv2d(input, buf1, conv1_w, conv1_b, 3, 128, 128, 16, 3, 1, 1);
    t1 = now_ms();
    layer_times[0] = t1 - t0;

    // Layer 2: AvgPool
    t0 = now_ms();
    avgpool(buf1, buf2, 16, 128, 128);
    t1 = now_ms();
    layer_times[1] = t1 - t0;

    // Layer 3: Conv2D
    t0 = now_ms();
    conv2d(buf2, buf3, conv2_w, conv2_b, 16, 64, 64, 32, 3, 2, 1);
    t1 = now_ms();
    layer_times[2] = t1 - t0;

    // Layer 4: AvgPool
    t0 = now_ms();
    avgpool(buf3, buf4, 32, 32, 32);
    t1 = now_ms();
    layer_times[3] = t1 - t0;

    // Layer 5: Conv2D
    t0 = now_ms();
    conv2d(buf4, buf5, conv3_w, conv3_b, 32, 16, 16, 64, 3, 2, 1);
    t1 = now_ms();
    layer_times[4] = t1 - t0;

    // Layer 6: MaxPool
    t0 = now_ms();
    maxpool(buf5, buf6, 64, 8, 8);
    t1 = now_ms();
    layer_times[5] = t1 - t0;

    // Layer 7: FC
    t0 = now_ms();
    fc(buf6, fc1_out, fc1_w, fc1_b, 1024, 256);
    t1 = now_ms();
    layer_times[6] = t1 - t0;

    // Layer 8: FC
    t0 = now_ms();
    fc(fc1_out, output, fc2_w, fc2_b, 256, 20);
    t1 = now_ms();
    layer_times[7] = t1 - t0;

    double total_time = 0.0;
    for (int i = 0; i < 8; i++)
        total_time += layer_times[i];

    printf("\n=== LAYER-WISE PERFORMANCE ===\n");
    printf("Conv1:    %.3f ms  (%.1f%%)\n", layer_times[0], 100.0*layer_times[0]/total_time);
    printf("AvgPool1: %.3f ms  (%.1f%%)\n", layer_times[1], 100.0*layer_times[1]/total_time);
    printf("Conv2:    %.3f ms  (%.1f%%)\n", layer_times[2], 100.0*layer_times[2]/total_time);
    printf("AvgPool2: %.3f ms  (%.1f%%)\n", layer_times[3], 100.0*layer_times[3]/total_time);
    printf("Conv3:    %.3f ms  (%.1f%%)\n", layer_times[4], 100.0*layer_times[4]/total_time);
    printf("MaxPool:  %.3f ms  (%.1f%%)\n", layer_times[5], 100.0*layer_times[5]/total_time);
    printf("FC1:      %.3f ms  (%.1f%%)\n", layer_times[6], 100.0*layer_times[6]/total_time);
    printf("FC2:      %.3f ms  (%.1f%%)\n", layer_times[7], 100.0*layer_times[7]/total_time);

    printf("\n=== OVERALL PERFORMANCE ===\n");
    printf("Total Inference Time:  %.3f ms\n", total_time);
    printf("Throughput:            %.2f inferences/sec\n", 1000.0 / total_time);
    printf("Performance:           %.2f GFLOPS\n", (total_ops / 1e9) / (total_time / 1000.0));
    printf("Average Latency:       %.3f ms\n", total_time);

    printf("\n=== OUTPUT VERIFICATION ===\n");
    printf("Output (first 10): ");
    for (int i = 0; i < 10 && i < 20; i++)
        printf("%.4f ", output[i]);
    printf("\n");
    printf("Output (last 10):  ");
    for (int i = 10; i < 20; i++)
        printf("%.4f ", output[i]);
    printf("\n");

    // Find max output for classification
    float max_val = output[0];
    int max_idx = 0;
    for (int i = 1; i < 20; i++) {
        if (output[i] > max_val) {
            max_val = output[i];
            max_idx = i;
        }
    }
    printf("Predicted class: %d (confidence: %.4f)\n", max_idx, max_val);

    return 0;
}
