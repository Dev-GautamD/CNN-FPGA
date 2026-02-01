#include <iostream>
#include "cnn_types.h"
#include "cnn_utils.h"
#include "embedded_weight_loader.h"
#include "ship_weights.h"  // Generated header with embedded weights

// External CNN function
extern void cnn_network(
    data_t input[CONV1_IN_CH][MAX_H][MAX_W],
    data_t output[FC2_OUT],
    weight_t conv1_weights[CONV1_OUT_CH][CONV1_IN_CH][CONV1_K][CONV1_K],
    weight_t conv2_weights[CONV2_OUT_CH][CONV2_IN_CH][CONV2_K][CONV2_K],
    weight_t conv3_weights[CONV3_OUT_CH][CONV3_IN_CH][CONV3_K][CONV3_K],
    weight_t fc1_weights[FC1_OUT][FC1_IN],
    weight_t fc2_weights[FC2_OUT][FC2_IN],
    acc_t fc1_bias[FC1_OUT],
    acc_t fc2_bias[FC2_OUT],
    int H,
    int W
);

int main() {
    std::cout << "╔════════════════════════════════════════════╗" << std::endl;
    std::cout << "║   Ship Detector - Embedded Weights        ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════╝" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Using embedded weights (no file I/O required!)" << std::endl;
    std::cout << "Weights are stored in ROM/BRAM at compile time." << std::endl;
    std::cout << std::endl;
    
    // Allocate working memory (not for weights!)
    static data_t input[CONV1_IN_CH][MAX_H][MAX_W];
    static data_t output[FC2_OUT];
    
    static weight_t conv1_weights[CONV1_OUT_CH][CONV1_IN_CH][CONV1_K][CONV1_K];
    static weight_t conv2_weights[CONV2_OUT_CH][CONV2_IN_CH][CONV2_K][CONV2_K];
    static weight_t conv3_weights[CONV3_OUT_CH][CONV3_IN_CH][CONV3_K][CONV3_K];
    static weight_t fc1_weights[FC1_OUT][FC1_IN];
    static weight_t fc2_weights[FC2_OUT][FC2_IN];
    static acc_t fc1_bias[FC1_OUT];
    static acc_t fc2_bias[FC2_OUT];
    
    // ========================================
    // STEP 1: Load Weights from Embedded Array
    // ========================================
    std::cout << "[Step 1/3] Loading weights from embedded array..." << std::endl;
    std::cout << "─────────────────────────────────────────────" << std::endl;
    
    // Use the embedded constant array (defined in ship_weights.h)
    EmbeddedWeightLoader loader(SHIP_DETECTOR_WEIGHTS);
    
    std::cout << "\nCopying weights from ROM to working memory:" << std::endl;
    loader.load_conv_weights<CONV1_OUT_CH, CONV1_IN_CH, CONV1_K>(conv1_weights);
    loader.load_conv_weights<CONV2_OUT_CH, CONV2_IN_CH, CONV2_K>(conv2_weights);
    loader.load_conv_weights<CONV3_OUT_CH, CONV3_IN_CH, CONV3_K>(conv3_weights);
    loader.load_fc_weights<FC1_OUT, FC1_IN>(fc1_weights);
    loader.load_fc_weights<FC2_OUT, FC2_IN>(fc2_weights);
    
    // Initialize biases
    std::cout << "\nInitializing biases to zero..." << std::endl;
    for (int i = 0; i < FC1_OUT; i++) fc1_bias[i] = 0;
    for (int i = 0; i < FC2_OUT; i++) fc2_bias[i] = 0;
    
    std::cout << "\n✓ Weights loaded from ROM!" << std::endl;
    std::cout << "  Total weights used: " << loader.get_offset() << std::endl;
    
    // ========================================
    // STEP 2: Load Input from Embedded Array
    // ========================================
    std::cout << "\n[Step 2/3] Loading input from embedded array..." << std::endl;
    std::cout << "─────────────────────────────────────────────" << std::endl;
    
    load_embedded_input(SHIP_DETECTOR_INPUT, input, 128, 128);
    
    // ========================================
    // STEP 3: Run CNN
    // ========================================
    std::cout << "\n[Step 3/3] Running CNN inference..." << std::endl;
    std::cout << "─────────────────────────────────────────────" << std::endl;
    std::cout << "Processing 128×128×3 image..." << std::endl;
    
    cnn_network(
        input, output,
        conv1_weights, conv2_weights, conv3_weights,
        fc1_weights, fc2_weights,
        fc1_bias, fc2_bias,
        128, 128
    );
    
    std::cout << "✓ Inference complete!" << std::endl;
    
    // ========================================
    // Display Results
    // ========================================
    std::cout << "\n╔════════════════════════════════════════════╗" << std::endl;
    std::cout << "║   Results                                  ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════╝" << std::endl;
    
    std::cout << "\nCNN Output:" << std::endl;
    for (int i = 0; i < FC2_OUT; i++) {
        std::cout << "  output[" << i << "] = " << (int)output[i] << std::endl;
    }
    
    // Find max
    data_t max_val = output[0];
    int max_idx = 0;
    for (int i = 1; i < FC2_OUT; i++) {
        if (output[i] > max_val) {
            max_val = output[i];
            max_idx = i;
        }
    }
    std::cout << "\nPredicted class: " << max_idx 
              << " (value=" << (int)max_val << ")" << std::endl;
    
    std::cout << "\n✓ Test complete! No files needed - everything embedded!" << std::endl;
    
    return 0;
}
