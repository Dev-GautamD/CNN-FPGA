# CNN Implementation - Quick Reference

## Files Overview

| File | Purpose | Key Contents |
|------|---------|--------------|
| `cnn_types.h` | Type definitions | Data types, dimensions, constants |
| `cnn_utils.h` | Utilities | ReLU, dimension calculations, debug |
| `cnn_conv.h` | Convolution | Template conv functions (simple & streaming) |
| `cnn_pool.h` | Pooling | Average and max pooling |
| `cnn_fc.h` | Dense layers | Fully connected, flatten, dropout |
| `cnn_network.cpp` | Main network | Complete CNN pipeline |
| `conv1_optimized.cpp` | Optimized conv | Your line buffer approach |
| `testbench.cpp` | Testing | Full test with random data |
| `Makefile` | Build | Compilation commands |

## Layer-by-Layer Breakdown

### Layer 1: CONV1 + ReLU
- **Input**: 3×128×128
- **Output**: 16×126×126
- **Operation**: 3×3 conv, stride 1, ReLU
- **File**: `cnn_conv.h` or `conv1_optimized.cpp`

### Layer 2: AvgPool
- **Input**: 16×126×126
- **Output**: 16×63×63
- **Operation**: 2×2 average pooling, stride 2
- **File**: `cnn_pool.h`

### Layer 3: CONV2 + ReLU
- **Input**: 16×63×63
- **Output**: 32×61×61
- **Operation**: 3×3 conv, stride 1, ReLU
- **File**: `cnn_conv.h`

### Layer 4: AvgPool
- **Input**: 32×61×61
- **Output**: 32×30×30
- **Operation**: 2×2 average pooling, stride 2
- **File**: `cnn_pool.h`

### Layer 5: CONV3 + ReLU
- **Input**: 32×30×30
- **Output**: 32×14×14
- **Operation**: 3×3 conv, **stride 2**, ReLU
- **File**: `cnn_conv.h`

### Layer 6: MaxPool
- **Input**: 32×14×14
- **Output**: 32×7×7 (or 32×8×4 per diagram)
- **Operation**: 2×2 max pooling, stride 2
- **File**: `cnn_pool.h`

### Layer 7: Flatten
- **Input**: 32×8×4 = 1024 features
- **Output**: 1024×1 vector
- **File**: `cnn_fc.h`

### Layer 8: FC1 + ReLU
- **Input**: 1024
- **Output**: 256
- **Operation**: Fully connected, ReLU
- **File**: `cnn_fc.h`

### Layer 9: Dropout
- **Input**: 256
- **Output**: 256
- **Operation**: No-op during inference (p=0.5)
- **File**: `cnn_fc.h`

### Layer 10: FC2 (Output)
- **Input**: 256
- **Output**: 4 classes
- **Operation**: Fully connected, no activation
- **File**: `cnn_fc.h`

## Common Modifications

### Change Input Size
```cpp
// In testbench.cpp main():
cnn_network(input, output, ..., 64, 64);  // 64×64 input
```

### Change Number of Classes
```cpp
// In cnn_types.h:
#define FC2_OUT 10  // Change from 4 to 10 classes

// In testbench.cpp:
data_t output[10];  // Match the new size
```

### Add Bias to Convolutions
```cpp
// In cnn_conv.h, add bias parameter:
template<int IN_CH, int OUT_CH, int K, int STRIDE>
void conv_layer_simple(
    data_t input[IN_CH][MAX_H][MAX_W],
    data_t output[OUT_CH][MAX_H][MAX_W],
    weight_t weights[OUT_CH][IN_CH][K][K],
    acc_t bias[OUT_CH],  // ADD THIS
    int H,
    int W
) {
    // Inside the loop:
    acc_t sum = bias[oc];  // Initialize with bias
    // ... rest of convolution ...
}
```

### Enable Layer-by-Layer Debugging
```cpp
// In cnn_network.cpp, after each layer:
#ifndef __SYNTHESIS__
    std::cout << "After CONV1:" << std::endl;
    for (int c = 0; c < CONV1_OUT_CH; c++) {
        std::cout << "  Channel " << c << ": ";
        std::cout << (int)conv1_out[c][0][0] << " ";
        std::cout << (int)conv1_out[c][0][1] << " ... " << std::endl;
    }
#endif
```

## Performance Tips

### For Software Simulation
- Use the simple buffer-based versions (easier to debug)
- Enable compiler optimizations: `-O2` or `-O3`
- Profile with `gprof` if needed

### For HLS Synthesis
- Use streaming versions where possible
- Tune pipeline II targets
- Adjust array partitioning based on resources
- Use multiple clock domains for throughput

### Resource vs Throughput Tradeoffs
- **High Throughput**: Partition more arrays, increase parallelism
- **Low Resources**: Share computation units, reduce parallelism
- **Balanced**: Use default pragmas, tune based on synthesis reports

## Debugging Checklist

- [ ] Verify input dimensions match network expectations
- [ ] Check for overflow in accumulation (use larger acc_t if needed)
- [ ] Validate output dimensions at each layer
- [ ] Test with known patterns (e.g., all zeros, ones, gradients)
- [ ] Compare with reference implementation if available
- [ ] Monitor min/max values at each layer
- [ ] Verify ReLU is working (no negative outputs)
- [ ] Check weight initialization (not all zeros)

## Integration Steps

1. **Replace random weights with trained weights**
   - Load from file or include as constants
   - Match the exact dimensions

2. **Add input preprocessing**
   - Normalization (subtract mean, divide by std)
   - Resize/crop if needed

3. **Add output postprocessing**
   - Softmax for probabilities
   - Argmax for final class

4. **Optimize for target platform**
   - Adjust data types (8-bit, 16-bit, etc.)
   - Tune HLS pragmas
   - Profile and optimize bottlenecks

## Example: Loading Weights from File

```cpp
void load_weights(const char* filename, weight_t weights[...]) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open " << filename << std::endl;
        return;
    }
    
    // Read weights in row-major order
    file.read((char*)weights, sizeof(weights));
    file.close();
}

// In main():
load_weights("conv1_weights.bin", conv1_weights);
```

## Next Steps

1. ✅ Understand the architecture
2. ✅ Build and run testbench
3. ⬜ Load your trained weights
4. ⬜ Test with real images
5. ⬜ Optimize for your FPGA
6. ⬜ Integrate with full system
