# CNN Network - Complete Implementation

This is a complete, debuggable implementation of the CNN architecture shown in your diagram.

## Architecture Overview

```
Input (3×128×128)
  ↓
CONV1 + ReLU (3→16 channels, 3×3 kernel)
  ↓
AvgPool (2×2, stride 2) → 16×63×63
  ↓
CONV2 + ReLU (16→32 channels, 3×3 kernel)
  ↓
AvgPool (2×2, stride 2) → 32×30×30
  ↓
CONV3 + ReLU (32→32 channels, 3×3 kernel, stride 2)
  ↓
MaxPool (2×2, stride 2) → 32×8×4
  ↓
Flatten → 1024
  ↓
FC1 + ReLU (1024→256)
  ↓
Dropout (p=0.5, no-op in inference)
  ↓
FC2 (256→4)
  ↓
Output (4 classes)
```

## File Structure

- **cnn_types.h** - Type definitions and constants
- **cnn_utils.h** - Utility functions (ReLU, debugging)
- **cnn_conv.h** - Convolution layer implementations
- **cnn_pool.h** - Pooling layer implementations (avg/max)
- **cnn_fc.h** - Fully connected and flatten layers
- **cnn_network.cpp** - Main network implementation
- **testbench.cpp** - Testbench for debugging
- **Makefile** - Build automation

## Features

✅ **Modular Design** - Each layer type in separate files
✅ **Easy Debugging** - Buffer-based operations with clear logic
✅ **Two Conv Implementations** - Streaming (HLS-optimized) and simple (debugging)
✅ **Comprehensive Testing** - Full testbench with reproducible results
✅ **HLS-Ready** - Pragma annotations for synthesis
✅ **Type Safety** - Fixed-point arithmetic (8-bit data, 32-bit accumulation)

## Quick Start

### 1. Software Simulation

```bash
# Build the testbench
make

# Run the test
make run
```

This will:
- Initialize random weights (reproducible with seed=42)
- Create test input with gradient pattern
- Run the complete network
- Display output logits and predicted class
- Save results to `cnn_output.txt`

### 2. HLS Synthesis (requires Vivado HLS)

```bash
# Create HLS project and synthesize
make hls
```

## Understanding the Code

### Layer Implementations

**Convolution (cnn_conv.h)**
- Two versions: streaming (line buffer) and simple (buffer-based)
- Supports stride 1 and stride 2
- Built-in ReLU activation
- Template-based for flexibility

**Pooling (cnn_pool.h)**
- Average pooling for layers 2 & 4
- Max pooling for layer 6
- 2×2 window with stride 2

**Fully Connected (cnn_fc.h)**
- Matrix-vector multiplication
- Optional ReLU activation
- Bias support
- Dropout (no-op in inference)

### Data Flow

The network uses intermediate buffers between layers:
```cpp
input → conv1_out → pool1_out → conv2_out → pool2_out 
     → conv3_out → pool3_out → flattened → fc1_out 
     → dropout_out → output
```

### Debugging Tips

1. **Check intermediate outputs** - Add print statements after each layer
2. **Verify dimensions** - The comments show expected sizes at each stage
3. **Monitor overflow** - 32-bit accumulators prevent overflow in convolutions
4. **Test incrementally** - Test each layer separately before full network

### Modifying the Network

**Change number of filters:**
```cpp
// In cnn_types.h
#define CONV1_OUT_CH 32  // Change from 16 to 32
```

**Change kernel size:**
```cpp
// In cnn_types.h
#define CONV1_K 5  // Change from 3 to 5
```

**Add/remove layers:**
Edit `cnn_network.cpp` and add layer calls with appropriate dimensions.

## HLS Optimization

The code includes several HLS optimizations:

1. **Array Partitioning** - Enables parallel access
2. **Pipeline Pragmas** - Maximizes throughput
3. **Streaming Interfaces** - For AXI4-Stream (in conv_layer_stream)
4. **BRAM Interfaces** - For weight storage

### Resource Usage Estimates

- **LUTs**: ~50K-100K (depends on optimizations)
- **DSPs**: ~500-1000 (for MAC operations)
- **BRAM**: ~100-200 blocks (for weights and buffers)

## Integration with Your Code

Your improved convolution code with line buffers is available as `conv_layer_stream()` in `cnn_conv.h`. 

To use it instead of the simple version:
```cpp
// In cnn_network.cpp, replace:
conv_layer_simple<...>(...);

// With:
conv_layer_stream<...>(...);
```

Note: The streaming version requires converting to/from streams.

## Testing and Validation

The testbench creates:
1. Reproducible random weights (seed=42)
2. Gradient pattern input (easy to visualize)
3. Output file with complete results

Expected output format:
```
Output logits:
  Class 0: <value>
  Class 1: <value>
  Class 2: <value>
  Class 3: <value>

Predicted class: <0-3>
```

## Common Issues

**Overflow in accumulation:**
- Solution: Use 32-bit `acc_t` for intermediate sums
- The code already handles this

**Dimension mismatch:**
- Check `conv_out_size()` and `pool_out_size()` calculations
- Verify H/W parameters match your input

**Synthesis errors:**
- Ensure all arrays are properly partitioned
- Check that templates are fully instantiated

## Next Steps

1. **Load real weights** - Replace random initialization with trained weights
2. **Add more layers** - Extend the architecture as needed
3. **Optimize for FPGA** - Tune array partitioning and pipeline depths
4. **Add data loaders** - Read images from files or cameras
5. **Performance profiling** - Measure latency and throughput

## License

This code is provided for educational purposes.
