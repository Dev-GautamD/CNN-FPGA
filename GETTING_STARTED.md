# Getting Started with CNN Implementation

## What You Have

A complete, debuggable CNN implementation matching your architecture diagram with:
- ✅ All 10 layers implemented
- ✅ Modular, easy-to-understand code structure
- ✅ Two testbenches (full network + simple conv test)
- ✅ HLS-ready with synthesis pragmas
- ✅ Your line buffer optimization integrated

## Quick Start (3 Steps)

### Step 1: Build Everything
```bash
make
```

This creates two executables:
- `cnn_test` - Full network test
- `simple_conv_test` - Simple convolution test

### Step 2: Run Simple Test First
```bash
./simple_conv_test
```

This tests just the convolution layer with a tiny 8×8 image. Perfect for understanding the basics!

Expected output:
```
=== Simple Convolution Layer Test ===
[1] Creating test input (3×8×8)...
[2] Initializing weights...
[3] Running 3×3 convolution (stride 1)...
Output size: 6×6
✓ PASS: Output matches expected value!
```

### Step 3: Run Full Network
```bash
./cnn_test
```

This runs the complete CNN with all 10 layers on a 128×128 input.

Expected output:
```
=== CNN Network Testbench ===
[1/3] Initializing weights...
[2/3] Initializing test input...
[3/3] Running CNN network...

=== Results ===
Output logits:
  Class 0: <value>
  Class 1: <value>
  Class 2: <value>
  Class 3: <value>

Predicted class: <0-3>
```

## File Guide

### Start Here
1. **README.md** - Full documentation
2. **QUICK_REFERENCE.md** - Layer-by-layer breakdown
3. **THIS FILE** - You are here!

### Core Implementation
- **cnn_network.cpp** - Main network (start reading here)
- **cnn_types.h** - Constants and types
- **cnn_conv.h** - Convolution layers
- **cnn_pool.h** - Pooling layers
- **cnn_fc.h** - Dense layers

### Testing & Examples
- **testbench.cpp** - Full network test
- **simple_conv_test.cpp** - Simple example
- **conv1_optimized.cpp** - Your line buffer version

### Build & Synthesis
- **Makefile** - Build commands
- Run `make hls` for HLS synthesis (requires Vivado HLS)

## Understanding the Code Flow

### 1. Data Types (cnn_types.h)
```cpp
data_t    // 8-bit input/output data
weight_t  // 8-bit weights
acc_t     // 32-bit accumulator (prevents overflow)
```

### 2. Network Structure (cnn_network.cpp)
```
Read input → CONV1 → POOL1 → CONV2 → POOL2 → CONV3 
→ POOL3 → FLATTEN → FC1 → DROPOUT → FC2 → Output
```

### 3. Each Layer Uses Templates
```cpp
// Example: Convolution with 3 input channels, 16 output, 3×3 kernel
conv_layer_simple<3, 16, 3, 1>(input, output, weights, H, W);
                  ↑   ↑  ↑  ↑
                  |   |  |  stride
                  |   |  kernel size
                  |   output channels
                  input channels
```

## Debugging Tips

### Problem: Compilation Errors
**Solution**: Make sure you have g++ with C++11 support
```bash
g++ --version  # Check version
```

### Problem: Need to see intermediate values
**Solution**: Add prints in cnn_network.cpp
```cpp
#ifndef __SYNTHESIS__
std::cout << "After CONV1, output[0][0][0] = " 
          << (int)conv1_out[0][0][0] << std::endl;
#endif
```

### Problem: Results don't make sense
**Solution**: 
1. Start with simple_conv_test (8×8 input)
2. Use known patterns (all zeros, ones, etc.)
3. Verify dimensions at each layer
4. Check for overflow (use larger acc_t if needed)

## Customization Examples

### Change Input Size
In testbench.cpp:
```cpp
// Change from 128×128 to 64×64
init_test_input(input, 64, 64);
cnn_network(..., 64, 64);
```

### Change to 10 Classes
In cnn_types.h:
```cpp
#define FC2_OUT 10  // Was 4
```

In testbench.cpp:
```cpp
data_t output[10];  // Match new size
```

### Add Your Own Weights
In testbench.cpp, replace init_random_weights() with:
```cpp
void load_my_weights() {
    // Load from file or hardcode
    for (int i = 0; i < ...; i++) {
        weights[...] = my_trained_values[i];
    }
}
```

## Next Steps

### Immediate (< 5 minutes)
- [x] Build and run simple test
- [x] Build and run full network
- [ ] Look at output file: `cnn_output.txt`

### Short-term (< 1 hour)
- [ ] Read through cnn_network.cpp
- [ ] Understand one layer in detail (start with pooling, it's simplest)
- [ ] Modify testbench to use smaller input (32×32)
- [ ] Add debug prints to see intermediate values

### Medium-term (few hours)
- [ ] Replace random weights with your trained weights
- [ ] Test with real images from your dataset
- [ ] Profile performance (time each layer)
- [ ] Optimize bottlenecks

### Long-term (days)
- [ ] Run HLS synthesis: `make hls`
- [ ] Analyze resource usage
- [ ] Optimize for your FPGA
- [ ] Integrate with larger system

## Common Questions

**Q: Why two accumulator types?**
A: 8-bit × 8-bit = 16-bit, but with 3×3×3 = 27 MACs, we need 32-bit to avoid overflow.

**Q: What's the difference between conv_layer_simple and conv_layer_stream?**
A: 
- `simple`: Uses 2D arrays, easier to debug
- `stream`: Uses line buffers, better for HLS, more efficient

**Q: Why is dropout a no-op?**
A: Dropout is only used during training. For inference (what we're doing), we don't drop neurons.

**Q: How do I load real images?**
A: You'll need to add image loading code. For JPG/PNG, use OpenCV or stb_image library.

**Q: Can I change from 8-bit to 16-bit?**
A: Yes! In cnn_types.h, change `ap_int<8>` to `ap_int<16>`. Be aware this doubles memory usage.

## Need Help?

1. **Check QUICK_REFERENCE.md** for layer details
2. **Check README.md** for comprehensive docs
3. **Run simple_conv_test** to isolate issues
4. **Add debug prints** to see what's happening
5. **Start small** - test with tiny inputs first

## Architecture Reminder

```
INPUT: 3×128×128 RGB image
  ↓
CONV1 (3→16, 3×3) + ReLU → 16×126×126
  ↓
AvgPool (2×2) → 16×63×63
  ↓
CONV2 (16→32, 3×3) + ReLU → 32×61×61
  ↓
AvgPool (2×2) → 32×30×30
  ↓
CONV3 (32→32, 3×3, stride 2) + ReLU → 32×14×14
  ↓
MaxPool (2×2) → 32×8×4
  ↓
Flatten → 1024
  ↓
FC1 (1024→256) + ReLU → 256
  ↓
Dropout (p=0.5, no-op) → 256
  ↓
FC2 (256→4) → 4 classes
  ↓
OUTPUT: Class logits
```

Good luck! 🚀
