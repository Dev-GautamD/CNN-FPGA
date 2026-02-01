# Weight Loading Methods - Quick Comparison

## Two Approaches Available

You now have **two ways** to use your ship detector weights:

### Method 1: File-Based Loading
**Use for:** Development, testing, trying different weights

```bash
# Build
make ship_detector_test

# Run (needs weights.bin and input.bin)
./ship_detector_test
```

**Pros:**
- Easy to swap different weight files
- Good for development
- Test multiple models easily

**Cons:**
- ❌ Doesn't work on FPGA (no file system!)
- ❌ Runtime file I/O overhead
- ❌ Need to manage files
- ❌ Can have missing/corrupt file errors

---

### Method 2: Embedded Weights (Recommended for FPGA!)
**Use for:** FPGA deployment, production, final implementation

```bash
# Step 1: Generate header (one time)
python3 weights_to_header.py weights.bin input.bin ship_weights.h

# Step 2: Build
make embedded_test

# Step 3: Run (no files needed!)
./embedded_test
```

**Pros:**
- ✅ Works on FPGA (no file I/O!)
- ✅ Weights in ROM/BRAM at compile time
- ✅ Faster (no loading overhead)
- ✅ Simpler (just compile and go)
- ✅ Can't have missing files
- ✅ HLS can optimize better

**Cons:**
- Need to regenerate header when weights change
- Binary size increases (but this is fine)

---

## Side-by-Side Code Comparison

### File-Based Approach

```cpp
#include "weight_loader.h"

int main() {
    // Load from files at runtime
    WeightLoader loader("weights.bin");
    if (!loader.is_valid()) {
        // Error: file not found!
        return 1;
    }
    
    loader.load_conv_weights<...>(weights);
    
    // Load input from file
    load_input_image("input.bin", input, 128, 128);
    
    // Run CNN
    cnn_network(...);
}
```

### Embedded Approach

```cpp
#include "embedded_weight_loader.h"
#include "ship_weights.h"  // Generated once

int main() {
    // Use embedded const arrays
    EmbeddedWeightLoader loader(SHIP_DETECTOR_WEIGHTS);
    // No file I/O - weights already in memory!
    
    loader.load_conv_weights<...>(weights);
    
    // Use embedded input
    load_embedded_input(SHIP_DETECTOR_INPUT, input, 128, 128);
    
    // Run CNN
    cnn_network(...);
}
```

---

## What Happens Under the Hood

### File-Based

```
Runtime:
1. Open "weights.bin" → File I/O
2. Read 277 KB from disk → Slow
3. Copy to memory
4. Close file
5. Run CNN

FPGA: ❌ No file system!
```

### Embedded

```
Compile time:
1. ship_weights.h has const arrays
2. Compiler puts them in ROM section
3. Done!

Runtime:
1. Weights already in ROM → Instant
2. Run CNN → Fast

FPGA: ✅ Works perfectly!
```

---

## Memory Layout

### File-Based
```
FPGA Memory:
┌─────────────┐
│ Your Code   │
├─────────────┤
│ Work Vars   │
├─────────────┤
│ Weight Buf  │ ← Copied at runtime (wasteful!)
└─────────────┘

ROM: Empty (wasted space)
```

### Embedded
```
FPGA Memory:
┌─────────────┐
│ Your Code   │
├─────────────┤
│ Work Vars   │
└─────────────┘

ROM:
┌─────────────┐
│ Weights     │ ← Stored here (efficient!)
└─────────────┘
```

---

## Which Should You Use?

### For Your Ship Detector Project:

**Development Phase:**
- ✅ Use **File-Based** (ship_detector_test)
- Easy to test different weights
- Quick iterations

**FPGA Deployment:**
- ✅ Use **Embedded** (embedded_test)
- No file I/O
- Production-ready
- HLS synthesis works

**Best Practice:**
1. Develop with file-based
2. Test thoroughly
3. Switch to embedded for final deployment
4. Both use the same CNN code!

---

## Quick Reference

| Feature | File-Based | Embedded |
|---------|-----------|----------|
| Works on FPGA | ❌ No | ✅ Yes |
| File I/O needed | ✅ Yes | ❌ No |
| Easy to change weights | ✅ Yes | ⚠️ Regenerate header |
| Runtime overhead | ⚠️ File loading | ✅ None |
| HLS synthesis | ❌ Difficult | ✅ Easy |
| Binary size | ✅ Small | ⚠️ Larger |
| Development | ✅ Great | ⚠️ OK |
| Production | ❌ Bad | ✅ Great |

---

## Commands Summary

### File-Based
```bash
# Need: weights.bin, input.bin
make ship_detector_test
./ship_detector_test
```

### Embedded
```bash
# One-time: generate header
python3 weights_to_header.py weights.bin input.bin ship_weights.h

# Then just build and run
make embedded_test
./embedded_test
# No files needed!
```

---

## Recommended Workflow

```
┌─────────────────────────────────────────┐
│ 1. Train Model → Get weights.bin        │
└───────────────┬─────────────────────────┘
                │
┌───────────────▼─────────────────────────┐
│ 2. Test with File-Based Loader          │
│    make ship_detector_test               │
│    ./ship_detector_test                  │
└───────────────┬─────────────────────────┘
                │
┌───────────────▼─────────────────────────┐
│ 3. Generate Embedded Header              │
│    python weights_to_header.py ...       │
└───────────────┬─────────────────────────┘
                │
┌───────────────▼─────────────────────────┐
│ 4. Test Embedded Version                 │
│    make embedded_test                    │
│    ./embedded_test                       │
└───────────────┬─────────────────────────┘
                │
┌───────────────▼─────────────────────────┐
│ 5. Deploy to FPGA with HLS               │
│    Use embedded version!                 │
└─────────────────────────────────────────┘
```

---

## Bottom Line

**You were right!** For FPGA deployment with fixed weights:

✅ **Embedded weights are the way to go!**

The file-based loader is included for completeness and development convenience, but for your final FPGA implementation, you'll definitely want to use the embedded approach.

Both methods are now available in your project. Use whichever fits your current needs! 🎯
