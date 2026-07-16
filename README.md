
Running NVFP4 models on the sm_121 compute capability (like the DGX Spark / GB10 Blackwell) often fails because
prebuilt binaries and attention/MoE kernels lack sm_121 targets. Without native compilation, you will experience 
catastrophic fallbacks (such as PTX illegal instructions) resulting in extremely slow generation or server 
crashes.

Why the Precompilation Issue Happens

Compute Split: The Blackwell architecture split into datacenter 
targets (sm_100a) and consumer targets (sm_120f/sm_121). Shipped .so files from PyTorch and frameworks only 
bundled kernels up to sm_120, which causes initialization to break on sm_121 GPUs.

Native NVFP4 Math:
Hardware-accelerated 4-bit float instructions require specialized backends to map directly to the Tensor Cores. 
Without a precompiled custom kernel, the framework attempts a software-level dequantization which produces 
garbage output (often displayed as exclamation marks !!!!!) or a segmentation fault.

How to Resolve and Run 
SuccessfullyTo get around the missing sm_121 target, the software stack must be manually compiled or 
configured with specific fallbacks:

Use FlashInfer JIT/Dual Arch: FlashInfer allows just-in-time (JIT) 
compiling which will correctly target the sm_120f/sm_121 hardware.
