
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

    What it does:
    - Reads flashinfer 0.6.13's gen_*_module() API to JIT-compile CUDA kernels for sm_120/sm_121
    - Auto-detects what a model needs by reading its config.json (MoE? NVFP4? Multimodal?)
    - Caches into the same ~/.cache/flashinfer/<ver>/<arch>/cached_ops/ that vLLM uses
    - Idempotent — re-running skips already-cached kernels
    - Caps nvcc parallelism via FLASHINFER_MAX_JOBS
    - Surfaces build errors clearly with tracebacks
    
    Verified working end-to-end:
    - --dry-run against 122B model: detected 12 kernels needed (MoE + sm120 FMHA + sm121 FP4 quant + comms)
    - Real build of norm kernel: 14.4s, 1 ok, 0 failed
    - Re-run of same kernel: correctly skipped as already-compiled
    - Dry-run against dense 2B model: detected 6 kernels (different set, no MoE/FP4)
    
    How to use it for the 122B     
    
        - run this script with the venv's Python:
         /home/ai/.venvs/vllm/bin/python precompile-kernels.py
         
       - or pass --arch explicitly:
         python3 precompile-kernels.py --arch sm_121 





      WHAT THIS DOES
        Compiles flashinfer CUDA kernels out-of-band, ahead of vLLM startup,
        so vLLM doesn't have to JIT-compile them while serving requests.
        On GB10 (sm_121a) the 122B NVFP4 MoE needs ~80+ kernel compiles
        that vLLM would otherwise do at startup, taking 30-60 minutes.
    
      COMMON USAGE
        --pick-model                        # interactive model picker
        --model <path>                      # build kernels for one model
        --modules fused_moe_sm120,page      # build specific kernels only
        --all-sm121                         # every sm_121 kernel known
        --dry-run                           # show what would happen
        --force                             # skip the cache-confirm prompt
    
      FULL HELP
        --help    (or run: python precompile-kernels.py --help)
        

         
