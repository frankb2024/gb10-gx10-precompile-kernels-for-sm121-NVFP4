#!/usr/bin/env python3
"""
precompile-kernels.py — pre-compile flashinfer JIT kernels for a target arch
WITHOUT needing vLLM running. Use this on GB10 / sm_121a (or any arch with
sparse prebuilts in flashinfer-cubin) to warm the cache so vLLM startup is fast.

WHAT THIS DOES:
  Calls flashinfer.jit.<module>.gen_*_module() for the kernels your model
  needs, then calls JitSpec.build() to actually run nvcc + ninja and write
  .cuda.o files into ~/.cache/flashinfer/<version>/<arch>/cached_ops/.

WHY THIS IS NEEDED:
  flashinfer-cubin (the prebuilt kernel wheel) ships cubins for sm_80/87/89/
  90/100 but NOT for sm_120/121 (the GB10/DGX Spark chip). So on first run
  vLLM must JIT-compile 80+ sm_120 CUTLASS MoE kernels via nvcc, which on
  a 20-core box takes 30-60 minutes and saturates RAM+disk. Pre-compiling
  out-of-band lets you do that work on your schedule, see clean progress
  and errors, and have vLLM come up in seconds afterwards.

USAGE:
  # 122B NVFP4 MoE (most affected):
  python3 precompile-kernels.py --model /path/to/Qwen3.5-122B-A10B-NVFP4

  # Dense Qwen3.5 (2B/9B), much smaller:
  python3 precompile-kernels.py --model /path/to/Qwen3.5-2B

  # Just run it — interactive model picker (DEFAULT when nothing specified):
  python3 precompile-kernels.py
  # (or explicitly: python3 precompile-kernels.py --pick-model)

  # All sm_120/sm_121 kernels for any model on this GPU:
  python3 precompile-kernels.py --all-sm121

  # Just the MoE path:
  python3 precompile-kernels.py --modules fused_moe_sm120,fp4_quantization_sm121

  # Control parallelism (cap nvcc jobs to keep box responsive):
  python3 precompile-kernels.py --model <path> --jobs 4

NOTES:
  * CUDA toolkit must be on PATH (nvcc + ninja). flashinfer uses nvcc from
    PATH or from torch.utils.cpp_extension.CUDA_HOME.
  * Disk usage during build: ~1-3 GB in ~/.cache/flashinfer/ for the 122B.
  * Wall time for the 122B on a 20-core box: 30-60 min with --jobs 20,
    60-90 min with --jobs 4. Flashinfer's gen_*_module() uses ninja
    internally with -j controlled by the env var FLASHINFER_MAX_JOBS.
  * After running this, vLLM startup for the same model skips the JIT
    phase entirely.
"""

import argparse
import inspect
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path


# ---- Module selection -----------------------------------------------------
#
# Each entry is (display_name, callable_that_returns_JitSpec_or_None).
# The callable is called with no args; if it raises or returns None we
# skip and report why. "None" can be returned if the kernel is incompatible
# with the target arch.

# All the flashinfer.jit kernel generators that don't require shape args
# (i.e. "compile the whole family" calls).
SIMPLE_KERNELS = [
    ("act_and_mul",            "flashinfer.jit.activation",         "gen_act_and_mul_module"),
    ("bgmv_moe",               "flashinfer.jit.bgmv_moe",           "gen_bgmv_moe_module"),
    ("cascade",                "flashinfer.jit.cascade",            "gen_cascade_module"),
    ("comm_alltoall",          "flashinfer.jit.comm",               "gen_comm_alltoall_module"),
    ("dcp_alltoall",           "flashinfer.jit.comm",               "gen_dcp_alltoall_module"),
    ("moe_alltoall",           "flashinfer.jit.comm",               "gen_moe_alltoall_module"),
    ("trtllm_comm",            "flashinfer.jit.comm",               "gen_trtllm_comm_module"),
    ("trtllm_mnnvl_comm",      "flashinfer.jit.comm",               "gen_trtllm_mnnvl_comm_module"),
    ("vllm_comm",              "flashinfer.jit.comm",               "gen_vllm_comm_module"),
    ("cudnn_fmha",             None,                                "gen_cudnn_fmha_module"),
    ("dsv3_fused_routing",     "flashinfer.jit.dsv3_optimizations", "gen_dsv3_fused_routing_module"),
    ("dsv3_router_gemm",       "flashinfer.jit.dsv3_optimizations", "gen_dsv3_router_gemm_module"),
    ("fp4_kv_dequantization",  "flashinfer.jit.fp4_kv_dequantization", "gen_fp4_kv_dequantization_module"),
    ("fp4_kv_quantization",    "flashinfer.jit.fp4_kv_quantization",   "gen_fp4_kv_quantization_module"),
    ("mhc",                    "flashinfer.jit.mhc",                "gen_mhc_module"),
    ("mla",                    "flashinfer.jit.mla",                "gen_mla_module"),
    ("moe_utils",              "flashinfer.jit.moe_utils",          "gen_moe_utils_module"),
    ("norm",                   "flashinfer.jit.norm",               "gen_norm_module"),
    ("page",                   "flashinfer.jit.page",               "gen_page_module"),
    ("quantization",           "flashinfer.jit.quantization",       "gen_quantization_module"),
    ("tinygemm2",              None,                                "gen_tinygemm2_module"),
    ("trtllm_gen_fmha",        None,                                "gen_trtllm_gen_fmha_module"),
]

# Arch-specific generators in flashinfer.jit.fused_moe
FUSED_MOE_SM = [
    ("fused_moe_sm120",  "flashinfer.jit.fused_moe", "gen_cutlass_fused_moe_sm120_module"),
    ("fused_moe_sm100",  "flashinfer.jit.fused_moe", "gen_cutlass_fused_moe_sm100_module"),
    ("fused_moe_sm90",   "flashinfer.jit.fused_moe", "gen_cutlass_fused_moe_sm90_module"),
    ("fused_moe_sm89",   "flashinfer.jit.fused_moe", "gen_cutlass_fused_moe_sm89_module"),
]

# Arch-specific FP4 quantization (has sm120, sm121, sm120f, sm100, etc.)
FP4_QUANT_SM = [
    ("fp4_quantization_sm121",  "flashinfer.jit.fp4_quantization", "gen_fp4_quantization_sm121_module"),
    ("fp4_quantization_sm120",  "flashinfer.jit.fp4_quantization", "gen_fp4_quantization_sm120_module"),
    ("fp4_quantization_sm120f", "flashinfer.jit.fp4_quantization", "gen_fp4_quantization_sm120f_module"),
    ("fp4_quantization_sm100",  "flashinfer.jit.fp4_quantization", "gen_fp4_quantization_sm100_module"),
    ("fp4_quantization_sm90",   "flashinfer.jit.fp4_quantization", "gen_fp4_quantization_sm90_module"),
]

# Attention kernels for sm120 specifically
ATTENTION_SM120 = [
    ("trtllm_fmha_v2_sm120",  "flashinfer.jit", "gen_trtllm_fmha_v2_sm120_module"),
]


# ---- Model-aware selection ----------------------------------------------

def detect_arch_from_torch(verbose=True):
    """
    Return torch arch string like 'sm_121a' for the local GPU.

    Set verbose=False to suppress the "could not detect" warning (useful
    when probing for the banner — the caller will handle missing arch
    with a more user-friendly message).
    """
    try:
        import torch
        cap = torch.cuda.get_device_capability(0)
        return f"sm_{cap[0]}{cap[1]}"
    except Exception as e:
        if verbose:
            print(f"  WARN: could not detect GPU arch via torch: {e}", file=sys.stderr)
        return None


def inspect_model(model_dir):
    """
    Read config.json and infer what kernels the model needs.

    Returns a set of kernel group names from the ones defined above.
    """
    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        print(f"  WARN: {config_path} not found — cannot auto-detect", file=sys.stderr)
        return None

    try:
        import json
        cfg = json.loads(config_path.read_text())
    except Exception as e:
        print(f"  WARN: could not read config.json: {e}", file=sys.stderr)
        return None

    arch = (cfg.get("architectures") or ["unknown"])[0]
    text_cfg = cfg.get("text_config", cfg)
    model_type = cfg.get("model_type", text_cfg.get("model_type", ""))
    quant_cfg = cfg.get("quantization_config") or {}
    quant_method = quant_cfg.get("quant_method") or quant_cfg.get("fmt") or ""
    has_vision = "vision_config" in cfg

    print(f"  model architecture : {arch}")
    print(f"  model_type         : {model_type}")
    print(f"  quant              : {quant_method or '(none)'}")
    print(f"  vision (multimodal): {has_vision}")

    # Always-needed baseline (attention + quantization)
    needed = set()

    # Detect MoE
    is_moe = (
        "Moe" in arch or "moe" in model_type
        or text_cfg.get("num_local_experts") or text_cfg.get("num_experts")
    )

    if is_moe:
        num_experts = text_cfg.get("num_local_experts") or text_cfg.get("num_experts") or 0
        print(f"  num_experts        : {num_experts}")
        # Add MoE + MoE-comms
        needed.update(["fused_moe_sm120", "moe_utils", "moe_alltoall",
                       "bgmv_moe", "comm_alltoall", "trtllm_comm"])
        # NVFP4 specifically → fp4 quantization kernels
        if "nvfp" in str(quant_method).lower() or "fp4" in str(quant_method).lower():
            needed.update(["fp4_quantization_sm121", "fp4_quantization_sm120"])
            # Also the FP4 KV cache helpers if vLLM uses them
            needed.update(["fp4_kv_quantization", "fp4_kv_dequantization"])

    # FlashAttention / FMHA — needed by all transformer models on this arch
    needed.add("trtllm_fmha_v2_sm120")
    needed.add("cudnn_fmha")
    needed.add("trtllm_gen_fmha")

    # Norm + page + quantization always useful
    needed.update(["norm", "page", "quantization"])

    return needed


def get_kernel_callable(module_path, func_name):
    """Import the module and return the callable."""
    if module_path is None:
        # Function lives in flashinfer.jit directly
        import flashinfer.jit as jit_pkg
        return getattr(jit_pkg, func_name)
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, func_name)


def _summarize_cache(cached_ops_dir, arch_label):
    """
    Print a 3-line summary of an existing flashinfer cache directory:
    full path, file count + total size, kernel count for the arch.
    Returns True if the dir exists, False otherwise.
    """
    p = Path(cached_ops_dir)
    if not p.exists():
        print(f"  flashinfer cache: (none yet) — first run on this box")
        return False
    total_files = 0
    total_bytes = 0
    for f in p.rglob("*"):
        if f.is_file():
            total_files += 1
            total_bytes += f.stat().st_size
    n_kernels = sum(1 for d in p.iterdir() if d.is_dir())
    print(f"  flashinfer cache: {p}")
    print(f"    total: {total_files} files, {_human_bytes(total_bytes)}")
    print(f"    {n_kernels} kernel(s) cached for arch {arch_label}")
    return True


def print_banner(args):
    """
    Print a top-of-output banner describing the tool, current environment,
    and the most common CLI options. Shown BEFORE any interactive prompts
    so the user knows what they're about to be asked.
    """
    # Detect arch silently for the banner; main() does its own verbose
    # detection and emits a clearer error if it fails.
    arch = args.arch or detect_arch_from_torch(verbose=False) or "?"
    # Determine venv name. Try (in order):
    #   1. $VIRTUAL_ENV env var (set by `source venv/bin/activate`)
    #   2. pyvenv.cfg within a few levels of sys.executable (works for
    #      symlinked venvs that point at the system Python)
    #   3. walk up sys.executable looking for a .venvs/<name> or venv/<name>
    venv = "(none)"
    ve = os.environ.get("VIRTUAL_ENV")
    py = Path(sys.executable)
    if ve:
        venv = Path(ve).name
    else:
        # Look for pyvenv.cfg in parents of sys.executable and bin/
        for candidate in [py.parent, py.parent.parent]:
            cfg = candidate / "pyvenv.cfg"
            if cfg.is_file():
                venv = candidate.name
                break
        if venv == "(none)":
            # Last resort: walk path parts
            parts = py.resolve().parts
            for i, p in enumerate(parts):
                if p in (".venvs", "venvs") and i + 1 < len(parts):
                    venv = parts[i + 1]
                    break

    # Try to read the existing flashinfer cache summary. If flashinfer
    # is not importable (system python, no venv) we silently skip this
    # so the banner still prints cleanly.
    cache_line = "  flashinfer cache: (flashinfer not importable in this Python)"
    try:
        import flashinfer
        import flashinfer.jit.env as fi_env
        fi_version = flashinfer.__version__
        # FLASHINFER_JIT_DIR points to the exact cached_ops dir for the
        # active arch (e.g. .../0.6.13/121a/cached_ops).
        fi_jit_dir = Path(fi_env.FLASHINFER_JIT_DIR)
        fi_arch_label = fi_jit_dir.parent.name  # e.g. "121a"
        # Build a 3-line summary in-place
        if fi_jit_dir.exists():
            total_files = 0
            total_bytes = 0
            for f in fi_jit_dir.rglob("*"):
                if f.is_file():
                    total_files += 1
                    total_bytes += f.stat().st_size
            n_kernels = sum(1 for d in fi_jit_dir.iterdir() if d.is_dir())
            cache_line = (
                f"  flashinfer cache: {fi_jit_dir}\n"
                f"    total: {total_files} files, {_human_bytes(total_bytes)}\n"
                f"    {n_kernels} kernel(s) cached for arch {fi_arch_label} "
                f"(flashinfer {fi_version})"
            )
        else:
            cache_line = (
                f"  flashinfer cache: (none yet) — first run on this box "
                f"(flashinfer {fi_version})"
            )
    except Exception:
        pass  # keep default cache_line

    W = 72
    border = "=" * W
    print()
    print(border)
    print("  flashinfer kernel pre-compiler for vLLM")
    print(border)
    print(f"  python      : {py}")
    print(f"  venv        : {venv}")
    print(f"  arch        : {arch}")
    print(f"  jobs (-j)   : {args.jobs}    (nvcc parallelism; 4 = safe default)")
    print()
    print(cache_line)
    print()
    print("  WHAT THIS DOES")
    print("    Compiles flashinfer CUDA kernels out-of-band, ahead of vLLM startup,")
    print("    so vLLM doesn't have to JIT-compile them while serving requests.")
    print("    On GB10 (sm_121a) the 122B NVFP4 MoE needs ~80+ kernel compiles")
    print("    that vLLM would otherwise do at startup, taking 30-60 minutes.")
    print()
    print("  COMMON USAGE")
    print("    --pick-model                        # interactive model picker")
    print("    --model <path>                      # build kernels for one model")
    print("    --modules fused_moe_sm120,page      # build specific kernels only")
    print("    --all-sm121                         # every sm_121 kernel known")
    print("    --dry-run                           # show what would happen")
    print("    --force                             # skip the cache-confirm prompt")
    print()
    print("  FULL HELP")
    print("    --help    (or run: python precompile-kernels.py --help)")
    print(border)


# ---- Cache preflight + interactive confirmation -------------------------

def list_and_pick_model(hub_dir):
    """
    Scan `hub_dir` for HF model directories (each named 'models--ORG--NAME'),
    display them in 3 numbered columns, prompt the user for a number, and
    return the chosen model's snapshot path.

    Returns a Path to the snapshot dir (containing config.json + weights),
    or None if the user cancels / no models found.
    """
    hub_path = Path(hub_dir).expanduser()
    if not hub_path.is_dir():
        print(f"ERROR: HF hub dir not found: {hub_path}", file=sys.stderr)
        return None

    # Each model is models--<org>--<name> with a snapshots/<hash>/ inside
    models = []
    for entry in sorted(hub_path.iterdir()):
        if not entry.is_dir():
            continue
        if not entry.name.startswith("models--"):
            continue
        # Find first snapshot dir (HF models typically have one)
        snaps = entry / "snapshots"
        snap_dir = None
        if snaps.is_dir():
            sd_list = [d for d in snaps.iterdir() if d.is_dir()]
            if sd_list:
                snap_dir = sd_list[0]
        models.append({
            "name": entry.name,
            "path": entry,
            "snap": snap_dir,
            # Display label: strip 'models--' prefix, replace '--' with '/'
            "label": entry.name.replace("models--", "", 1).replace("--", "/", 1),
        })

    if not models:
        print(f"No model directories found in {hub_path}", file=sys.stderr)
        return None

    # Display in 3 columns with number prefix
    n = len(models)
    cols = 3
    # Compute column widths: number + label (truncate to fit)
    # Use terminal width if available, else 100
    try:
        term_w = os.get_terminal_size().columns
    except OSError:
        term_w = 100
    # Each entry like "  12. org/name            "
    # Reserve ~14 chars for "  N. " + " " = overhead
    col_w = max(30, min(50, (term_w - 4) // cols))

    rows = (n + cols - 1) // cols
    print()
    print("=" * term_w)
    print(f"  HF models in {hub_path}")
    print(f"  ({n} found)")
    print("=" * term_w)

    for r in range(rows):
        row_items = []
        for c in range(cols):
            i = r + c * rows
            if i >= n:
                row_items.append(" " * col_w)
                continue
            m = models[i]
            num = f"{i+1:3d}."
            # Truncate label to fit column
            label_max = col_w - len(num) - 2  # 2 spaces of padding
            label = m["label"]
            if len(label) > label_max:
                label = label[:label_max - 1] + "\u2026"
            row_items.append(f"{num} {label}".ljust(col_w))
        print("".join(row_items))

    print()
    print("=" * term_w)

    # Prompt for selection
    if not sys.stdin.isatty():
        print("ERROR: stdin is not a TTY; can't prompt for selection.", file=sys.stderr)
        print("       Pass --model <path> explicitly instead.", file=sys.stderr)
        return None

    while True:
        try:
            raw = input(f"  Pick a model [1-{n}] (or 'q' to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if raw.lower() in ("q", "quit", "exit"):
            return None
        try:
            idx = int(raw)
        except ValueError:
            print(f"  Please enter a number between 1 and {n} (or 'q' to quit).")
            continue
        if not (1 <= idx <= n):
            print(f"  Please enter a number between 1 and {n}.")
            continue
        chosen = models[idx - 1]
        if chosen["snap"] is None:
            print(f"  WARNING: {chosen['name']} has no snapshot dir (weights not "
                  f"downloaded yet?).")
            print(f"           Proceeding anyway — config.json won't be readable.")
            return chosen["path"]
        return chosen["snap"]


def _dir_size_and_count(path):
    """Return (file_count, total_bytes) for path, or None if missing."""
    if not path.is_dir():
        return None
    n = 0
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            n += 1
            total += f.stat().st_size
    return n, total


def _human_bytes(n):
    """Format bytes as a human string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def _kernel_artifact_path(flashinfer_version, arch, name):
    """
    Return the expected artifact (.so) path for a kernel, or None.
    This mirrors flashinfer's internal layout so we can detect what
    already exists.
    """
    # flashinfer puts results in FLASHINFER_JIT_DIR / <kernel_dir> / <name>.so
    # For the kernels we list, the 'kernel_dir' matches the kernel name in
    # nearly all cases. Per-kernel exceptions could be added here if needed.
    cache_root = Path.home() / ".cache" / "flashinfer"
    return cache_root / flashinfer_version / arch / "cached_ops" / name / f"{name}.so"


def preflight_cache(selected, arch, flashinfer_version):
    """
    Inspect the on-disk flashinfer cache and report:

    * every relevant directory (full path, exists, file count, size)
    * for each selected kernel: status (will-build / already-cached)
    * summary of work remaining

    The `arch` argument is the flashinfer-format arch dir name (e.g. '121a'),
    NOT torch's 'sm_121'. Resolved by main() from FLASHINFER_WORKSPACE_DIR.

    Returns (will_build_list, already_cached_list).
    """
    cache_root = Path.home() / ".cache" / "flashinfer"
    workspace = cache_root / flashinfer_version / arch
    jit_dir = workspace / "cached_ops"
    gen_src = workspace / "generated"

    print()
    print("=" * 70)
    print("EXISTING CACHE STATE")
    print("=" * 70)

    # Top-level cache
    if not cache_root.exists():
        print(f"  (no cache yet — first run on this box)")
    else:
        versions = sorted([d for d in cache_root.iterdir() if d.is_dir()])
        print(f"  {cache_root}/")
        for v in versions:
            n_files = sum(1 for _ in v.rglob("*") if _.is_file())
            sz = sum(f.stat().st_size for f in v.rglob("*") if f.is_file())
            print(f"    {v.name}/  ({n_files} files, {_human_bytes(sz)})")
            for a in sorted(v.iterdir()):
                if a.is_dir():
                    n_files = sum(1 for _ in a.rglob("*") if _.is_file())
                    sz = sum(f.stat().st_size for f in a.rglob("*") if f.is_file())
                    print(f"      {a.name}/  ({n_files} files, {_human_bytes(sz)})")

    # The specific arch workspace (this is what we'll be writing to)
    print()
    print(f"  Target arch workspace: {workspace}")
    if workspace.exists():
        for sub in ("cached_ops", "generated"):
            p = workspace / sub
            r = _dir_size_and_count(p)
            if r is None:
                print(f"    {sub}/  (missing)")
            else:
                n, sz = r
                print(f"    {sub}/  ({n} files, {_human_bytes(sz)})")
                # Per-kernel breakdown if cached_ops exists
                if sub == "cached_ops" and n > 0:
                    for kdir in sorted(p.iterdir()):
                        if kdir.is_dir():
                            kr = _dir_size_and_count(kdir)
                            if kr:
                                kn, ksz = kr
                                print(f"      {kdir.name}/  ({kn} files, "
                                      f"{_human_bytes(ksz)})")
    else:
        print(f"    (does not exist yet — will be created)")

    # Per-selected-kernel status
    print()
    print("=" * 70)
    print(f"PER-KERNEL STATUS ({len(selected)} selected for {arch})")
    print("=" * 70)
    already_cached = []
    will_build = []
    for name in sorted(selected):
        artifact = _kernel_artifact_path(flashinfer_version, arch, name)
        if artifact is not None and artifact.exists():
            sz = artifact.stat().st_size
            already_cached.append((name, artifact, sz))
            print(f"  [CACHED] {name}")
            print(f"            {artifact}  ({_human_bytes(sz)})")
        else:
            will_build.append((name, artifact))
            print(f"  [BUILD ] {name}")
            if artifact is not None:
                print(f"            -> {artifact}")

    return will_build, already_cached


def maybe_confirm_with_user(will_build, already_cached, force):
    """
    If there's substantial existing cached work AND there's also new
    build work, prompt the user to confirm. Returns True to proceed,
    False to abort.

    Rules:
      * If --force given, always proceed.
      * If nothing to build (everything cached), proceed silently.
      * If cache is empty (nothing cached), proceed silently.
      * Otherwise (some cached + some to build) prompt unless input is
        a TTY we can't read.
    """
    if force or not will_build or not already_cached:
        return True

    print()
    print("=" * 70)
    print("CONFIRMATION NEEDED")
    print("=" * 70)
    print(f"  {len(already_cached)} kernel(s) already cached.")
    print(f"  {len(will_build)} kernel(s) still need building.")
    print()
    print("  Building is slow (nvcc + ninja per kernel, 1-5 min each on GB10)")
    print("  and writes ~50-500 MB per MoE kernel to the cache directory.")
    print()

    # If stdin isn't a TTY (script piped), skip the prompt — assume yes.
    # User can re-run with --no-confirm / --force-style flag if they want.
    if not sys.stdin.isatty():
        print("  (stdin is not a TTY; proceeding automatically. Pass --force to suppress)")
        return True

    try:
        answer = input("  Continue and rebuild the missing kernels? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False

    return answer in ("y", "yes")


# ---- Build driver --------------------------------------------------------

class BuildStatus:
    def __init__(self):
        self.ok = []
        self.failed = []
        self.skipped = []


def build_one(name, gen_callable, jobs, verbose):
    """
    Call gen_callable() to get a JitSpec, then build() it.
    Returns ("ok", message) | ("fail", message) | ("skip", reason).
    """
    try:
        spec = gen_callable()
        if spec is None:
            return ("skip", "gen returned None")
    except Exception as e:
        return ("skip", f"gen failed: {type(e).__name__}: {e}")

    if spec.is_compiled:
        return ("skip", "already compiled")

    # flashinfer invokes ninja via subprocess which inherits os.environ.
    # We need to: (1) put ninja on PATH, (2) tell flashinfer's build to
    # use our --jobs value. The ninja binary lives at <venv>/bin/ninja
    # (pip-installed); flashinfer reads FLASHINFER_MAX_JOBS at module
    # import time so we also have to set it BEFORE the first import.
    venv_bin = Path(sys.executable).parent
    os.environ["PATH"] = str(venv_bin) + os.pathsep + os.environ.get("PATH", "")
    os.environ["FLASHINFER_MAX_JOBS"] = str(jobs)
    os.environ["MAX_JOBS"] = str(jobs)
    try:
        t0 = time.time()
        spec.build(verbose=verbose)
        dt = time.time() - t0
        return ("ok", f"built in {dt:.1f}s -> {spec.get_library_path()}")
    except Exception as e:
        tb = traceback.format_exc(limit=3)
        return ("fail", f"{type(e).__name__}: {e}\n{tb}")


def main():
    ap = argparse.ArgumentParser(
        description="Pre-compile flashinfer JIT kernels for a target arch "
                    "(sm_120/sm_121a on GB10, etc.).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--model", help="Path to a HF model dir (used to auto-detect "
                                   "which kernels to build). config.json is read.")
    ap.add_argument("--pick-model", action="store_true",
                    help="Interactively pick a model from ~/USER/.cache/huggingface/hub/ "
                         "(lists in 3 columns, prompts for a number). Sets --model "
                         "to the chosen model's snapshot dir.")
    ap.add_argument("--hub-dir", default=None,
                    help="Override the HF hub dir to scan for --pick-model. "
                         "Default: $HOME/.cache/huggingface/hub/")
    ap.add_argument("--arch", help="Target arch (e.g. sm_121). Auto-detected from "
                                  "torch if omitted.")
    ap.add_argument("--modules", help="Comma-separated list of kernel names from "
                                      "the curated set. Overrides --model "
                                      "auto-detection if both given.")
    ap.add_argument("--all-sm120", action="store_true",
                    help="Build every kernel we know about for sm_120 (the 122B path).")
    ap.add_argument("--all-sm121", action="store_true",
                    help="Build every kernel we know about for sm_121 (full GB10 set).")
    ap.add_argument("--jobs", "-j", type=int, default=4,
                    help="Max parallel nvcc jobs (env FLASHINFER_MAX_JOBS). Default 4.")
    ap.add_argument("--verbose", "-v", action="store_true", help="Pass verbose=True to spec.build().")
    ap.add_argument("--list", action="store_true",
                    help="List available kernels for the detected arch and exit.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be built without actually building.")
    ap.add_argument("--force", action="store_true",
                    help="Skip the 'existing cache detected, continue?' prompt.")
    args = ap.parse_args()

    # Show the banner before any interactive prompt or other output.
    # This is the first thing the user sees so they understand what
    # they're about to be asked.
    print_banner(args)

    # Default behavior: if the user didn't pick a model AND didn't say
    # which kernels to build, drop into the interactive model picker.
    # This makes `python3 precompile-kernels.py` do the right thing
    # without forcing the user to remember a flag.
    has_any_selector = (
        args.pick_model
        or args.model
        or args.modules
        or args.all_sm120
        or args.all_sm121
    )
    if not has_any_selector:
        print("  No model or kernel selection given; entering interactive picker.\n")
        args.pick_model = True

    # Handle --pick-model BEFORE other detection so it can set args.model.
    if args.pick_model:
        hub_dir = args.hub_dir or str(Path.home() / ".cache" / "huggingface" / "hub")
        chosen = list_and_pick_model(hub_dir)
        if chosen is None:
            return 0
        args.model = str(chosen)
        print(f"  -> Selected: {chosen}")
        print()

    # Detect arch. The banner already showed `arch: ?` if torch is missing,
    # so suppress the warning here and just emit one clean error line.
    arch = args.arch or detect_arch_from_torch(verbose=False)
    if not arch:
        print(f"ERROR: cannot detect GPU arch via torch. Either:", file=sys.stderr)
        print(f"       - run this script with the venv's Python:", file=sys.stderr)
        print(f"         /home/ai/.venvs/vllm/bin/python {sys.argv[0]}", file=sys.stderr)
        print(f"       - or pass --arch explicitly:", file=sys.stderr)
        print(f"         python3 {sys.argv[0]} --arch sm_121 ...", file=sys.stderr)
        return 2
    print(f"Target arch: {arch}")
    print(f"Parallelism: {args.jobs} nvcc jobs")

    # Resolve which kernels to build
    selected = set()

    if args.modules:
        selected.update(args.modules.split(","))

    if args.all_sm120:
        selected.update(n for n, _, _ in FUSED_MOE_SM if "sm120" in n)
        selected.update(n for n, _, _ in FP4_QUANT_SM if "sm120" in n)
        selected.update(n for n, _, _ in ATTENTION_SM120 if "sm120" in n)
        selected.update(n for n, _, _ in SIMPLE_KERNELS)

    if args.all_sm121:
        selected.update(n for n, _, _ in FUSED_MOE_SM if "sm120" in n)  # sm120 is forward-compat
        selected.update(n for n, _, _ in FP4_QUANT_SM if "sm121" in n or "sm120" in n)
        selected.update(n for n, _, _ in ATTENTION_SM120 if "sm120" in n)
        selected.update(n for n, _, _ in SIMPLE_KERNELS)

    if args.model:
        detected = inspect_model(args.model)
        if detected is not None:
            print(f"  auto-detected needed kernels: {sorted(detected)}")
            selected.update(detected)

    if not selected:
        print("ERROR: no kernels selected. Use --model, --modules, --all-sm120, or --all-sm121.",
              file=sys.stderr)
        return 2

    # Match selected names against known kernels.
    # Each entry in *_KERNELS is (name, module_path_or_None, func_name).
    # Build a {name: (module_path, func_name)} dict.
    all_known = {}
    for entry in (SIMPLE_KERNELS + FUSED_MOE_SM + FP4_QUANT_SM + ATTENTION_SM120):
        all_known[entry[0]] = (entry[1], entry[2])
    unknown = selected - set(all_known.keys())
    if unknown:
        print(f"ERROR: unknown kernel names: {sorted(unknown)}", file=sys.stderr)
        print(f"       known: {sorted(all_known.keys())}", file=sys.stderr)
        return 2

    print(f"Will build {len(selected)} kernel(s): {sorted(selected)}")

    # Cache preflight + interactive confirmation.
    # Only do this when we'd actually build (i.e. not --list / not --dry-run
    # for the model-detection case, etc.).
    if args.list:
        return 0

    # Resolve flashinfer version dynamically (don't hardcode; flashinfer
    # versions can differ between venvs). Also get the actual arch dir
    # name flashinfer uses (it appends 'a' for accelerated features, e.g.
    # torch says "sm_121" but flashinfer's dir is "121a").
    try:
        import flashinfer
        import flashinfer.jit.env as fi_env
        flashinfer_version = flashinfer.__version__
        # Use the actual dir name flashinfer uses, not torch's naming.
        flashinfer_arch = Path(fi_env.FLASHINFER_WORKSPACE_DIR).name
    except ImportError:
        print("ERROR: flashinfer is not importable. Run this script with the venv's Python "
              "(e.g. /home/ai/.venvs/vllm/bin/python).", file=sys.stderr)
        return 2

    will_build, already_cached = preflight_cache(
        selected, flashinfer_arch, flashinfer_version,
    )

    if args.dry_run:
        return 0

    if not maybe_confirm_with_user(will_build, already_cached, args.force):
        print("Aborted by user.")
        return 0

    # Set parallelism BEFORE importing flashinfer (it reads FLASHINFER_MAX_JOBS at import time)
    os.environ["FLASHINFER_MAX_JOBS"] = str(args.jobs)
    os.environ["MAX_JOBS"] = str(args.jobs)

    print()
    print("=" * 70)
    print(f"Building {len(selected)} kernels with {args.jobs} parallel jobs")
    print("=" * 70)

    status = BuildStatus()
    t_start = time.time()

    for i, name in enumerate(sorted(selected), 1):
        module_path, func_name = all_known[name]

        print(f"[{i}/{len(selected)}] {name} ...", end=" ", flush=True)
        try:
            gen_callable = get_kernel_callable(module_path, func_name)
        except Exception as e:
            print(f"SKIP (import: {e})")
            status.skipped.append((name, f"import: {e}"))
            continue

        result, msg = build_one(name, gen_callable, args.jobs, args.verbose)
        if result == "ok":
            print(f"OK ({msg})")
            status.ok.append(name)
        elif result == "skip":
            print(f"SKIP ({msg})")
            status.skipped.append((name, msg))
        else:
            print(f"FAIL")
            print(f"    {msg}")
            status.failed.append(name)

    dt = time.time() - t_start
    print()
    print("=" * 70)
    print(f"Done in {dt/60:.1f} min: {len(status.ok)} ok, "
          f"{len(status.skipped)} skipped, {len(status.failed)} failed")
    print("=" * 70)

    # Show final cache stats using the same helper the banner uses.
    cache_root = Path.home() / ".cache" / "flashinfer" / flashinfer_version / flashinfer_arch / "cached_ops"
    print()
    _summarize_cache(cache_root, flashinfer_arch)

    if status.failed:
        print()
        print(f"WARNING: {len(status.failed)} kernel(s) failed to build:")
        for n in status.failed:
            print(f"  - {n}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())