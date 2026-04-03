#!/usr/bin/env python3
"""
Patch upstream ggml-sycl to match ollama's modified ggml backend API
and fix known upstream bugs not yet merged.

Accepts either the ggml-sycl directory or a single .cpp file (legacy).

Patches applied:
  ggml-sycl.cpp:
    1. graph_compute() batch_size parameter  (old ollama API mismatch)
    2. GGML_TENSOR_FLAG_COMPUTE removal      (old ollama API mismatch)
    3. Add BF16 to GET_ROWS supports_op      (upstream bug, PR #21391)
  getrows.cpp:
    4. Add BF16 dispatch for GET_ROWS        (upstream bug, PR #21391)

As of ollama v0.16.1+, patches 1-2 are no longer needed (APIs converged).
Patches 3-4 are needed until upstream llama.cpp PR #21391 is merged.
"""

import os
import re
import sys


def patch_file(path, patches):
    """Apply a list of (name, needs_patch_fn, apply_fn) to a file."""
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        src = f.read()
    original = src
    applied = []
    for name, needs_patch, apply_patch in patches:
        if needs_patch(src):
            new_src = apply_patch(src)
            if new_src != src:
                src = new_src
                applied.append(name)
    if applied:
        with open(path, "w") as f:
            f.write(src)
    return applied


def patch_ggml_sycl_cpp(sycl_dir):
    """Patch ggml-sycl.cpp: API compat + BF16 GET_ROWS supports_op."""
    path = os.path.join(sycl_dir, "ggml-sycl.cpp")
    return patch_file(path, [
        # 1. Fix graph_compute signature: add 'int batch_size' parameter.
        #    Only needed if the function does NOT already have batch_size.
        (
            "batch_size parameter",
            lambda src: "ggml_backend_sycl_graph_compute" in src
                        and "int batch_size" not in src,
            lambda src: re.sub(
                r'(ggml_backend_sycl_graph_compute\([^)]*int\s+batch_size\)\s*\{)',
                r'\1\n    GGML_UNUSED(batch_size);',
                re.sub(
                    r'(static\s+(?:enum\s+)?ggml_status\s+ggml_backend_sycl_graph_compute\s*\([^)]*cgraph)\s*\)',
                    r'\1, int batch_size)',
                    src,
                ),
            ),
        ),
        # 2. Remove GGML_TENSOR_FLAG_COMPUTE skip-check.
        #    Only needed if the flag is still referenced in the source.
        (
            "GGML_TENSOR_FLAG_COMPUTE removed",
            lambda src: "GGML_TENSOR_FLAG_COMPUTE" in src,
            lambda src: re.sub(
                r'\s*if\s*\(\(node->flags\s*&\s*GGML_TENSOR_FLAG_COMPUTE\)\s*==\s*0\)\s*\{\s*continue;\s*\}',
                '',
                src,
            ),
        ),
        # 3. Add BF16 to GET_ROWS supports_op.
        #    Upstream PR: https://github.com/ggml-org/llama.cpp/pull/21391
        #    The switch lists F16, F32, Q4_0, ... but omits BF16, causing
        #    models with BF16 tensors (e.g. Gemma4) to fall back to CPU.
        #    Matches the unique sequence "F16 / F32 / Q4_0" in the supports_op switch.
        (
            "BF16 added to GET_ROWS supports_op",
            lambda src: bool(re.search(
                r'case\s+GGML_TYPE_F16\s*:\s*\n\s*case\s+GGML_TYPE_F32\s*:\s*\n\s*case\s+GGML_TYPE_Q4_0',
                src,
            )) and not bool(re.search(
                r'case\s+GGML_TYPE_F16\s*:\s*\n\s*case\s+GGML_TYPE_BF16\s*:\s*\n\s*case\s+GGML_TYPE_F32',
                src,
            )),
            lambda src: re.sub(
                r'(case\s+GGML_TYPE_F16\s*:\s*\n)(\s*case\s+GGML_TYPE_F32\s*:\s*\n\s*case\s+GGML_TYPE_Q4_0)',
                r'\1                    case GGML_TYPE_BF16:\n\2',
                src,
                count=1,
            ),
        ),
    ])


def patch_getrows_cpp(sycl_dir):
    """Patch getrows.cpp: add BF16 dispatch for GET_ROWS.

    Upstream PR: https://github.com/ggml-org/llama.cpp/pull/21391
    Adds a GGML_TYPE_BF16 case using sycl::ext::oneapi::bfloat16,
    matching the existing sycl::half (F16) pattern.
    """
    path = os.path.join(sycl_dir, "getrows.cpp")
    bf16_case = (
        "        case GGML_TYPE_BF16:\n"
        "            get_rows_sycl_float(ctx, dst->src[0], dst->src[1], dst, "
        "(const sycl::ext::oneapi::bfloat16 *)dst->src[0]->data,\n"
        "                                src1_i32, (float *)dst->data, ctx.stream());\n"
        "            break;\n"
    )
    return patch_file(path, [
        (
            "BF16 dispatch added to GET_ROWS",
            lambda src: "GGML_TYPE_BF16" not in src
                        and "sycl::half" in src,
            lambda src: re.sub(
                r'(case\s+GGML_TYPE_F16\s*:\s*\n'
                r'\s*get_rows_sycl_float\([^;]+sycl::half[^;]+;\s*\n'
                r'\s*break;\s*\n)',
                r'\1' + bf16_case,
                src,
                count=1,
            ),
        ),
    ])


def main():
    target = sys.argv[1]

    # Support legacy single-file invocation (backward compat)
    if target.endswith(".cpp"):
        sycl_dir = os.path.dirname(target)
    else:
        sycl_dir = target

    all_applied = []
    all_applied += patch_ggml_sycl_cpp(sycl_dir)
    all_applied += patch_getrows_cpp(sycl_dir)

    if all_applied:
        for name in all_applied:
            print(f"  [OK] {name}")
        print(f"Patched {len(all_applied)} item(s) in {sycl_dir}")
    else:
        print(f"No patches needed for {sycl_dir}")


if __name__ == "__main__":
    main()
