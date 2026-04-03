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
    4. Add BF16 kernel + wrapper function    (upstream bug, PR #21391)
    5. Add BF16 dispatch case for GET_ROWS   (upstream bug, PR #21391)

As of ollama v0.16.1+, patches 1-2 are no longer needed (APIs converged).
Patches 3-5 are needed until upstream llama.cpp PR #21391 is merged.

Note: The BF16 GET_ROWS implementation uses explicit bit-level conversion
(uint16 << 16 → float32) instead of sycl::ext::oneapi::bfloat16's implicit
operator float(), which may produce garbled output on some Intel GPU
architectures (observed on Battlemage / Arc Pro B50).
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


def _apply_bf16_dispatch(src, bf16_dispatch):
    """Add new-style BF16 dispatch, replacing old sycl::ext::oneapi style if present."""
    old_pattern = (
        r'        case GGML_TYPE_BF16\s*:\s*\n'
        r'\s*get_rows_sycl_float\([^;]+sycl::ext::oneapi::bfloat16[^;]+;\s*\n'
        r'\s*break;\s*\n'
    )
    if re.search(old_pattern, src):
        return re.sub(old_pattern, bf16_dispatch, src, count=1)
    return re.sub(
        r'(case\s+GGML_TYPE_F16\s*:\s*\n'
        r'\s*get_rows_sycl_float\([^;]+sycl::half[^;]+;\s*\n'
        r'\s*break;\s*\n)',
        r'\1' + bf16_dispatch,
        src,
        count=1,
    )


def patch_getrows_cpp(sycl_dir):
    """Patch getrows.cpp: add BF16 GET_ROWS with explicit bit-level conversion.

    Upstream PR: https://github.com/ggml-org/llama.cpp/pull/21391

    Instead of using sycl::ext::oneapi::bfloat16 with implicit operator float()
    (which may produce garbled output on some Intel GPU architectures), this
    patch injects a dedicated kernel that converts BF16→F32 via bit shifting:
      float = union{ uint32_t u = bf16_bits << 16; }.f
    This is the same approach ggml uses in ggml_compute_bf16_to_fp32().
    """
    path = os.path.join(sycl_dir, "getrows.cpp")

    # Standalone BF16→F32 kernel and wrapper function.
    # Injected before ggml_sycl_op_get_rows in the file.
    bf16_functions = (
        "\n"
        "// BF16 -> F32 GET_ROWS: explicit bit-level conversion.\n"
        "// Uses manual bit shifting instead of sycl::ext::oneapi::bfloat16\n"
        "// to ensure correct BF16->F32 conversion on all GPU architectures.\n"
        "static void k_get_rows_bf16_f32(\n"
        "            const void * src0, const int32_t * src1, float * dst,\n"
        "            int64_t ne00, int64_t ne12,\n"
        "            size_t s1, size_t s2, size_t s3,\n"
        "            size_t nb01, size_t nb02, size_t nb03,\n"
        "            size_t s10, size_t s11, size_t s12,\n"
        "            const sycl::nd_item<3> &item_ct1) {\n"
        "\n"
        "    const int i00 = item_ct1.get_group(2) * item_ct1.get_local_range(2) +\n"
        "                    item_ct1.get_local_id(2);\n"
        "    const int i10 = item_ct1.get_local_range(1) * item_ct1.get_group(1) +\n"
        "                    item_ct1.get_local_id(1);\n"
        "    const int i11 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) +\n"
        "                     item_ct1.get_local_id(0)) /\n"
        "                    ne12;\n"
        "    const int i12 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) +\n"
        "                     item_ct1.get_local_id(0)) %\n"
        "                    ne12;\n"
        "\n"
        "    if (i00 >= ne00) {\n"
        "        return;\n"
        "    }\n"
        "\n"
        "    const int i01 = src1[i10*s10 + i11*s11 + i12*s12];\n"
        "\n"
        "    float * dst_row = dst + i10*s1 + i11*s2 + i12*s3;\n"
        "    const uint16_t * src0_row = (const uint16_t *)((const char *)src0 + i01*nb01 + i11*nb02 + i12*nb03);\n"
        "\n"
        "    // BF16 to FP32: BF16 occupies the upper 16 bits of float32\n"
        "    union { uint32_t u; float f; } conv;\n"
        "    conv.u = ((uint32_t)src0_row[i00]) << 16;\n"
        "    dst_row[i00] = conv.f;\n"
        "}\n"
        "\n"
        "static void get_rows_sycl_bf16(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,\n"
        "                                const ggml_tensor *src1, ggml_tensor *dst,\n"
        "                                const void *src0_dd, const int32_t *src1_dd,\n"
        "                                float *dst_dd, queue_ptr stream) {\n"
        "\n"
        "    GGML_TENSOR_BINARY_OP_LOCALS\n"
        "\n"
        "    const sycl::range<3> block_dims(1, 1, SYCL_GET_ROWS_BLOCK_SIZE);\n"
        "    const int block_num_x = (ne00 + SYCL_GET_ROWS_BLOCK_SIZE - 1) / SYCL_GET_ROWS_BLOCK_SIZE;\n"
        "    const sycl::range<3> block_nums(ne11 * ne12, ne10, block_num_x);\n"
        "\n"
        "    const size_t s1 = nb1 / ggml_element_size(dst);\n"
        "    const size_t s2 = nb2 / ggml_element_size(dst);\n"
        "    const size_t s3 = nb3 / ggml_element_size(dst);\n"
        "\n"
        "    const size_t s10 = nb10 / ggml_element_size(src1);\n"
        "    const size_t s11 = nb11 / ggml_element_size(src1);\n"
        "    const size_t s12 = nb12 / ggml_element_size(src1);\n"
        "\n"
        "    stream->parallel_for(\n"
        "        sycl::nd_range<3>(block_nums * block_dims, block_dims),\n"
        "        [=](sycl::nd_item<3> item_ct1) {\n"
        "            k_get_rows_bf16_f32(src0_dd, src1_dd, dst_dd, ne00, ne12, s1, s2,\n"
        "                                s3, nb01, nb02, nb03, s10, s11, s12, item_ct1);\n"
        "        });\n"
        "\n"
        "    GGML_UNUSED(dst);\n"
        "    GGML_UNUSED(ctx);\n"
        "}\n"
        "\n"
    )

    bf16_dispatch = (
        "        case GGML_TYPE_BF16:\n"
        "            get_rows_sycl_bf16(ctx, dst->src[0], dst->src[1], dst, dst->src[0]->data,\n"
        "                                src1_i32, (float *)dst->data, ctx.stream());\n"
        "            break;\n"
    )

    return patch_file(path, [
        # 4. Insert BF16 kernel + wrapper before ggml_sycl_op_get_rows
        (
            "BF16->F32 kernel added",
            lambda src: "k_get_rows_bf16_f32" not in src,
            lambda src: re.sub(
                r'(void ggml_sycl_op_get_rows\b)',
                bf16_functions + r'\1',
                src,
                count=1,
            ),
        ),
        # 5. Add BF16 dispatch case (or replace old sycl::ext::oneapi style)
        (
            "BF16 dispatch in GET_ROWS",
            lambda src: "get_rows_sycl_bf16(ctx" not in src,
            lambda src: _apply_bf16_dispatch(src, bf16_dispatch),
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
