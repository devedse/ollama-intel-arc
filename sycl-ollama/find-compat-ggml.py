#!/usr/bin/env python3
"""
Find a llama.cpp commit whose ggml headers (ggml.h, ggml-common.h) are
byte-identical to the copies vendored inside an ollama source tree.

Usage:  python3 find-compat-ggml.py /path/to/ollama
Prints the compatible commit SHA to stdout.
"""

import os
import subprocess
import sys

SEARCH_DIR = "/tmp/llama-compat-search"

# Map: llama.cpp path → relative path inside ollama source
HEADER_MAP = {
    "ggml/include/ggml.h": "ml/backend/ggml/ggml/include/ggml.h",
    "ggml/src/ggml-common.h": "ml/backend/ggml/ggml/src/ggml-common.h",
}


def die(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def git(*args, cwd=None):
    return (
        subprocess.check_output(
            ["git"] + list(args), cwd=cwd, stderr=subprocess.DEVNULL
        )
        .decode()
        .strip()
    )


def blob_hash(filepath):
    """Compute the git blob hash of a file (same as `git hash-object`)."""
    return git("hash-object", filepath)


def ls_tree_blob(repo, commit, path):
    """Get the blob hash of a file at a specific commit."""
    try:
        line = git("-C", repo, "ls-tree", commit, "--", path)
        return line.split()[2] if line else None
    except (subprocess.CalledProcessError, IndexError):
        return None


def main():
    if len(sys.argv) < 2:
        die("Usage: find-compat-ggml.py /path/to/ollama")

    ollama_dir = sys.argv[1]

    # Compute target blob hashes from ollama's vendored headers
    targets = {}
    for llama_path, rel in HEADER_MAP.items():
        full = os.path.join(ollama_dir, rel)
        if not os.path.isfile(full):
            print(f"  skip {rel} (not found)", file=sys.stderr)
            continue
        targets[llama_path] = blob_hash(full)
        print(f"  {llama_path} -> {targets[llama_path]}", file=sys.stderr)

    if not targets:
        die("No ggml headers found in ollama source")

    # Clone llama.cpp (blobless = fast, only tree/commit objects)
    print("Cloning llama.cpp tree history...", file=sys.stderr)
    subprocess.check_call(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            "https://github.com/ggml-org/llama.cpp.git",
            SEARCH_DIR,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # List recent commits that touched any of the target headers
    commits_out = git(
        "-C",
        SEARCH_DIR,
        "log",
        "--format=%H",
        "-500",
        "--",
        *targets.keys(),
    )
    commits = [c for c in commits_out.split("\n") if c]
    print(f"Searching {len(commits)} commits...", file=sys.stderr)

    for commit in commits:
        if all(
            ls_tree_blob(SEARCH_DIR, commit, p) == h for p, h in targets.items()
        ):
            print(f"  Compatible commit: {commit}", file=sys.stderr)
            subprocess.call(["rm", "-rf", SEARCH_DIR])
            # Print just the SHA to stdout (captured by caller)
            print(commit)
            sys.exit(0)

    subprocess.call(["rm", "-rf", SEARCH_DIR])
    die(f"No compatible commit found in last {len(commits)} commits")


if __name__ == "__main__":
    main()
