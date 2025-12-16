from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

from extractors import collect_all_from_scratch
from train import split_autoverus_val
from verusynth.harness import DEFAULT_BASE_MODEL, run_annotation_loop_on_function


def _resolve_example_path(
    repositories_root: Path,
    meta: Dict[str, Any],
) -> Path:
    """
    Resolve the absolute path to the Rust file for a dataset example.

    Expected meta fields (as you set in the extractors):
      meta["data_source"]  folder name in repositories, e.g., "verus-proof-synthesis", "human-eval-verus", "verus"
      meta["file_path"]    path to the .rs file within that repository
      meta["fn_name"]      name of the function in that file
    """
    data_source = meta["data_source"]
    rel_path = meta["file_path"]
    return repositories_root / data_source / rel_path


def eval_verusynth_on_test(
    base_model: Optional[str] = None,
    adapter_path: Optional[str] = None,
    repositories_root: Path = Path("repositories"),
    max_iterations: int = 5,
    max_new_tokens: int = 1024,
) -> Dict[str, Any]:
    """
    Run the VeruSynth harness on all examples in the test/validation split.

    Returns a dict with simple aggregate statistics.
    """
    if base_model is None:
        base_model = DEFAULT_BASE_MODEL
    
    repositories_root = repositories_root.resolve()

    # 1) Build full dataset and split train/val (test set == val_examples here)
    examples: List[Dict[str, Any]] = collect_all_from_scratch(repositories_root)
    _, val_examples = split_autoverus_val(examples)

    total = len(val_examples)
    success = 0
    failures: List[Dict[str, Any]] = []

    print(f"[eval] base_model={base_model}, adapter_path={adapter_path}")
    print(f"[eval] repositories_root={repositories_root}")
    print(f"[eval] test examples={total}")

    for idx, ex in enumerate(val_examples):
        meta = ex.get("meta", {})
        fn_name = meta["fn_name"]
        file_path = _resolve_example_path(repositories_root, meta)

        print(f"\n[eval {idx+1}/{total}] {meta.get('data_source')} :: {file_path} :: {fn_name}")

        ok = run_annotation_loop_on_function(
            file_path=str(file_path),
            fn_name=fn_name,
            max_iterations=max_iterations,
            base_model=base_model,
            adapter_path=adapter_path,
            max_new_tokens=max_new_tokens,
            ignore_existing=True,
        )

        if ok:
            print("    ✅ verified")
            success += 1
        else:
            print("    ❌ failed")
            failures.append(
                {
                    "meta": meta,
                    "file_path": str(file_path),
                    "fn_name": fn_name,
                }
            )

    print("\n====== VeruSynth evaluation summary ======")
    print(f"Total test examples:   {total}")
    print(f"Successfully verified: {success}")
    print(f"Failed:                {total - success}")

    return {
        "total": total,
        "success": success,
        "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate VeruSynth on all test examples."
    )
    parser.add_argument(
        "--base-model",
        type=str,
        help='HuggingFace base model, e.g. "Qwen/Qwen3-8B"',
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help=(
            "Optional fine-tune name; if omitted, verusynth.harness will "
            "auto-select the latest adapter for this base model."
        ),
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum VeruSynth refinement iterations per function.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum number of new tokens to sample per LLM call.",
    )

    args = parser.parse_args()

    eval_verusynth_on_test(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        repositories_root=Path("repositories"),
        max_iterations=args.max_iterations,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
