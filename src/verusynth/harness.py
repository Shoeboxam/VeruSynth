import os
from pathlib import Path
import shutil
from typing import Dict, Any, Optional

from verusynth.annotate.detect import detect_sites_from_comments
from verusynth.annotate.extract import build_annotations_from_sites, strip_verus_annotations_and_collect
from verusynth.annotate.insert import insert_site_comments
from verusynth.annotate.merge import merge_annotations_from_comments
from verusynth.llm import (
    build_annotation_prompt,
    call_annotator_model,
    load_model,
    summarize_verus_errors,
)
from verusynth.rust_io import find_function_span, run_verus

DEFAULT_MAX_ITERATIONS = 20
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_BASE_MODEL = "Qwen/Qwen3-4B-Thinking-2507"


def resolve_finetune_adapter(base_model: str) -> Optional[str]:
    """
    Determine which LoRA adapter to use.

    Layout:

        src/verusynth/lora/<org>/<model>/<run_id>/

    Example for base_model="Qwen/Qwen1.5-4B-Chat":

        src/verusynth/lora/Qwen/Qwen1.5-4B-Chat/1/
        src/verusynth/lora/Qwen/Qwen1.5-4B-Chat/2/
        ...

    Rules:
      1. Look for numeric subdirectories under the model root and pick the one with the highest integer name.
      2. If nothing exists, return None (use base model only).
    """
    lora_root = Path(__file__).resolve().parent / "lora"

    # Derive model root from base_model, e.g. "Qwen/Qwen1.5-4B-Chat"
    parts = base_model.split("/")
    if len(parts) == 2:
        org, model = parts
        model_root = lora_root / org / model
    else:
        # Fallback: flatten into a single directory name
        model_root = lora_root / base_model.replace("/", "_")

    # pick highest-numbered run under model_root
    if not model_root.is_dir():
        return None

    numeric_runs: list[tuple[int, Path]] = []
    for child in model_root.iterdir():
        if not child.is_dir():
            continue
        try:
            run_id = int(child.name)
        except ValueError:
            continue
        numeric_runs.append((run_id, child))

    if not numeric_runs:
        return None

    # Pick the run with the largest integer name
    _, best_dir = max(numeric_runs, key=lambda x: x[0])
    return str(best_dir)


def run_annotation_loop_on_function(
    file_path: str,
    fn_name: str,
    base_model: str,
    adapter_path: Optional[str] = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    backup_suffix: str = ".bak",
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ignore_existing: bool = False,
) -> bool:
    """
    Verifier-in-the-loop harness:

      1. Read the Rust file and extract the target function (including attrs/docs).
      2. Strip existing Verus annotations to get a base function.
      3. Insert LOOP/ASSERT site comments into the base function.
      4. Initialize the annotation set (optionally from existing annotations).
      5. Iterate:
         a) Merge annotations onto the commented base function.
         b) Splice into file, run Verus.
         c) On failure, summarize errors and ask the LLM for new annotations.
      6. Restore backup if verification never succeeds.
    """

    # Determine adapter path under src/verusynth/lora/<name>/
    if not adapter_path:
        adapter_path = resolve_finetune_adapter(base_model)

    # --- Backup original file once ---
    backup_path = file_path + backup_suffix
    if not os.path.exists(backup_path):
        shutil.copy2(file_path, backup_path)

    # Read from backup (canonical original)
    with open(backup_path, "r", encoding="utf8") as f:
        original_text = f.read()
    original_lines = original_text.splitlines(keepends=True)

    span = find_function_span(original_text, original_lines, fn_name)
    if span is None:
        raise ValueError(f"Function `{fn_name}` not found in {file_path}")

    start_idx, end_idx = span
    raw_func_src = "".join(original_lines[start_idx : end_idx + 1])

    # Strip existing Verus annotations -> base function
    extracted = strip_verus_annotations_and_collect(raw_func_src)
    base_func_src = extracted.base_func_src

    # Insert LOOP/ASSERT site comments into the *base* function.
    # This chooses site IDs (L0, A0, …) on the fly.
    commented_base_func_src = insert_site_comments(base_func_src)

    # Detect sites for prompting / existing annotation reconstruction
    sites = detect_sites_from_comments(commented_base_func_src)

    # Persistent state
    annotations_json: Dict[str, Any] = {"annotations": []}
    latest_errors: str = ""
    error_history: list[str] = []
    error_summary: str = ""

    # Initialize annotation set from existing annotations unless ignored
    if not ignore_existing:
        annotations_json = build_annotations_from_sites(
            sites=sites,
            extracted=extracted,
        )

    # Load summarization model (always base model)
    summarization_model = load_model(base_model=base_model)

    # Load annotator model (base or fine-tuned)
    annotator_model = load_model(
        base_model=base_model,
        finetuned_adapter_path=adapter_path,
    )

    for iteration in range(max_iterations):
        try:
            print(f"[iteration {iteration}] verifying `{fn_name}` in {file_path}")

            # Merge current annotations into the commented base function.
            # This attaches invariants/decreases/asserts to the site comments.
            temp_func_src = merge_annotations_from_comments(
                func_src_with_comments=commented_base_func_src,
                annotations_json=annotations_json,
                emit_site_comments=True,  # keep markers visible for LLM
            )

            # Re-detect sites for the *current* annotated function (for prompt)
            sites_for_prompt = detect_sites_from_comments(temp_func_src)

            # Rewrite file with the temporary annotated function
            new_func_lines = temp_func_src.splitlines(keepends=True)
            new_file_lines = (
                original_lines[:start_idx]
                + new_func_lines
                + original_lines[end_idx + 1 :]
            )
            with open(file_path, "w", encoding="utf8") as f:
                f.write("".join(new_file_lines))

            # Run Verus
            ok, output = run_verus(file_path)
            print(f"[iteration {iteration}] verus success={ok}")

            if ok:
                # Verified successfully; keep the updated file
                return True

            # --- Verification failed: collect and summarize errors ---
            latest_errors = output or ""
            error_history.append(latest_errors)
            error_summary = summarize_verus_errors(error_history, summarization_model)

            # Build LLM prompt
            prompt = build_annotation_prompt(
                func_src=temp_func_src,
                sites=sites_for_prompt,
                current_annotations=annotations_json,
                latest_errors=latest_errors,
                error_summary=error_summary,
            )

            # Ask the annotator model for new annotations
            annotations_struct = call_annotator_model(
                prompt,
                annotator_model,
                max_new_tokens=max_new_tokens,
            )
            annotations_json = annotations_struct.model_dump()

        except Exception as e:
            # Don't abort the whole run on a single iteration failure;
            # just log and continue to the next attempt.
            print(f"[iteration {iteration}] exception: {e}")

    # --- Give up: restore file from backup ---
    print(f"[FAIL] Max iterations {max_iterations} reached. Restoring backup.")
    if os.path.exists(backup_path):
        shutil.copy2(backup_path, file_path)
        os.remove(backup_path)
    else:
        with open(file_path, "w", encoding="utf8") as f:
            f.write(original_text)
    return False


def main():
    """
    CLI entry point:

    verusynth <path/to/file.rs> <function_name>
                [--max-iterations N]
                [--base-model NAME]
                [--adapter-path NAME]
                [--max-new-tokens N]
                [--ignore-existing]
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Automatically synthesize Verus invariants/assertions using an LLM."
    )

    parser.add_argument("file_path", type=str, help="Path to the target .rs file")
    parser.add_argument("fn_name", type=str, help="Name of the function to annotate")

    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--ignore-existing", type=bool, default=False)

    args = parser.parse_args()

    ok = run_annotation_loop_on_function(
        file_path=args.file_path,
        fn_name=args.fn_name,
        max_iterations=args.max_iterations,
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        max_new_tokens=args.max_new_tokens,
        ignore_existing=args.ignore_existing,
    )

    if ok:
        print("✅ Verus succeeded with synthesized annotations.")
    else:
        print("❌ Verification failed; file restored from backup.")
