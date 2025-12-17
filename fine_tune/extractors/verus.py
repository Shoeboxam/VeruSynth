
from pathlib import Path
from typing import Any, Dict, List

from extractors.utilities import list_functions_in_text, make_from_scratch_example


def iter_verus_example_files(verus_root: Path) -> List[Path]:
    """
    Return all .rs files under:

        <verus_root>/examples/**/*.rs
    """
    examples_root = verus_root / "examples"
    if not examples_root.is_dir():
        return []
    return sorted(examples_root.rglob("*.rs"))


def collect_verus_examples_from_scratch(
    verus_root: Path,
    min_annotations: int = 1,
) -> List[Dict[str, Any]]:
    """
    Collect from-scratch annotation examples from the Verus repo's examples.

    - Traverses: <verus_root>/examples/**/*.rs
    - Uses list_functions_in_text, which only picks exec functions
      (plain `fn` or `exec fn`, not `spec/proof`).
    - For each function with at least `min_annotations` annotations, returns
      a JSON-style dict in a list.

    Each example has:
      - "input": base function + sites + empty annotations
      - "output": annotations
      - "meta": {
            "data_source": "verus",
            "file_path": "<path relative to verus_root>",
            "fn_name": "<function name>"
        }
    """
    verus_root = verus_root.resolve()
    rs_files = iter_verus_example_files(verus_root)

    dataset: List[Dict[str, Any]] = []

    for rs_path in rs_files:
        text = rs_path.read_text(encoding="utf8")
        lines = text.splitlines(keepends=True)
        fn_spans = list_functions_in_text(text)

        if not fn_spans:
            continue

        for fn_name, start_idx, end_idx in fn_spans:
            func_src = "".join(lines[start_idx : end_idx + 1])

            example = make_from_scratch_example(func_src)

            if len(example["output"]["annotations"]) < min_annotations:
                continue

            example.setdefault("meta", {})
            example["meta"].update(
                {
                    "data_source": "verus",
                    "file_path": str(rs_path.relative_to(verus_root)),
                    "fn_name": fn_name,
                }
            )

            dataset.append(example)

    return dataset