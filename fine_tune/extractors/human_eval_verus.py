from pathlib import Path
from typing import Any, Dict, List

from extractors.utilities import list_functions_in_text, make_from_scratch_example


def iter_humaneval_verus_task_files(human_eval_root: Path) -> List[Path]:
    """
    Return all .rs files under:

        <human_eval_root>/tasks/**/*.rs
    """
    tasks_root = human_eval_root / "tasks"
    if not tasks_root.is_dir():
        return []

    return sorted(tasks_root.rglob("*.rs"))


def collect_humaneval_verus_from_scratch(
    human_eval_root: Path,
    min_annotations: int = 1,
) -> List[Dict[str, Any]]:
    """
    Return a list of from-scratch examples extracted from human-eval-verus's
    exec functions under tasks/.
    """
    human_eval_root = human_eval_root.resolve()
    rs_files = iter_humaneval_verus_task_files(human_eval_root)

    dataset: List[Dict[str, Any]] = []

    for rs_path in rs_files:
        text = rs_path.read_text(encoding="utf8")
        lines = text.splitlines(keepends=True)
        fn_spans = list_functions_in_text(text)

        for fn_name, s, e in fn_spans:
            func_src = "".join(lines[s:e+1])

            try:
                example = make_from_scratch_example(func_src)
            except Exception:
                continue

            if len(example["output"]["annotations"]) < min_annotations:
                continue

            example.setdefault("meta", {})
            example["meta"].update({
                "data_source": "human-eval-verus",
                "file_path": str(rs_path.relative_to(human_eval_root)),
                "fn_name": fn_name,
            })

            dataset.append(example)

    return dataset

