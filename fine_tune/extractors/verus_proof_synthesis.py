from pathlib import Path
from typing import List, Dict, Any

from extractors.utilities import list_functions_in_text, make_from_scratch_example

def iter_autoverus_verified_files(autoverus_root: Path) -> List[Path]:
    """
    Return all .rs files under:
        <autoverus_root>/benchmarks/**/verified/**/*.rs
    """
    bench_root = autoverus_root / "benchmarks"
    if not bench_root.is_dir():
        return []

    results = []
    for path in bench_root.rglob("*.rs"):
        if "verified" in path.relative_to(bench_root).parts:
            results.append(path)
    return sorted(results)


def collect_autoverus_from_scratch(
    autoverus_root: Path,
    min_annotations: int = 1,
) -> List[Dict[str, Any]]:
    """
    Return a list of from-scratch examples extracted from AutoVerus's
    verified benchmark tasks.
    """
    autoverus_root = autoverus_root.resolve()
    rs_files = iter_autoverus_verified_files(autoverus_root)

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
                "data_source": "verus-proof-synthesis",
                "file_path": str(rs_path.relative_to(autoverus_root)),
                "fn_name": fn_name,
            })

            dataset.append(example)

    return dataset
