from pathlib import Path
from typing import Any, Dict, List
from .human_eval_verus import collect_humaneval_verus_from_scratch
from .verus_proof_synthesis import collect_autoverus_from_scratch
from .verus import collect_verus_examples_from_scratch


def collect_all_from_scratch(
    repositories_root: Path,
    min_annotations: int = 1,
) -> List[Dict[str, Any]]:
    """
    Run all from-scratch extractors and return one combined dataset.
    """

    # Register each data source + extractor + subdirectory name.
    SOURCES = {
        "verus-proof-synthesis": collect_autoverus_from_scratch,
        "human-eval-verus": collect_humaneval_verus_from_scratch,
        "verus": collect_verus_examples_from_scratch,
    }

    repositories_root = repositories_root.resolve()
    dataset: List[Dict[str, Any]] = []

    for repo_name, extractor_fn in SOURCES.items():
        repo_root = repositories_root / repo_name
        if not repo_root.is_dir():
            print(f"[collect] {repo_name} is missing. Skipping.")
            continue

        print(f"[collect] Extracting from {repo_name} ...")

        extracted = extractor_fn(
            repo_root,
            min_annotations=min_annotations,
        )
        dataset.extend(extracted)

    return dataset
