import re
from collections import defaultdict
from typing import Dict, List, Any, Tuple
from verusynth.annotate import build_annotations_from_sites, extract_site_locations, strip_verus_annotations_and_collect
from dataclasses import dataclass
from verusynth.rust_io import find_function_span



# Match `fn name(` with optional pub/async/unsafe
_FN_HEADER_RE = re.compile(
    r"^\s*(pub\s+)?(async\s+)?(unsafe\s+)?fn\s+([A-Za-z0-9_]+)\s*\(",
)


def list_functions_in_text(text: str) -> List[Tuple[str, int, int]]:
    """
    Return a list of (fn_name, start_idx, end_idx) for each function in `text`.

    - start_idx, end_idx are 0-based *line indices* (inclusive).
    - Uses the existing `find_function_span` helper to locate full function bodies,
      including doc comments and attributes above the header.

    We also deduplicate by function name to avoid re-processing the same function
    multiple times if its header is matched more than once.
    """
    lines = text.splitlines(keepends=True)
    results: List[Tuple[str, int, int]] = []
    seen: set[str] = set()

    for line in lines:
        m = _FN_HEADER_RE.match(line)
        if not m:
            continue

        fn_name = m.group(4)
        if fn_name in seen:
            continue

        span = find_function_span(text, lines, fn_name)
        if span is None:
            continue

        start_idx, end_idx = span
        results.append((fn_name, start_idx, end_idx))
        seen.add(fn_name)

    return results


def make_from_scratch_example(func_src: str) -> Dict[str, Any]:
    """
    Given a Verus-annotated function, build a from-scratch example.
    """
    extracted = strip_verus_annotations_and_collect(func_src)

    # Sites are computed on the stripped base function
    sites = extract_site_locations(extracted.base_func_src)

    annotations_json = build_annotations_from_sites(
        sites=sites,
        extracted=extracted,
    )

    if not annotations_json["annotations"]:
        raise ValueError("No annotations found for this function")

    return {
        "input": {
            "base_func_src": extracted.base_func_src,
            "sites": sites,
            "current_annotations": {"annotations": []},
            "latest_errors": "",
            "error_summary": "",
        },
        "output": annotations_json,
    }
