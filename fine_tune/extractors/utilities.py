import re
from typing import Dict, List, Any, Optional, Tuple
from verusynth.annotate import build_annotations_from_sites, strip_verus_annotations_and_collect
from verusynth.annotate.detect import detect_sites_from_comments
from verusynth.annotate.insert import insert_site_comments
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


def make_from_scratch_example(func_src: str) -> Optional[Dict[str, Any]]:
    """
    Given a Verus-annotated function, build a from-scratch example.
    """
    
    # 1) Strip existing Verus annotations, but keep a structured record of them.
    extracted = strip_verus_annotations_and_collect(func_src)
    base_func_src = extracted.base_func_src

    # 2) Insert site comments (choosing Lk / Ak as we go).
    func_with_sites = insert_site_comments(base_func_src)
    print(func_with_sites)

    # 3) Detect sites from the comments in the augmented function.
    sites = detect_sites_from_comments(func_with_sites)

    # 4) Map the extracted annotations onto the nearest / most likely sites.
    #    This uses whatever heuristics you've encoded in build_annotations_from_sites
    #    (line proximity, basic block structure, etc.).
    annotations_json = build_annotations_from_sites(
        sites=sites,
        extracted=extracted,
    )

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
