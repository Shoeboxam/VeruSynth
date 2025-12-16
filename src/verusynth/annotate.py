from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import re


def _compute_offsets(lines: List[str]) -> List[int]:
    offsets, pos = [], 0
    for l in lines:
        offsets.append(pos)
        pos += len(l)
    return offsets


def _char_to_line(offsets: List[int], lines: List[str], idx: int) -> int:
    lo, hi = 0, len(offsets) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        start = offsets[mid]
        end = start + len(lines[mid])
        if start <= idx < end:
            return mid
        if idx < start:
            hi = mid - 1
        else:
            lo = mid + 1
    return max(0, min(len(lines) - 1, lo))


def _find_basic_blocks(
    func_src: str,
) -> Tuple[List[Dict[str, int]], Optional[int], Optional[int]]:
    """
    Heuristically identify basic blocks [start_line, end_line] (1-based)
    in a Rust function body, and also return the outer function body range.

    Returns:
        (blocks, body_start_line, body_end_line)

        - blocks: list of dicts with:
            {
              "start_line": <int>,  # 1-based
              "end_line":   <int>,  # 1-based
            }

        - body_start_line: 1-based line number of the first line *inside*
          the outermost `{ ... }` of the function body, or None if not found.

        - body_end_line: 1-based line number of the last line *inside* the
          outermost `{ ... }` of the function body, or None if not found.
    """
    text = func_src
    lines = text.splitlines(keepends=True)
    n = len(lines)
    offsets = _compute_offsets(lines)

    # --- Find function body (between outermost braces after `fn`) ---
    body_start_idx, body_end_idx = 0, n - 1  # indices into `lines`
    body_start_line: Optional[int] = None  # 1-based
    body_end_line: Optional[int] = None  # 1-based

    fn_match = re.search(r"\bfn\b", text)
    if fn_match:
        open_idx = text.find("{", fn_match.end())
        if open_idx != -1:
            depth, i, close_idx = 1, open_idx + 1, None
            while i < len(text) and depth > 0:
                c = text[i]
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        close_idx = i
                        break
                i += 1
            if close_idx is not None:
                s_idx = _char_to_line(offsets, lines, open_idx) + 1  # line after `{`
                e_idx = _char_to_line(offsets, lines, close_idx) - 1  # line before `}`
                if 0 <= s_idx <= e_idx < n:
                    body_start_idx, body_end_idx = s_idx, e_idx
                    body_start_line = body_start_idx + 1
                    body_end_line = body_end_idx + 1

    branch_kw = re.compile(r"\b(if|while|for|loop|match)\b")
    jump_kw = re.compile(r"\b(return|break|continue)\b")

    blocks: List[Dict[str, int]] = []
    current_start: Optional[int] = None  # 0-based index in `lines`

    def is_closing_brace_only(line: str) -> bool:
        s = line.strip()
        return s == "}" or s == "};"

    for idx in range(body_start_idx, body_end_idx + 1):
        line = lines[idx]
        stripped = line.strip()
        is_comment = stripped.startswith("//")

        if current_start is None:
            current_start = idx

        end_block = False
        if not is_comment:
            if (
                branch_kw.search(line)
                or jump_kw.search(line)
                or is_closing_brace_only(line)
            ):
                end_block = True

        if end_block or idx == body_end_idx:
            blocks.append(
                {
                    "start_line": current_start + 1,  # 1-based
                    "end_line": idx + 1,
                }
            )
            current_start = None

    return blocks, body_start_line, body_end_line


def extract_site_locations(func_src: str) -> List[Dict[str, Any]]:
    """
    Compute site locations for a Rust/Verus function *without* modifying it.

    Returns a list of site dicts, each of the form:

        {
          "id":   "L0" or "A0",
          "kind": "loop_site" or "assert_site",
          "line": <int>,  # 1-based, relative to the base function
        }

    Semantics:
      - For loop sites ("loop_site"):
          * "line" is the line containing the `while`, `for`, or `loop`
            keyword in the base function (non-comment line).

      - For assert sites ("assert_site"):
          * "line" is the line AFTER which the assertion logically sits in
            the base function (end of a basic block), but never the very
            end of the outer function body (to avoid implicit returns).
    """
    text = func_src
    lines = text.splitlines(keepends=True)
    offsets = _compute_offsets(lines)
    n = len(lines)

    sites: List[Dict[str, Any]] = []

    # --- Basic blocks + outer body range ---
    blocks, body_start_line, body_end_line = _find_basic_blocks(func_src)
    # body_start_line / body_end_line are expected 1-based indices into `lines`

    # Sanity clamp just in case
    if body_start_line is None:
        body_start_line = 1
    if body_end_line is None:
        body_end_line = n
    body_start_idx = max(0, min(n - 1, body_start_line - 1))
    body_end_idx = max(0, min(n - 1, body_end_line - 1))

    # -------------------------------------------------------------------------
    # 1) Loop sites (L#) on while/for/loop headers
    # -------------------------------------------------------------------------
    loop_pattern = re.compile(r"\b(while|for|loop)\b")
    loop_count = 0

    for m in loop_pattern.finditer(text):
        line_idx = _char_to_line(offsets, lines, m.start())  # 0-based
        if not (0 <= line_idx < n):
            continue

        # Restrict to lines inside the function body
        if line_idx < body_start_idx or line_idx > body_end_idx:
            continue

        # Skip pure comment lines
        stripped = lines[line_idx].lstrip()
        if stripped.startswith("//"):
            continue

        site_id = f"L{loop_count}"
        loop_count += 1
        sites.append(
            {
                "id": site_id,
                "kind": "loop_site",
                "line": line_idx + 1,  # 1-based
            }
        )

    # -------------------------------------------------------------------------
    # 2) Assert sites (A#) at reachable basic block ends,
    #    but not at the very end of the outer function body.
    # -------------------------------------------------------------------------
    jump_kw = re.compile(r"\b(return|break|continue)\b")

    a_idx = 0
    for bb in blocks:
        end_line = bb["end_line"]  # 1-based, same coordinate system as body_end_line
        end_idx = end_line - 1     # 0-based

        if not (0 <= end_idx < n):
            continue

        # Don't place sites at the very end of the outer body
        if body_end_line is not None and end_line == body_end_line:
            continue

        last_line = lines[end_idx]
        stripped = last_line.strip()

        # Skip blocks whose last non-comment line is an explicit jump
        # (we only look for jumps if the line isn't a comment-only line)
        if not stripped.startswith("//") and jump_kw.search(last_line):
            continue

        site_id = f"A{a_idx}"
        a_idx += 1

        sites.append(
            {
                "id": site_id,
                "kind": "assert_site",
                "line": end_line,   # "after this line" in the base function
            }
        )

    return sites


def merge_verus_annotations(
    base_func_src: str,
    sites: List[Dict[str, Any]],
    annotations_json: Dict[str, Any],
    emit_site_comments: bool = False,
) -> str:
    """
    Merge annotations into a *base* Rust/Verus function, using site locations.

    Inputs:
      - base_func_src:
          The original function source, without any site markers or annotations.

      - sites:
          List of site dicts from `extract_site_locations`, e.g.:
            {"id": "L0", "kind": "loop_site",   "line": 23}
            {"id": "A0", "kind": "assert_site", "line": 30}
          For assert sites, "line" means "insert after line <line>".

      - annotations_json:
          JSON-like object with the key "annotations", each entry like:
            {
              "site": "L0",
              "kind": "invariant" | "decreases" | "assert",
              "clauses": [...],   # for invariant/decreases
              "expr": "..."       # for asserts
            }

      - emit_site_comments:
          If True, emit comments of the form:
            - `// [LOOP SITE L0]` for loop sites
            - `// [ASSERT SITE A0]` for assert sites
          at the corresponding locations, regardless of whether there are
          annotations for that site.

    Semantics:
      - For each loop site L#:
          * Treat `line` as the loop header line containing `while` or `for`.
          * Insert:

                invariant
                    ...
                decreases
                    ...

            between the loop header and the brace, or move the brace to a
            new line. Site comments are attached to the header line if
            emit_site_comments is True.

      - For each assert site A#:
          * Treat `line` as "insert after this line".
          * Insert (optionally) a comment and zero or more `assert(...)`
            lines after it.
    """

    lines = base_func_src.splitlines(keepends=True)
    n_lines = len(lines)

    # ---- Deduplicate sites by ID (keep first occurrence) ----
    seen_ids: set[str] = set()
    dedup_sites: List[Dict[str, Any]] = []
    for s in sites:
        sid = s.get("id")
        if not sid or sid in seen_ids:
            continue
        seen_ids.add(sid)
        dedup_sites.append(s)

    loop_sites_by_line: Dict[int, List[str]] = defaultdict(list)   # 0-based -> [L#]
    assert_sites_by_line: Dict[int, List[str]] = defaultdict(list) # 0-based -> [A#]

    for s in dedup_sites:
        sid = s.get("id")
        kind = s.get("kind")
        line_no = s.get("line")  # 1-based
        if not sid or not kind or not isinstance(line_no, int):
            continue
        idx = max(0, min(n_lines - 1, line_no - 1))  # clamp
        if kind == "loop_site":
            loop_sites_by_line[idx].append(sid)
        elif kind == "assert_site":
            assert_sites_by_line[idx].append(sid)

    # ---- Build annotation map: site_id -> list[annotation dicts] ----
    anno_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for a in annotations_json.get("annotations", []):
        if isinstance(a, dict):
            site_id = a.get("site")
            if site_id:
                anno_map[site_id].append(a)
        else:
            # Pydantic or similar object
            if hasattr(a, "site"):
                site_id = a.site
                if hasattr(a, "model_dump"):
                    anno_map[site_id].append(a.model_dump())
                elif hasattr(a, "dict"):
                    anno_map[site_id].append(a.dict())

    out_lines: List[str] = []

    def _leading_ws_local(s: str) -> str:
        m = re.match(r"\s*", s)
        return m.group(0) if m else ""

    for i in range(n_lines):
        line = lines[i]

        # ---- Loop site on this line? ----
        loop_site_ids = loop_sites_by_line.get(i)
        if loop_site_ids:
            # Assume at most one loop header per line; use the first.
            site_id = loop_site_ids[0]
            entries = anno_map.get(site_id, [])

            invariant_clauses: List[str] = []
            decreases_clauses: List[str] = []

            for entry in entries:
                kind = entry.get("kind")
                clauses = entry.get("clauses") or []
                if not isinstance(clauses, list):
                    continue
                if kind == "invariant":
                    invariant_clauses.extend(
                        [c for c in clauses if isinstance(c, str) and c.strip()]
                    )
                elif kind == "decreases":
                    decreases_clauses.extend(
                        [c for c in clauses if isinstance(c, str) and c.strip()]
                    )

            raw = line.rstrip("\n")

            # If there are no annotations AND we are not emitting comments,
            # just keep the original line as-is.
            if (
                not emit_site_comments
                and not invariant_clauses
                and not decreases_clauses
            ):
                out_lines.append(line)
                continue

            # Split header into "core" and optional brace part.
            brace_idx = raw.find("{")
            if brace_idx != -1:
                header_core = raw[:brace_idx].rstrip()
                brace_text = raw[brace_idx:].strip() or "{"
            else:
                header_core = raw.rstrip()
                brace_text = "{"

            indent = _leading_ws_local(header_core or raw)

            # Header line: always end with '\n' after rewriting, to keep
            # invariant/decreases on their own lines.
            if emit_site_comments:
                header_line_out = f"{header_core}  // [LOOP SITE {site_id}]\n"
            else:
                header_line_out = header_core + "\n"
            out_lines.append(header_line_out)

            # Invariant block
            if invariant_clauses:
                out_lines.append(indent + "    invariant\n")
                for c in invariant_clauses:
                    out_lines.append(indent + f"        {c},\n")

            # Decreases block
            if decreases_clauses:
                out_lines.append(indent + "    decreases\n")
                for c in decreases_clauses:
                    out_lines.append(indent + f"        {c},\n")

            # Brace line, if there was one on the original header
            if brace_idx != -1:
                out_lines.append(indent + brace_text + "\n")

            continue

        # ---- Normal line (not a loop header) ----
        out_lines.append(line)

        # ---- After this line, do we have any assert sites? ----
        assert_site_ids = assert_sites_by_line.get(i)
        if assert_site_ids:
            indent_above = _leading_ws_local(lines[i])
            indent_below = _leading_ws_local(lines[i + 1]) if i + 1 < n_lines else ""
            indent = indent_above if len(indent_above) >= len(indent_below) else indent_below

            for site_id in assert_site_ids:
                entries = anno_map.get(site_id, [])

                # Collect all assert entries with non-empty expr
                assert_exprs: List[str] = []
                for entry in entries:
                    if entry.get("kind") != "assert":
                        continue
                    expr = (entry.get("expr") or "").strip()
                    if expr:
                        assert_exprs.append(expr)

                # Always emit the site comment if requested, even if there
                # are zero asserts for this site.
                if emit_site_comments:
                    out_lines.append(f"{indent}// [ASSERT SITE {site_id}]\n")

                # Emit all assertions for this site (may be zero)
                for expr in assert_exprs:
                    out_lines.append(f"{indent}assert({expr});\n")

    return "".join(out_lines)


def remap_sites_to_annotated_function(
    annotated_func_src: str,
    sites: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Given:
      - annotated_func_src: function *with* site comments like
           `// [LOOP SITE L0]`, `// [ASSERT SITE A0]`
      - sites: original site list (with ids/kinds, line numbers relative
        to the base function)

    Return a new list of sites where 'line' has been updated to match
    the *annotated* function's line numbers (1-based).

    If we can't find a marker for a site id, we keep its original line.
    """
    lines = annotated_func_src.splitlines(keepends=True)

    # Build an id -> (kind, line) map based on the comments actually present
    loop_pat = re.compile(r'//\s*\[LOOP SITE (L\d+)\]')
    assert_pat = re.compile(r'//\s*\[ASSERT SITE (A\d+)\]')

    marker_positions: Dict[str, int] = {}  # site_id -> 1-based line

    for idx, line in enumerate(lines):
        m_loop = loop_pat.search(line)
        if m_loop:
            site_id = m_loop.group(1)
            marker_positions[site_id] = idx + 1
            continue
        m_assert = assert_pat.search(line)
        if m_assert:
            site_id = m_assert.group(1)
            marker_positions[site_id] = idx + 1
            continue

    # Produce updated sites list
    new_sites: List[Dict[str, Any]] = []
    for s in sites:
        sid = s.get("id")
        if not sid:
            new_sites.append(dict(s))
            continue

        new_s = dict(s)
        if sid in marker_positions:
            new_s["line"] = marker_positions[sid]
        new_sites.append(new_s)

    return new_sites


@dataclass
class ExtractedAnnotations:
    """
    Carries a base function without annotations, as well as loop/assert annotations.

    Line numbers are all in the coordinate system of the stripped base
    function (because we maintain a base_line_no counter as we build it).
    """

    base_func_src: str
    # base_header_line -> {"invariant": [...], "decreases": [...]}
    loop_annos_by_line: Dict[int, Dict[str, List[str]]]
    # base_line -> [expr, ...]
    assert_annos_by_line: Dict[int, List[str]]


def strip_verus_annotations_and_collect(func_src: str) -> ExtractedAnnotations:
    lines = func_src.splitlines(keepends=True)
    n = len(lines)

    loop_header_re = re.compile(r"\b(while|for)\b")
    invariant_re = re.compile(r"^\s*invariant\b")
    decreases_re = re.compile(r"^\s*decreases\b")
    assert_re = re.compile(r"^\s*assert\s*\(")

    base_lines: List[str] = []
    base_line_no = 0

    loop_annos_by_line: Dict[int, Dict[str, List[str]]] = defaultdict(
        lambda: {"invariant": [], "decreases": []}
    )
    assert_annos_by_line: Dict[int, List[str]] = defaultdict(list)

    i = 0
    while i < n:
        line = lines[i]

        # Loop header
        if loop_header_re.search(line):
            base_lines.append(line)
            base_line_no += 1
            header_base_line = base_line_no
            i += 1

            # Consume following invariant/decreases blocks without adding
            # them to base_lines.
            while i < n:
                s = lines[i].strip()

                if not s:
                    base_lines.append(lines[i])
                    base_line_no += 1
                    i += 1
                    break

                if invariant_re.match(lines[i]):
                    i += 1
                    while i < n:
                        s2 = lines[i].strip()
                        if not s2:
                            break
                        if (
                            invariant_re.match(lines[i])
                            or decreases_re.match(lines[i])
                            or loop_header_re.search(lines[i])
                        ):
                            break
                        if s2.endswith(","):
                            clause = s2.rstrip(",").strip()
                            if clause:
                                loop_annos_by_line[header_base_line][
                                    "invariant"
                                ].append(clause)
                            i += 1
                        else:
                            clause = s2.strip()
                            if clause:
                                loop_annos_by_line[header_base_line][
                                    "invariant"
                                ].append(clause)
                            i += 1
                            break
                    continue

                if decreases_re.match(lines[i]):
                    i += 1
                    while i < n:
                        s2 = lines[i].strip()
                        if not s2:
                            break
                        if (
                            invariant_re.match(lines[i])
                            or decreases_re.match(lines[i])
                            or loop_header_re.search(lines[i])
                        ):
                            break
                        if s2.endswith(","):
                            clause = s2.rstrip(",").strip()
                            if clause:
                                loop_annos_by_line[header_base_line][
                                    "decreases"
                                ].append(clause)
                            i += 1
                        else:
                            clause = s2.strip()
                            if clause:
                                loop_annos_by_line[header_base_line][
                                    "decreases"
                                ].append(clause)
                            i += 1
                            break
                    continue

                break

            continue

        # Assert line
        if assert_re.match(line):
            if base_line_no > 0:
                expr = line.strip()
                inner = re.sub(r"^\s*assert\s*\(", "", expr)
                inner = re.sub(r"\)\s*;?\s*$", "", inner).strip()
                if inner:
                    assert_annos_by_line[base_line_no].append(inner)
            i += 1
            continue

        # Normal line
        base_lines.append(line)
        base_line_no += 1
        i += 1

    base_func_src = "".join(base_lines)

    return ExtractedAnnotations(
        base_func_src=base_func_src,
        loop_annos_by_line=dict(loop_annos_by_line),
        assert_annos_by_line=dict(assert_annos_by_line),
    )


def build_annotations_from_sites(
    sites: List[Dict[str, Any]],
    extracted: ExtractedAnnotations,
) -> Dict[str, Any]:
    """
    Build {"annotations": [...]} using:

      - sites:
          List of site dicts from extract_site_locations(base_func_src), each like:
            {"id": "L0", "kind": "loop_site",   "line": <int>}
            {"id": "A0", "kind": "assert_site", "line": <int>}

      - extracted: ExtractedAnnotations
          Dataclass returned by strip_verus_annotations_and_collect, holding:
            * base_func_src
            * loop_annos_by_line: base_header_line -> {"invariant": [...], "decreases": [...]}
            * assert_annos_by_line: base_line -> [expr, ...]

    Line numbers in `sites` and `extracted.*_by_line` are all in the coordinate
    system of the stripped base function.
    """
    annotations: List[Dict[str, Any]] = []

    loop_annos_by_line = extracted.loop_annos_by_line
    assert_annos_by_line = extracted.assert_annos_by_line

    for s in sites:
        sid = s["id"]
        kind = s["kind"]
        line = s["line"]  # 1-based, base coordinate

        if kind == "loop_site":
            info = loop_annos_by_line.get(line)
            if not info:
                continue
            inv = info.get("invariant") or []
            dec = info.get("decreases") or []

            if inv:
                annotations.append(
                    {
                        "site": sid,
                        "kind": "invariant",
                        "clauses": inv,
                    }
                )
            if dec:
                annotations.append(
                    {
                        "site": sid,
                        "kind": "decreases",
                        "clauses": dec,
                    }
                )

        elif kind == "assert_site":
            exprs = assert_annos_by_line.get(line, [])
            for expr in exprs:
                annotations.append(
                    {
                        "site": sid,
                        "kind": "assert",
                        "expr": expr,
                    }
                )

    return {"annotations": annotations}
