from collections import defaultdict
from typing import List, Dict, Optional, Tuple
import re

import re
from typing import List


def _leading_ws(s: str) -> str:
    m = re.match(r"\s*", s)
    return m.group(0) if m else ""


def insert_loop_site_comments(func_src: str) -> str:
    """
    Insert loop-site comments into a Rust/Verus function by *choosing sites
    on the fly*.

    For each loop header (while/for/loop), assign a fresh site id Lk and
    insert:

        // [LOOP SITE Lk]

    between the loop condition and the opening brace, even for multi-line
    conditions.

    Examples:

        while i < n {
            ...
        }

    becomes:

        while i < n
            // [LOOP SITE L0]
            {
            ...
        }

        while (a
                   && b) {

    becomes:

        while (a
                   && b)
            // [LOOP SITE L1]
            {
    """
    lines = func_src.splitlines(keepends=True)
    n = len(lines)

    out_lines: List[str] = []
    loop_idx = 0

    # Helper: is this line a loop header candidate?
    loop_kw = re.compile(r"\b(while|for|loop)\b")

    i = 0
    while i < n:
        line = lines[i]
        # Strip off trailing comment for detection (avoid "while" in // comments)
        code_part = line.split("//", 1)[0]
        if loop_kw.search(code_part):
            # We treat this as the start of a loop header.
            site_id = f"L{loop_idx}"
            loop_idx += 1

            # Find the first '{' that belongs to this loop, starting at this line.
            brace_line_idx = None
            brace_col = None

            j = i
            while j < n:
                candidate = lines[j]
                idx = candidate.find("{")
                if idx != -1:
                    brace_line_idx = j
                    brace_col = idx
                    break
                j += 1

            if brace_line_idx is None:
                # No brace found at all after this header.
                # Fallback: just insert the loop-site comment *after the header line*.
                out_lines.append(line)
                indent = _leading_ws(line)
                out_lines.append(f"{indent}// [LOOP SITE {site_id}]\n")
                i += 1
                continue

            # Emit all lines from header up to (but not including) the brace line.
            for k in range(i, brace_line_idx):
                out_lines.append(lines[k])

            # Now process the brace line.
            brace_line = lines[brace_line_idx]
            brace_raw = brace_line.rstrip("\n")

            # Split at first '{'
            before = brace_raw[:brace_col].rstrip()
            after = brace_raw[brace_col:]  # includes '{' and anything after
            indent = _leading_ws(brace_raw)

            # If there is text before '{' (e.g., "while (a && b)"), keep it on its own line.
            if before:
                out_lines.append(before + "\n")

            # Insert loop-site comment between condition and brace
            out_lines.append(f"{indent}// [LOOP SITE {site_id}]\n")

            # Emit the brace (plus any trailing text), aligned with original indent
            after_stripped = after.lstrip()
            out_lines.append(f"{indent}{after_stripped}\n")

            # Continue from the line after the brace
            i = brace_line_idx + 1
            continue

        # Not a loop header: copy as-is
        out_lines.append(line)
        i += 1

    return "".join(out_lines)


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


def _leading_ws(s: str) -> str:
    m = re.match(r"\s*", s)
    return m.group(0) if m else ""


def _leading_ws(s: str) -> str:
    m = re.match(r"\s*", s)
    return m.group(0) if m else ""


def insert_assert_site_comments(func_src: str) -> str:
    """
    Insert assert-site comments into a Rust/Verus function by choosing sites
    on the fly using basic-block structure from `_find_basic_blocks`.

    For each basic block end that is:
      - not the very end of the outer function body,
      - not ending in an explicit `return`, `break`, or `continue`, and
      - not on a `while`/`for`/`loop` line,

    assign a fresh site id Ak and insert:

        // [ASSERT SITE Ak]

    *after* that line.

    Indentation for the inserted comment is the max of the indentation of
    the line before (the end line itself) and the line after it.
    """

    lines = func_src.splitlines(keepends=True)
    n = len(lines)

    # _find_basic_blocks is expected to return:
    #   (blocks, body_start_line, body_end_line)
    # where blocks is a list of dicts with "start_line"/"end_line" (1-based).
    blocks, body_start_line, body_end_line = _find_basic_blocks(func_src)

    if not blocks:
        return func_src

    if not isinstance(body_end_line, int):
        body_end_line = None

    assert_sites_by_line = defaultdict(list)  # end_idx (0-based) -> [site_id]
    jump_kw = re.compile(r"\b(return|break|continue)\b")
    loop_kw = re.compile(r"\b(while|for|loop)\b")

    a_idx = 0
    for bb in blocks:
        end_line = bb.get("end_line")
        if not isinstance(end_line, int):
            continue

        # Skip the very last block of the outer body to avoid implicit-return sites
        if body_end_line is not None and end_line == body_end_line:
            continue

        end_idx = end_line - 1
        if not (0 <= end_idx < n):
            continue

        last_line = lines[end_idx]
        stripped = last_line.strip()

        # Skip blocks whose last non-comment line has an explicit jump
        if stripped and not stripped.startswith("//") and jump_kw.search(last_line):
            continue

        # NEW: don't insert assert sites directly after while/for/loop
        code_part = last_line.split("//", 1)[0]
        if loop_kw.search(code_part):
            continue

        site_id = f"A{a_idx}"
        a_idx += 1
        assert_sites_by_line[end_idx].append(site_id)

    # Build new source with assert-site comments inserted
    out_lines: List[str] = []
    for i, line in enumerate(lines):
        out_lines.append(line)
        if i in assert_sites_by_line:
            # NEW: indentation = max(indent of this line, indent of next line)
            indent_this = _leading_ws(line)
            if i + 1 < n:
                indent_next = _leading_ws(lines[i + 1])
                indent = indent_this if len(indent_this) >= len(indent_next) else indent_next
            else:
                indent = indent_this

            for sid in assert_sites_by_line[i]:
                out_lines.append(f"{indent}// [ASSERT SITE {sid}]\n")

    return "".join(out_lines)


def insert_site_comments(func_src: str) -> str:
    """
    Insert both loop-site and assert-site comments into a function.

    1. Insert `// [LOOP SITE Lk]` between loop conditions and braces.
    2. Run `_find_basic_blocks` on the result and insert `// [ASSERT SITE Ak]`
       after selected basic-block ends.

    The resulting function contains only comments as markers; a later pass
    can scan for:

        // [LOOP SITE Lk]
        // [ASSERT SITE Ak]

    and treat those as the canonical site locations for invariants/decreases
    and assertions.
    """
    with_loops = insert_loop_site_comments(func_src)
    with_both = insert_assert_site_comments(with_loops)
    return with_both
