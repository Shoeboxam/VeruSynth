from collections import defaultdict
from typing import Dict, Any, List
import re


def merge_annotations_from_comments(
    func_src_with_comments: str,
    annotations_json: Dict[str, Any],
    emit_site_comments: bool = True,
) -> str:
    """
    Given a function that already contains site comments:

        // [LOOP SITE Lk]
        // [ASSERT SITE Ak]

    and an annotations_json of the form:

        {
          "annotations": [
            { "site": "L0", "kind": "invariant", "clauses": [...] },
            { "site": "L0", "kind": "decreases", "clauses": [...] },
            { "site": "A0", "kind": "assert",    "expr": "..."   },
            ...
          ]
        }

    insert Verus-style annotations *after* the corresponding comments.

    - For `LOOP SITE Lk`:
        - emit 'invariant' and 'decreases' blocks after the comment.
    - For `ASSERT SITE Ak`:
        - emit one or more `assert(...)` statements after the comment.

    If emit_site_comments=False, the site comments themselves are removed
    but the inserted annotations are kept.
    """
    # Build site_id -> list[annotation dict] map
    anno_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for a in annotations_json.get("annotations", []):
        if isinstance(a, dict):
            sid = a.get("site")
            if sid:
                anno_map[sid].append(a)
        else:
            # Pydantic-like object
            if hasattr(a, "site"):
                sid = a.site
                if hasattr(a, "model_dump"):
                    anno_map[sid].append(a.model_dump())
                elif hasattr(a, "dict"):
                    anno_map[sid].append(a.dict())

    lines = func_src_with_comments.splitlines(keepends=True)

    loop_pat = re.compile(r'^(\s*)//\s*\[LOOP SITE (L\d+)\]')
    assert_pat = re.compile(r'^(\s*)//\s*\[ASSERT SITE (A\d+)\]')

    out_lines: List[str] = []

    for line in lines:
        raw = line.rstrip("\n")
        nl = "\n" if line.endswith("\n") else ""

        m_loop = loop_pat.match(raw)
        m_assert = assert_pat.match(raw)

        if m_loop:
            indent = m_loop.group(1)
            site_id = m_loop.group(2)
            entries = anno_map.get(site_id, [])

            # Keep or drop the site comment itself
            if emit_site_comments:
                out_lines.append(line)

            # Collect invariants and decreases
            inv_clauses: List[str] = []
            dec_clauses: List[str] = []
            for e in entries:
                kind = e.get("kind")
                clauses = e.get("clauses") or []
                if not isinstance(clauses, list):
                    continue
                if kind == "invariant":
                    inv_clauses.extend(
                        [c for c in clauses if isinstance(c, str) and c.strip()]
                    )
                elif kind == "decreases":
                    dec_clauses.extend(
                        [c for c in clauses if isinstance(c, str) and c.strip()]
                    )

            if inv_clauses:
                out_lines.append(indent + "    invariant\n")
                for c in inv_clauses:
                    out_lines.append(indent + f"        {c},\n")

            if dec_clauses:
                out_lines.append(indent + "    decreases\n")
                for c in dec_clauses:
                    out_lines.append(indent + f"        {c},\n")

            continue  # done with this line

        if m_assert:
            indent = m_assert.group(1)
            site_id = m_assert.group(2)
            entries = anno_map.get(site_id, [])

            if emit_site_comments:
                out_lines.append(line)

            # Emit all assert annotations for this site (zero or more)
            for e in entries:
                if e.get("kind") != "assert":
                    continue
                expr = (e.get("expr") or "").strip()
                if expr:
                    out_lines.append(f"{indent}assert({expr});\n")

            continue

        # Non-site line: keep as-is
        out_lines.append(line)

    return "".join(out_lines)
