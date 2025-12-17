from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Any
import re



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

    loop_header_re = re.compile(r"\b(while|for|loop)\b")
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
