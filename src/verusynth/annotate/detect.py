import re
from typing import List, Dict, Any


def detect_sites_from_comments(func_src_with_comments: str) -> List[Dict[str, Any]]:
    """
    Scan a function that contains site comments of the form:

        // [LOOP SITE Lk]
        // [ASSERT SITE Ak]

    and reconstruct a site list:

        { "id": "L0", "kind": "loop_site", "line": <int> }
        { "id": "A0", "kind": "assert_site", "line": <int> }

    where `line` is 1-based and refers to the line of the comment itself.
    """
    lines = func_src_with_comments.splitlines(keepends=True)

    loop_pat = re.compile(r'^\s*//\s*\[LOOP SITE (L\d+)\]')
    assert_pat = re.compile(r'^\s*//\s*\[ASSERT SITE (A\d+)\]')

    sites: List[Dict[str, Any]] = []

    for idx, line in enumerate(lines):
        line_no = idx + 1
        m_loop = loop_pat.match(line)
        if m_loop:
            sid = m_loop.group(1)
            sites.append(
                {
                    "id": sid,
                    "kind": "loop_site",
                    "line": line_no,
                }
            )
            continue

        m_assert = assert_pat.match(line)
        if m_assert:
            sid = m_assert.group(1)
            sites.append(
                {
                    "id": sid,
                    "kind": "assert_site",
                    "line": line_no,
                }
            )
            continue

    return sites
