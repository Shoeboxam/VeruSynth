import os
import re
import shutil
import subprocess
from typing import List, Tuple, Optional


def _compute_offsets(lines: List[str]) -> List[int]:
    """Compute starting character offset for each line in the concatenated text."""
    offsets, pos = [], 0
    for l in lines:
        offsets.append(pos)
        pos += len(l)
    return offsets


def _char_to_line(offsets: List[int], lines: List[str], idx: int) -> int:
    """Map a character index in the full text to a line index (0-based)."""
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


def find_function_span(
    text: str, lines: List[str], fn_name: str
) -> Optional[Tuple[int, int]]:
    """
    Find the [start_line, end_line] (0-based, inclusive) of a Rust function
    named `fn_name`, including its doc comments and attributes.

    Assumptions:
      - No multiline doc comments (/** ... */).
      - Doc comments and attributes directly precede the `fn` line:
            /// doc
            #[attr]
            fn name(...) { ... }
      - The function body is a brace-delimited block `{ ... }` with balanced braces.
    """
    # Match things like: fn foo( ...   OR   pub async unsafe fn foo(
    fn_header = re.compile(
        rf"^\s*(pub\s+)?(async\s+)?(unsafe\s+)?fn\s+{fn_name}\s*\(",
    )

    header_idx = None
    for i, line in enumerate(lines):
        if fn_header.search(line):
            header_idx = i
            break

    if header_idx is None:
        return None

    # Include preceding single-line docs and attributes
    start_idx = header_idx
    while start_idx > 0:
        prev = lines[start_idx - 1].lstrip()
        if prev.startswith("///") or prev.startswith("//!") or prev.startswith("#["):
            start_idx -= 1
        else:
            break

    # Find the first '{' after the header line in the full text
    offsets = _compute_offsets(lines)
    header_offset = offsets[header_idx]
    open_pos = text.find("{", header_offset)
    if open_pos == -1:
        return None

    # Find the matching closing brace by brace counting over the full text
    depth = 1
    i = open_pos + 1
    close_pos = None
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                close_pos = i
                break
        i += 1

    if close_pos is None:
        return None  # malformed / unbalanced

    end_idx = _char_to_line(offsets, lines, close_pos)
    return start_idx, end_idx


def replace_rust_function_in_file(
    path: str,
    fn_name: str,
    new_function_src: str,
    backup_suffix: str = ".bak",
) -> bool:
    """
    Replace a Rust function in a file with a new function definition.

    Behavior:
      1. Reads the file at `path`.
      2. Locates the function `fn fn_name(...)` (including preceding doc comments
         and attributes) using brace counting.
      3. Replaces that entire region with `new_function_src`.
      4. Before writing, if a backup file (path + backup_suffix) does NOT exist,
         saves a copy of the original file there.

    Assumptions:
      - No multiline doc comments (/** ... */).
      - Functions are standard Rust or Verus free functions/methods:
          [doc/attrs...]
          [pub] [async] [unsafe] fn name(...) { ... }
      - `new_function_src` is a complete function definition, including any
        doc comments, attributes, signature, and body.

    Args:
        path:
            Path to the `.rs` file.

        fn_name:
            Name of the function to replace (e.g., "linear_search").

        new_function_src:
            The full source of the replacement function as a string.
            It should end with a newline if you want a trailing newline.

        backup_suffix:
            Suffix to append for the backup file (default: ".bak").
            E.g., "src/lib.rs" → "src/lib.rs.bak".

    Returns:
        True if the replacement succeeded, False if the function was not found.

    Raises:
        OSError / IOError if file I/O fails.
    """
    with open(path, "r", encoding="utf8") as f:
        text = f.read()

    lines = text.splitlines(keepends=True)

    span = find_function_span(text, lines, fn_name)
    if span is None:
        return False

    start_idx, end_idx = span

    # Prepare new function lines
    new_lines = new_function_src.splitlines(keepends=True)
    if new_lines and not new_lines[-1].endswith("\n"):
        new_lines[-1] = new_lines[-1] + "\n"

    # Build new file content
    updated_lines = lines[:start_idx] + new_lines + lines[end_idx + 1 :]
    updated_text = "".join(updated_lines)

    # Backup original file if backup does not exist
    backup_path = path + backup_suffix
    if not os.path.exists(backup_path):
        shutil.copy2(path, backup_path)

    # Write updated file
    with open(path, "w", encoding="utf8") as f:
        f.write(updated_text)

    return True


def read_rust_function_from_file(path: str, fn_name: str) -> Optional[str]:
    """
    Read a Rust function by name from `path`, including doc comments and
    attributes above it, using `_find_function_span`.
    """
    with open(path, "r", encoding="utf8") as f:
        text = f.read()
    lines = text.splitlines(keepends=True)
    span = find_function_span(text, lines, fn_name)
    if span is None:
        return None
    s, e = span
    return "".join(lines[s : e + 1])


def run_verus(file_path: str) -> Tuple[bool, str]:
    """Run `cargo build` in project_root, return (success, combined_output)."""
    proc = subprocess.run(
        ["verus", file_path],
        capture_output=True,
        text=True,
    )
    ok = proc.returncode == 0
    output = (proc.stdout or "") + (proc.stderr or "")
    return ok, output
