import json
from typing import List, Dict, Any, Literal, Optional, Union
from pydantic import BaseModel

import outlines
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch


class ErrorSummaryOutput(BaseModel):
    summary: str


class Annotation(BaseModel):
    site: str
    kind: Literal["decreases", "invariant", "assert"]
    clauses: Optional[List[str]] = None  # for invariants, decreases
    expr: Optional[str] = None  # for asserts


class AnnotationOutput(BaseModel):
    annotations: List[Annotation]


def summarize_verus_errors(
    error_history: list[str],
    model: Union[outlines.Transformers, outlines.TransformersMultiModal],
    max_entries: int = 5,
    max_chars: int = 4000,
) -> str:
    """
    Summarize the *history* of Verus errors via the LLM.

    Steps:
      1. Build a labeled, truncated aggregate of recent error logs.
      2. Ask Qwen to extract high-level, recurring issues and return a concise summary.
    """
    if not error_history:
        return ""

    # 1) Aggregate recent errors into a single text blob
    recent = error_history[-max_entries:]

    labeled = []
    # Offset index so numbering is stable across iterations
    first_idx = max(1, len(error_history) - len(recent) + 1)
    for i, err in enumerate(recent, start=first_idx):
        labeled.append(f"--- Error #{i} ---\n{(err or '').strip()}")

    raw_history = "\n\n".join(labeled)
    if len(raw_history) > max_chars:
        raw_history = raw_history[:max_chars] + "\n\n[truncated]"

    # 2) Build a concise prompt asking for *high-level* takeaways
    prompt = f"""
You are helping to iteratively verify Rust code with Verus. Below is a series of
Verus error logs from previous iterations of the same function. Your job is to
give general advice on how to avoid these kinds of errors.

Error history:

```text
{raw_history}
````

Return a JSON object of the form:

{{
"summary": "<advice for avoiding errors, in at most 250 words>"
}}
""".strip()

    # Outlines returns a JSON string matching the Pydantic schema
    raw = model(prompt, ErrorSummaryOutput, max_new_tokens=256)
    return ErrorSummaryOutput.model_validate_json(raw).summary


def build_annotation_prompt(
    *,
    func_src: str,
    sites: List[Dict[str, Any]],
    current_annotations: Dict[str, Any] | None = None,
    latest_errors: str = "",
    error_summary: str = "",
) -> str:
    """
    Universal prompt builder for both from-scratch and iterative repair.

    Arguments:
      func_src:
          The version of the function to show the LLM.
          For from-scratch: base_func_src or annotated-with-site-markers.
          For iterative repair: the annotated function (with invariants/decreases/asserts).

      sites:
          List of site dicts:
            {"id": "L0", "kind": "loop_site", "line": <int>}
            {"id": "A0", "kind": "assert_site", "line": <int>}

      current_annotations:
          Optional: the existing annotation set ("annotations": [...]).
          Will be displayed only if non-empty.

      latest_errors, error_summary:
          Optional Verus error strings.

      include_error_context:
          If False → omit error text entirely (from-scratch mode).
          If True → show both latest_errors + error_summary (iterative mode).

    Returns:
        A natural-language prompt instructing the LLM to produce annotation JSON.
    """

    # -------------------------
    # Organize and sort sites
    # -------------------------
    loop_sites = [s for s in sites if s["kind"] == "loop_site"]
    assert_sites = [s for s in sites if s["kind"] == "assert_site"]

    def site_key(s):
        sid = s["id"]
        prefix = sid[0] if sid else ""
        num = int(sid[1:]) if sid[1:].isdigit() else 0
        return (prefix, num)

    loop_sites.sort(key=site_key)
    assert_sites.sort(key=site_key)

    # Turn site metadata into readable text
    site_lines = []
    for s in loop_sites:
        site_lines.append(f"- {s['id']} (loop site on line {s['line']})")
    for s in assert_sites:
        site_lines.append(f"- {s['id']} (assert site on line {s['line']})")
    sites_section = "\n".join(site_lines) if site_lines else "(none)"

    # -------------------------
    # Current annotations section
    # -------------------------
    current_ann_section = ""
    if current_annotations and current_annotations.get("annotations"):
        formatted = json.dumps(current_annotations, indent=2, ensure_ascii=False)
        current_ann_section = f"""
Current annotation set (this will be COMPLETELY REPLACED):

```json
{formatted}
````

""".rstrip()

    # -------------------------
    # Error context section (optional)
    # -------------------------
    error_section = ""
    parts = []
    if latest_errors.strip():
        parts.append(
            f"Most recent Verus error output:\n```text\n{latest_errors.strip()}\n```"
        )
    if error_summary.strip():
        parts.append(
            f"Summary of previous verification errors:\n```text\n{error_summary.strip()}\n```"
        )
    if parts:
        error_section = "\n\n".join(parts)

    # -------------------------
    # DSL example (invariant / decreases / assert)
    # -------------------------
    dsl_spec = r"""{
    "annotations": [
        {
            "site": "L0",
            "kind": "invariant",
            "clauses": [
                "i <= N",
                "sum == 2*i"
            ]
        },
        {
            "site": "L0",
            "kind": "decreases",
                "clauses": [
                "N - i"
            ]
        },
        {
            "site": "A0",
            "kind": "assert",
            "expr": "i == nums.len() || sum + nums[i as int] > 100"
        }
    ]
}"""

    # -------------------------
    # Assemble final prompt
    # -------------------------

    prompt = f"""You are helping to generate Verus-style annotations for Rust code.
```

The function below contains annotation *sites* but may or may not contain existing annotations.
Your job is to output a COMPLETE JSON annotation set for all listed sites.

Rules:
• Do NOT modify the Rust code structure.
• Do NOT invent new site IDs; use only the ones provided.
• For each loop site L#:
- invariants:    {{ "kind": "invariant", "clauses": [...] }}
- decreases:     {{ "kind": "decreases", "clauses": [...] }}
• For each assert site A#:
- assertions:    {{ "kind": "assert",     "expr": "..." }}

Rust function:

```rust
{func_src}
```

Valid annotation sites:
{sites_section}
"""

    # Append current annotations if present
    if current_ann_section:
        prompt += f"\n\n{current_ann_section}\n"

    # Append error context if requested
    if error_section:
        prompt += f"\n\n{error_section}\n"

    # Output format spec
    prompt += f"""
```

Output format:

Return a single JSON object exactly of the form:

```json
{dsl_spec}
```

Do not include commentary, markdown, or explanation—only the JSON.
"""
    return prompt.strip()


def load_model(
    base_model: str,
    finetuned_adapter_path: Optional[str] = None,
):
    print(f"[verusynth] Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )

    if finetuned_adapter_path:
        print(f"[verusynth] Applying LoRA adapter from: {finetuned_adapter_path}")
        model = PeftModel.from_pretrained(model, finetuned_adapter_path)
    else:
        print("[verusynth] Using base model (no adapter)")

    model.eval()

    return outlines.from_transformers(model, tokenizer)



def call_annotator_model(
    prompt: str,
    model: Union[outlines.Transformers, outlines.TransformersMultiModal],
    max_new_tokens: int = 1024,
) -> AnnotationOutput:
    # Outlines returns a JSON string matching the Pydantic schema
    raw = model(prompt, AnnotationOutput, max_new_tokens=max_new_tokens)
    return AnnotationOutput.model_validate_json(raw)
