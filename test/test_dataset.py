from pathlib import Path
from fine_tune.extractors import collect_all_from_scratch


def _check_example_sane(ex):
    # 1) required top-level keys
    assert "input" in ex
    assert "output" in ex

    inp = ex["input"]
    out = ex["output"]

    # 2) required input structure
    assert isinstance(inp.get("base_func_src"), str) and inp["base_func_src"].strip()
    assert isinstance(inp.get("sites"), list) and len(inp["sites"]) > 0

    # 3) required output structure
    assert isinstance(out.get("annotations"), list)
    assert len(out["annotations"]) > 0

    site_ids = {s["id"] for s in inp["sites"]}
    assert site_ids  # non-empty

    for s in inp["sites"]:
        assert s["kind"] in ("loop_site", "assert_site")
        assert isinstance(s["line"], int) and s["line"] >= 1

    for a in out["annotations"]:
        assert a["site"] in site_ids
        assert a["kind"] in ("invariant", "decreases", "assert")

        if a["kind"] in ("invariant", "decreases"):
            clauses = a.get("clauses")
            assert isinstance(clauses, list)
            assert len(clauses) > 0
            for c in clauses:
                assert isinstance(c, str)
                assert c.strip()
        else:  # assert
            expr = a.get("expr")
            assert isinstance(expr, str)
            assert expr.strip()


def test_dataset_sanity_all_examples():
    """
    Load your prebuilt dataset (e.g. from JSONL) and check basic invariants.
    """
    examples = collect_all_from_scratch(Path("repositories"))
    assert len(examples) > 0

    for ex in examples:
        _check_example_sane(ex)
