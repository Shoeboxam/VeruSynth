
from verusynth.harness import run_annotation_loop_on_function


def test_loop():
    run_annotation_loop_on_function(
        file_path="../src/main.rs",
        fn_name="linear_search"
    )

test_loop()