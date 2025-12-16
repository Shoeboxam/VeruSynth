from verusynth.annotate import _find_basic_blocks, annotate_rust_function

func = """
fn linear_search(nums: Vec<i32>) -> (ret: i32) {
    let mut i = 0;
    let mut sum = 0; 
    while i < nums.len() {
        if sum + nums[i] > 100 {
            break;
        }
        sum = sum + nums[i];
        i = i + 1;
    }
    let r = sum;
    return r;
}"""

def test_basic_blocks():

    for block, line in zip(_find_basic_blocks(func), [3, 6, 7, 8, 9, 12]):
        assert block["start_line"] == line

def test_annotate():
    annotated, sites = annotate_rust_function(func)
    print(annotated)
