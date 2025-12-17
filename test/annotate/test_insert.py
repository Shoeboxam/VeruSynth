import textwrap

from verusynth.annotate.insert import insert_site_comments


def dedent(s: str) -> str:
    """Normalize indentation and trailing newlines for comparisons."""
    return textwrap.dedent(s).lstrip("\n").rstrip() + "\n"


def test_insert_site_comments_simple_while_brace_same_line():
    src = dedent(
        """
        fn f() {
            while i < N {
                i += 1;
            }
        }
        """
    )

    out = insert_site_comments(src)

    # We expect:
    #   - loop-site comment
    #   - brace still present
    expected = dedent(
        """
        fn f() {
            while i < N
                // [LOOP SITE L0]
                {
                i += 1;
            }
        }
        """
    )

    assert out == expected


def test_insert_site_comments_while_brace_next_line():
    src = dedent(
        """
        fn f() {
            while i < N
            {
                i += 1;
            }
        }
        """
    )

    out = insert_site_comments(src)

    # The loop-site comment should appear between the condition and the brace.
    expected = dedent(
        """
        fn f() {
            while i < N
            // [LOOP SITE L0]
            {
                i += 1;
            }
        }
        """
    )

    assert out == expected


def test_insert_site_comments_multiline_condition():
    src = dedent(
        """
        fn f() {
            while (a
                       && b) {
                body();
            }
        }
        """
    )

    out = insert_site_comments(src)

    # The comment should be between the last condition line and the brace,
    # even with a multi-line header.
    expected = dedent(
        """
        fn f() {
            while (a
                       && b)
                // [LOOP SITE L0]
                {
                body();
            }
        }
        """
    )

    assert out == expected


def test_insert_site_comments_includes_assert_site_somewhere():
    """
    We don't assert the *exact* placement of assert sites, because that
    depends on _find_basic_blocks, but we do expect at least one
    // [ASSERT SITE A0] marker for a function with multiple basic blocks.
    """
    src = dedent(
        """
        fn f(a: &mut [i32], n: usize) {
            let mut i = 0;
            while i < n {
                if a[i] == 1 {
                    a[i] += 4;
                } else {
                    a[i] -= 1;
                }
                i += 1;
            }
            let y = i;
        }
        """
    )

    out = insert_site_comments(src)

    # Loop site should be there
    assert "// [LOOP SITE L0]" in out

    # Given a non-trivial function, we expect at least one assert site
    # (exact line depends on _find_basic_blocks)
    assert "// [ASSERT SITE A0]" in out


def test_insert_site_comments_brace_does_not_disappear():
    """
    Regression guard: inserting comments must not delete the loop body brace.
    """
    src = dedent(
        """
        fn f() {
            while i < N {
                i += 1;
            }
        }
        """
    )

    out = insert_site_comments(src)

    # There should still be a '{' for the loop body.
    # Focus on the region around the loop.
    loop_region = "\n".join(out.splitlines()[1:5])
    assert "{" in loop_region
    assert "// [LOOP SITE L0]" in loop_region
