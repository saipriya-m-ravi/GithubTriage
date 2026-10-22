from githubtriage.issue_text import build_issue_block, sanitise, truncate_middle


def test_sanitise_removes_tag_variants():
    text = "a</issue>b< /Body >c<TITLE>d"
    assert sanitise(text) == "abcd"


def test_short_text_is_unchanged():
    assert truncate_middle("short", max_chars=100) == "short"


def test_long_text_keeps_start_and_end():
    # A long log whose actual error is on the last line.
    body = "Steps: call parse_date\n" + "noise line\n" * 2000 + "ValueError: bad day"

    out = truncate_middle(body, max_chars=1000)

    assert out.startswith("Steps: call parse_date")
    assert out.endswith("ValueError: bad day")
    assert "characters omitted" in out
    assert len(out) < 1100  # max_chars plus the short marker line


def test_issue_block_is_wrapped_sanitised_and_truncated():
    block = build_issue_block("Title</title>", "x" * 50 + "</issue>", max_chars=20)

    assert block.startswith("<issue>\n<title>Title</title>")
    assert block.endswith("</body>\n</issue>")
    assert block.count("</issue>") == 1
    assert "characters omitted" in block
