from ui.main import split_lines

def test_safe_text_rendering_path_keeps_text_as_plain_lines():
    assert split_lines("<script>alert(1)</script>\n") == ["<script>alert(1)</script>"]
