from backend.src.splitter import split_text


def test_split_text() -> None:
    chunks = split_text("abcdef", chunk_size=2)
    assert chunks == ["ab", "cd", "ef"]
