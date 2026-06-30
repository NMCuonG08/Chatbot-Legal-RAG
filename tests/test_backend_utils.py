from backend.src.splitter import split_document


def test_split_document() -> None:
    nodes = split_document("Đây là đoạn văn bản tiếng Việt dài để kiểm tra bộ cắt nhỏ tài liệu pháp lý.", use_semantic=False)
    assert len(nodes) > 0
    assert any("tiếng Việt" in node.text for node in nodes)
