"""Unit tests for PII detector: CCCD, phone, email, address, no-false-positive,
immutability."""
from guardrails_manager import detect_pii_vietnamese, verify_output_pii


def test_detect_cccd_12_digit():
    f = detect_pii_vietnamese("CCCD của tôi là 079200012345")
    assert "cccd" in f
    assert "079200012345" in f["cccd"]


def test_detect_cccd_9_digit():
    f = detect_pii_vietnamese("số 123456789")
    assert "cccd" in f


def test_detect_phone():
    f = detect_pii_vietnamese("liên hệ 0912345678 hoặc +84912345678")
    assert "phone" in f
    assert "0912345678" in f["phone"]


def test_detect_email():
    f = detect_pii_vietnamese("gửi cho user@example.com nhé")
    assert "email" in f
    assert "user@example.com" in f["email"]


def test_detect_address():
    f = detect_pii_vietnamese("Tôi ở số 12 phường X, quận Y")
    assert "address" in f


def test_no_false_positive_legal_text():
    f = detect_pii_vietnamese("Theo Điều 10 của Bộ luật Dân sự 2015...")
    assert f.get("cccd") is None
    assert "email" not in f
    assert "phone" not in f


def test_empty_and_none():
    assert detect_pii_vietnamese("") == {}
    assert detect_pii_vietnamese(None) == {}


def test_verify_output_pii_blocked_flag():
    blocked, findings = verify_output_pii("CCCD 079200012345")
    assert blocked is True
    assert "cccd" in findings


def test_verify_output_pii_clean():
    blocked, findings = verify_output_pii("Theo Điều 10, hợp đồng có hiệu lực.")
    assert blocked is False
    assert findings == {}


def test_dedup_repeated_pii():
    f = detect_pii_vietnamese("0912345678 và 0912345678 nữa")
    assert f["phone"].count("0912345678") == 1