"""Phase 2 — audit log tests (sqlite in-memory)."""
from audit import list_audit_entries, log_audit
from models import _new_db_session


def test_log_audit_persists_entry(sqlite_db):
    log_audit(user_id="u1", action="login", resource="user", ip="127.0.0.1", payload={"k": "v"})
    rows = list_audit_entries(limit=10)
    assert len(rows) == 1
    row = rows[0]
    assert row["user_id"] == "u1"
    assert row["action"] == "login"
    assert row["resource"] == "user"
    assert row["ip"] == "127.0.0.1"
    assert row["payload"] == {"k": "v"}


def test_log_audit_never_raises_on_db_error(sqlite_db, monkeypatch):
    def _boom(*a, **kw):
        raise RuntimeError("simulated db failure")

    monkeypatch.setattr("audit._new_db_session", _boom)
    log_audit(user_id="u2", action="chat")  # must not raise
    monkeypatch.undo()  # restore real session so the read works
    assert list_audit_entries(limit=10) == []


def test_list_audit_filters_by_action(sqlite_db):
    log_audit(user_id="u1", action="login")
    log_audit(user_id="u1", action="chat")
    log_audit(user_id="u2", action="login")
    logins = list_audit_entries(limit=10, action="login")
    assert len(logins) == 2
    assert all(r["action"] == "login" for r in logins)


def test_list_audit_filters_by_user(sqlite_db):
    log_audit(user_id="u1", action="login")
    log_audit(user_id="u2", action="login")
    rows = list_audit_entries(limit=10, user_id="u2")
    assert len(rows) == 1
    assert rows[0]["user_id"] == "u2"


def test_log_audit_uses_provided_session(sqlite_db):
    db = _new_db_session()
    try:
        log_audit(user_id="u3", action="register", db=db)
        db.commit()
    finally:
        db.close()
    assert len(list_audit_entries(limit=10)) == 1