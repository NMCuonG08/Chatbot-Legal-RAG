"""Phase 2 — RBAC tool-policy tests (no DB, pure set logic)."""
from types import SimpleNamespace

from rbac import (
    APPROVAL_EXEMPT_ROLES,
    LEGAL_TOOLS,
    Role,
    SENSITIVE_TOOLS,
    Principal,
    allowed_tool_names,
    filter_tools_by_policy,
    is_tool_allowed,
    needs_approval,
)


def _tool(name: str):
    return SimpleNamespace(name=name)


def test_principal_admin_is_approval_exempt():
    p = Principal(user_id="u1", username="a", role=Role.ADMIN)
    assert p.is_admin is True
    assert p.is_approval_exempt is True


def test_principal_user_not_exempt():
    p = Principal(user_id="u2", username="b", role=Role.USER)
    assert p.is_admin is False
    assert p.is_approval_exempt is False


def test_filter_tools_admin_keeps_all():
    tools = [_tool(n) for n in ("web_search_tool", "get_current_time", "contract_penalty_calculator")]
    p = Principal(user_id="a", username="admin", role=Role.ADMIN)
    kept = filter_tools_by_policy(tools, p)
    assert {t.name for t in kept} == {t.name for t in tools}


def test_filter_tools_user_drops_web():
    tools = [_tool("web_search_tool"), _tool("contract_penalty_calculator"), _tool("recall_user_memory_tool")]
    p = Principal(user_id="u", username="user", role=Role.USER)
    kept = filter_tools_by_policy(tools, p)
    names = {t.name for t in kept}
    assert "web_search_tool" not in names
    assert "contract_penalty_calculator" in names
    assert "recall_user_memory_tool" in names


def test_filter_tools_guest_readonly_subset():
    tools = [_tool("article_lookup_tool"), _tool("web_search_tool"), _tool("contract_penalty_calculator")]
    p = Principal(user_id="g", username="guest", role=Role.GUEST)
    kept = filter_tools_by_policy(tools, p)
    names = {t.name for t in kept}
    assert names == {"article_lookup_tool"}
    assert "web_search_tool" not in names
    assert "contract_penalty_calculator" not in names


def test_filter_tools_none_principal_guest():
    tools = [_tool("article_lookup_tool"), _tool("web_search_tool")]
    kept = filter_tools_by_policy(tools, None)
    assert {t.name for t in kept} == {"article_lookup_tool"}


def test_is_tool_allowed_role_matrix():
    assert is_tool_allowed("web_search_tool", Role.ADMIN) is True
    assert is_tool_allowed("web_search_tool", Role.LAWYER) is True
    assert is_tool_allowed("web_search_tool", Role.USER) is False
    assert is_tool_allowed("contract_penalty_calculator", Role.USER) is True


def test_needs_approval_sensitive_for_non_exempt():
    assert needs_approval("web_search_tool", Principal("u", "u", Role.USER)) is True
    assert needs_approval("generate_document_template_tool", Principal("u", "u", Role.GUEST)) is True


def test_needs_approval_exempt_roles_skip():
    assert needs_approval("web_search_tool", Principal("a", "a", Role.ADMIN)) is False
    assert needs_approval("web_search_tool", Principal("l", "l", Role.LAWYER)) is False


def test_needs_approval_non_sensitive_never():
    assert needs_approval("get_current_time", Principal("u", "u", Role.USER)) is False
    assert needs_approval("contract_penalty_calculator", None) is False


def test_sensitive_tools_subset_of_legal_or_web():
    assert SENSITIVE_TOOLS.issubset(LEGAL_TOOLS | {"web_search_tool", "tavily_search_tool"})


def test_exempt_roles_are_admin_lawyer():
    assert APPROVAL_EXEMPT_ROLES == {Role.ADMIN, Role.LAWYER}


def test_allowed_tool_names_wildcard_admin():
    assert allowed_tool_names(Role.ADMIN) == {"*"}


def test_allowed_tool_names_user_subset():
    names = allowed_tool_names(Role.USER)
    assert "contract_penalty_calculator" in names
    assert "web_search_tool" not in names


# ---- FastAPI dependencies (get_current_user / require_role) ----
# These cover the HTTP-boundary deps via a tiny TestClient + real JWT.


def _build_app():
    from fastapi import FastAPI

    import rbac

    app = FastAPI()

    @app.get("/me-optional")
    def me_optional(p=rbac.Depends(rbac.get_current_user_optional)):
        return {"user_id": p.user_id if p else None, "role": p.role if p else None}

    @app.get("/me-strict")
    def me_strict(p: rbac.Principal = rbac.Depends(rbac.get_current_user)):
        return {"user_id": p.user_id, "role": p.role}

    @app.get("/admin-only")
    def admin_only(p: rbac.Principal = rbac.Depends(rbac.require_admin)):
        return {"user_id": p.user_id}

    @app.get("/lawyer-only")
    def lawyer_only(p: rbac.Principal = rbac.Depends(rbac.require_role(rbac.Role.LAWYER))):
        return {"user_id": p.user_id}

    return app


def test_get_current_user_optional_no_token(jwt_secret, sqlite_db):
    from fastapi.testclient import TestClient

    client = TestClient(_build_app())
    r = client.get("/me-optional")
    assert r.status_code == 200
    assert r.json() == {"user_id": None, "role": None}


def test_get_current_user_optional_valid_token(jwt_secret, sqlite_db):
    from fastapi.testclient import TestClient

    import auth

    token = auth.create_access_token(subject="u123", claims={"role": Role.USER, "username": "bob"})
    client = TestClient(_build_app())
    r = client.get("/me-optional", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert r.json() == {"user_id": "u123", "role": Role.USER}


def test_get_current_user_optional_bad_token_returns_none(jwt_secret, sqlite_db):
    from fastapi.testclient import TestClient

    client = TestClient(_build_app())
    r = client.get("/me-optional", headers={"Authorization": "Bearer not-a-jwt"})
    assert r.status_code == 200
    assert r.json() == {"user_id": None, "role": None}


def test_get_current_user_strict_missing_token_401(jwt_secret, sqlite_db):
    from fastapi.testclient import TestClient

    client = TestClient(_build_app())
    r = client.get("/me-strict")
    assert r.status_code == 401


def test_get_current_user_strict_valid_token(jwt_secret, sqlite_db):
    from fastapi.testclient import TestClient

    import auth

    token = auth.create_access_token(subject="u123", claims={"role": Role.USER, "username": "bob"})
    client = TestClient(_build_app())
    r = client.get("/me-strict", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert r.json() == {"user_id": "u123", "role": Role.USER}


def test_require_admin_blocks_user(jwt_secret, sqlite_db):
    from fastapi.testclient import TestClient

    import auth

    token = auth.create_access_token(subject="u1", claims={"role": Role.USER, "username": "u"})
    client = TestClient(_build_app())
    r = client.get("/admin-only", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 403


def test_require_admin_allows_admin(jwt_secret, sqlite_db):
    from fastapi.testclient import TestClient

    import auth

    token = auth.create_access_token(subject="a1", claims={"role": Role.ADMIN, "username": "admin"})
    client = TestClient(_build_app())
    r = client.get("/admin-only", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert r.json() == {"user_id": "a1"}


def test_require_role_blocks_wrong_role(jwt_secret, sqlite_db):
    from fastapi.testclient import TestClient

    import auth

    token = auth.create_access_token(subject="u1", claims={"role": Role.USER, "username": "u"})
    client = TestClient(_build_app())
    r = client.get("/lawyer-only", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 403


def test_require_role_allows_matching_role(jwt_secret, sqlite_db):
    from fastapi.testclient import TestClient

    import auth

    token = auth.create_access_token(subject="l1", claims={"role": Role.LAWYER, "username": "lw"})
    client = TestClient(_build_app())
    r = client.get("/lawyer-only", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200