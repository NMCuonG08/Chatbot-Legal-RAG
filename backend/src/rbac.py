"""Role-based access control + tool policy for the legal chatbot.

Roles (enum string stored on User.role):
- admin  : full access, approval-exempt, admin endpoints.
- lawyer : all legal/web tools, approval-exempt, no admin endpoints.
- user   : legal + memory + utility tools (no web), approval required for
           sensitive tools.
- guest  : read-only retrieval/disclaimer only, approval required for sensitive.

Tool names follow the LlamaIndex FunctionTool names defined in
``agent_tool_wrappers.py`` (= the underlying wrapper function ``__name__``).

Policy surfaces:
- ``filter_tools_by_policy(tools, principal)`` — keep only tools the principal's
  role may call. Used by agent.py before building the ReAct agent.
- ``needs_approval(tool_name, principal)`` — True when a sensitive tool is
  requested by a non-exempt role.
- ``require_role(*roles)`` — FastAPI dependency gating endpoints by role.
- ``get_current_user`` / ``get_current_user_optional`` — JWT -> Principal.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Set

from fastapi import Depends, Header, HTTPException, status

import auth
from database import get_db

logger = logging.getLogger(__name__)


# ---- Roles ----
class Role:
    ADMIN = "admin"
    LAWYER = "lawyer"
    USER = "user"
    GUEST = "guest"
    ALL = (ADMIN, LAWYER, USER, GUEST)


@dataclass(frozen=True)
class Principal:
    """Caller identity surfaced to the graph/agent (no password hash)."""
    user_id: str
    username: str
    role: str

    @property
    def is_admin(self) -> bool:
        return self.role == Role.ADMIN

    @property
    def is_approval_exempt(self) -> bool:
        return self.role in (Role.ADMIN, Role.LAWYER)


# ---- Tool name sets (must match FunctionTool names in agent_tool_wrappers) ----
LEGAL_TOOLS: Set[str] = {
    "contract_penalty_calculator", "legal_age_checker", "inheritance_calculator",
    "business_name_validator", "statute_lookup",
    "article_lookup_tool", "precedent_lookup_tool", "cross_reference_tool",
    "verify_citation_tool",
    "severance_pay_tool", "overtime_pay_tool", "pit_monthly_tool",
    "land_registration_fee_tool", "vehicle_registration_fee_tool",
    "court_fee_tool", "admin_fine_lookup_tool", "child_support_tool",
    "procedure_wizard_tool", "jurisdiction_resolver_tool",
    "generate_document_template_tool", "law_version_tool", "legal_disclaimer_tool",
    "recall_legal_graph_tool",
}
WEB_TOOLS: Set[str] = {"web_search_tool", "tavily_search_tool", "quick_answer_tool"}
MEMORY_TOOLS: Set[str] = {"recall_user_memory_tool"}
UTILITY_TOOLS: Set[str] = {"get_current_time"}

GUEST_TOOLS: Set[str] = {
    "article_lookup_tool", "precedent_lookup_tool", "cross_reference_tool",
    "verify_citation_tool", "law_version_tool", "legal_disclaimer_tool",
    "get_current_time",
}

# Sensitive tools: a non-exempt role must obtain approval before these run.
SENSITIVE_TOOLS: Set[str] = {
    "generate_document_template_tool", "web_search_tool", "tavily_search_tool",
}

# role -> allowed tool names ("*" = all)
ROLE_PERMISSIONS: dict[str, Set[str]] = {
    Role.ADMIN: {"*"},
    Role.LAWYER: LEGAL_TOOLS | WEB_TOOLS | MEMORY_TOOLS | UTILITY_TOOLS,
    Role.USER: LEGAL_TOOLS | MEMORY_TOOLS | UTILITY_TOOLS,
    Role.GUEST: GUEST_TOOLS,
}

APPROVAL_EXEMPT_ROLES: Set[str] = {Role.ADMIN, Role.LAWYER}


# ---- Tool policy helpers ----
def _tool_name(tool: Any) -> str:
    """Read a LlamaIndex FunctionTool name across versions."""
    name = getattr(tool, "name", None)
    if name:
        return name
    meta = getattr(tool, "metadata", None)
    return getattr(meta, "name", "") or ""


def allowed_tool_names(role: str) -> Set[str]:
    perms = ROLE_PERMISSIONS.get(role, set())
    if perms == {"*"}:
        return {"*"}
    return set(perms)


def filter_tools_by_policy(tools: Iterable[Any], principal: Optional[Principal]) -> list:
    """Keep only tools the principal's role may call. None principal -> guest."""
    role = principal.role if principal else Role.GUEST
    perms = ROLE_PERMISSIONS.get(role, set())
    if perms == {"*"}:
        return list(tools)
    return [t for t in tools if _tool_name(t) in perms]


def is_tool_allowed(tool_name: str, role: str) -> bool:
    perms = ROLE_PERMISSIONS.get(role, set())
    return perms == {"*"} or tool_name in perms


def needs_approval(tool_name: str, principal: Optional[Principal]) -> bool:
    if tool_name not in SENSITIVE_TOOLS:
        return False
    if principal is None:
        return True
    return principal.role not in APPROVAL_EXEMPT_ROLES


# ---- FastAPI dependencies ----
def _load_principal_from_token(token: Optional[str], db) -> Optional[Principal]:
    if not token:
        return None
    try:
        payload = auth.decode_token(token)
    except Exception as exc:  # JWTError or any decode issue
        logger.info("JWT decode failed: %s", exc)
        return None
    user_id = payload.get("sub")
    role = payload.get("role", Role.GUEST)
    username = payload.get("username", "")
    if not user_id:
        return None
    return Principal(user_id=str(user_id), username=str(username), role=str(role))


def get_current_user_optional(
    authorization: Optional[str] = Header(default=None),
    db=Depends(get_db),
) -> Optional[Principal]:
    token = auth.extract_bearer(authorization)
    return _load_principal_from_token(token, db)


def get_current_user(
    authorization: Optional[str] = Header(default=None),
    db=Depends(get_db),
) -> Principal:
    """Strict dependency: requires a valid Bearer JWT."""
    principal = get_current_user_optional(authorization, db)
    if principal is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token không hợp lệ hoặc thiếu. Đăng nhập để lấy JWT.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return principal


def require_role(*roles: str) -> Callable[..., Principal]:
    """Dependency factory: require the caller's role to be one of ``roles``."""
    allowed = set(roles)

    def _dep(principal: Principal = Depends(get_current_user)) -> Principal:
        if principal.role not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{principal.role}' không có quyền thực hiện tác vụ này.",
            )
        return principal

    return _dep


require_admin = require_role(Role.ADMIN)


__all__ = [
    "Role",
    "Principal",
    "LEGAL_TOOLS",
    "WEB_TOOLS",
    "MEMORY_TOOLS",
    "UTILITY_TOOLS",
    "GUEST_TOOLS",
    "SENSITIVE_TOOLS",
    "ROLE_PERMISSIONS",
    "APPROVAL_EXEMPT_ROLES",
    "allowed_tool_names",
    "filter_tools_by_policy",
    "is_tool_allowed",
    "needs_approval",
    "get_current_user",
    "get_current_user_optional",
    "require_role",
    "require_admin",
]