"""Security primitives for the Vietnamese Legal Chatbot backend.

Provides:
- Safe data-file path resolution (prevents path traversal on /data/import).
- API-key dependency gating admin/destructive endpoints.
- CORS allowlist configuration.

Security notes:
- Data import is restricted to a configurable data directory; user-supplied
  paths are resolved and checked to stay within that directory (no `..`
  escapes, no absolute paths outside the allowlist).
- Admin endpoints require `X-API-Key` matching `ADMIN_API_KEY` env var.
  When `ADMIN_API_KEY` is unset, admin endpoints are REFUSED unless
  ALLOW_UNSAFE_ADMIN=1 (dev convenience only).
"""
import logging
import os
import re
import secrets
from pathlib import Path
from typing import Optional

from fastapi import Header, HTTPException, status

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_DIR = REPO_ROOT / "data"


def _get_data_dir() -> Path:
    env_dir = os.getenv("IMPORT_DATA_DIR")
    base = Path(env_dir).resolve() if env_dir else _DEFAULT_DATA_DIR.resolve()
    base.mkdir(parents=True, exist_ok=True)
    return base


def resolve_safe_data_path(user_path: Optional[str]) -> Path:
    """Resolve a user-supplied path against the configured data directory.

    Raises HTTPException(400) if the resolved path escapes the data dir or
    HTTPException(404) if the file does not exist. Prevents path traversal
    (e.g. ``../../etc/shadow``).
    """
    base = _get_data_dir()

    if not user_path:
        default_file = os.getenv("IMPORT_DATA_FILE_PATH")
        candidate = Path(default_file).resolve() if default_file else base / "train.jsonl"
    else:
        raw = Path(user_path)
        if raw.is_absolute():
            candidate = raw.resolve()
        else:
            candidate = (base / raw).resolve()

    try:
        candidate.relative_to(base)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Đường dẫn file nằm ngoài thư mục dữ liệu được phép.",
        )

    if not candidate.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Không tìm thấy file: {candidate.name}",
        )
    return candidate


def _admin_key_configured() -> bool:
    return bool(os.getenv("ADMIN_API_KEY"))


def _unsafe_admin_allowed() -> bool:
    return os.getenv("ALLOW_UNSAFE_ADMIN", "0") == "1"


async def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """FastAPI dependency: gates admin/destructive endpoints behind an API key.

    - ADMIN_API_KEY set: request must send matching X-API-Key header.
    - ADMIN_API_KEY unset: allowed only if ALLOW_UNSAFE_ADMIN=1 (dev); otherwise
      refused so open admin endpoints never ship.
    """
    expected = os.getenv("ADMIN_API_KEY")
    if expected:
        if not x_api_key or not secrets.compare_digest(x_api_key, expected):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key không hợp lệ hoặc thiếu.",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        return

    if _unsafe_admin_allowed():
        logger.warning(
            "Admin endpoint gọi mà không có ADMIN_API_KEY (ALLOW_UNSAFE_ADMIN=1). "
            "KHÔNG dùng trong production."
        )
        return

    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Admin endpoint chưa cấu hình ADMIN_API_KEY. Thiết lập biến môi trường để bật.",
    )


def get_cors_origins() -> list[str]:
    """Return CORS allowlist from env. Empty list = deny all cross-origin."""
    raw = os.getenv("CORS_ALLOWED_ORIGINS", "")
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    if not origins:
        logger.warning(
            "CORS_ALLOWED_ORIGINS chưa cấu hình. Cross-origin requests sẽ bị từ chối."
        )
    return origins


def get_legal_collection_name(value: str) -> str:
    """Validate a Qdrant collection name: lowercase alnum + underscore/hyphen, len 1-64."""
    if not value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tên collection không được để trống.",
        )
    if not re.fullmatch(r"[a-z0-9_-]{1,64}", value):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tên collection chỉ gồm chữ thường, số, dấu gạch dưới/gạch ngang (tối đa 64 ký tự).",
        )
    return value