"""Seed a default admin user from env on startup (best-effort, idempotent).

Env:
- SEED_ADMIN_USERNAME (default "admin")
- SEED_ADMIN_PASSWORD (default "change-me" — refused in production unless
  explicitly set; here we only warn)

Call ``seed_admin()`` at app startup. If the admin already exists, no-op.
"""
from __future__ import annotations

import logging
import os

import auth
from models import create_user, get_user_by_username
from rbac import Role

logger = logging.getLogger(__name__)


def seed_admin() -> None:
    username = os.getenv("SEED_ADMIN_USERNAME", "admin")
    password = os.getenv("SEED_ADMIN_PASSWORD", "change-me")
    try:
        if get_user_by_username(username) is not None:
            return
        if password == "change-me":
            logger.warning(
                "Seeding admin with default password 'change-me' — set "
                "SEED_ADMIN_PASSWORD before production use."
            )
        create_user(username, auth.hash_password(password), role=Role.ADMIN)
        logger.info("Seeded default admin user %r (role=admin).", username)
    except Exception as exc:
        logger.warning("Seed admin failed (non-fatal): %s", exc)


if __name__ == "__main__":  # pragma: no cover
    seed_admin()