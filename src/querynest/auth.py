"""Login: password hashing (Argon2) and JWT access tokens.

JWT = a signed token the browser sends with every request ("Authorization: Bearer <token>").
Like Passport-JWT in NestJS: the server can verify it without a session store, because only
the server knows JWT_SECRET.
"""

from datetime import datetime, timedelta, timezone

import jwt
from pwdlib import PasswordHash

from querynest.config import settings

_hasher = PasswordHash.recommended()  # Argon2: slow on purpose, so stolen hashes are hard to crack


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return _hasher.verify(password, password_hash)


def _secret() -> str:
    if settings.jwt_secret is None:
        raise RuntimeError("JWT_SECRET missing in .env. Run `uv run setup-db` first.")
    return settings.jwt_secret.get_secret_value()


def create_token(user_id: int, username: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {"sub": str(user_id), "name": username, "iat": now,
               "exp": now + timedelta(minutes=settings.jwt_expire_minutes)}
    return jwt.encode(payload, _secret(), algorithm="HS256")


def decode_token(token: str) -> int:
    """Return the user id, or raise jwt.InvalidTokenError (expired, tampered, malformed)."""
    payload = jwt.decode(token, _secret(), algorithms=["HS256"])
    return int(payload["sub"])
