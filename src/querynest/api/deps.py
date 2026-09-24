"""Shared FastAPI dependencies: the current user (from the JWT) and the LLM router.

Depends(...) is FastAPI's dependency injection, like NestJS providers + guards: an endpoint
declaring `user: User = Depends(current_user)` only runs for a valid logged-in user.
"""

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from querynest.appdb import User, session
from querynest.auth import decode_token
from querynest.llm import LLMRouter
from querynest.permissions import ROLES, UserContext

bearer = HTTPBearer(auto_error=False)
_router: LLMRouter | None = None


def llm_router() -> LLMRouter:
    global _router
    if _router is None:
        _router = LLMRouter()  # one router per process, so its response cache is shared
    return _router


def current_user(creds: HTTPAuthorizationCredentials | None = Depends(bearer)) -> User:
    if creds is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not logged in")
    try:
        user_id = decode_token(creds.credentials)
    except jwt.InvalidTokenError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Session expired, please log in again") from None
    with session() as s:
        user = s.get(User, user_id)
    if user is None or not user.active or user.role not in ROLES:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Account disabled")
    return user


def admin_user(user: User = Depends(current_user)) -> User:
    if not ROLES[user.role].is_admin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admins only")
    return user


def user_context(user: User, conversation_id: str | None = None) -> UserContext:
    return UserContext(username=user.username, role=user.role, user_id=user.id,
                       attributes=dict(user.attributes or {}), conversation_id=conversation_id)
