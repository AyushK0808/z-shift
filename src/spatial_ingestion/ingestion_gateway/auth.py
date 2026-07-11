from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AuthContext:
    subject: str
    scopes: tuple[str, ...] = ()


class AuthProvider:
    """Interface stub for production auth integration."""

    async def authenticate(self, authorization: str | None) -> AuthContext:
        if authorization:
            return AuthContext(subject="authenticated-client", scopes=("ingest:write",))
        return AuthContext(subject="anonymous", scopes=("ingest:write",))

