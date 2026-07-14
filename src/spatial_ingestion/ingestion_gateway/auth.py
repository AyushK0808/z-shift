from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256


@dataclass(frozen=True)
class AuthContext:
    subject: str
    scopes: tuple[str, ...] = ()


class AuthProvider:
    """Interface stub for production auth integration."""

    async def authenticate(
        self,
        authorization: str | None,
        client_host: str | None = None,
    ) -> AuthContext:
        if authorization:
            digest = sha256(authorization.encode("utf-8")).hexdigest()[:16]
            return AuthContext(subject=f"auth:{digest}", scopes=("ingest:write",))

        host = client_host or "unknown"
        return AuthContext(subject=f"anonymous:{host}", scopes=("ingest:write",))
