import ipaddress
from typing import Mapping


def is_loopback_host(host: str) -> bool:
    normalized = (host or '').strip().lower()
    if normalized == 'localhost':
        return True
    if normalized.startswith('[') and normalized.endswith(']'):
        normalized = normalized[1:-1]
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def require_auth_for_public_bind(
    host: str,
    tokens: Mapping[str, str],
    allow_unauthenticated_public: bool = False,
) -> None:
    if is_loopback_host(host) or allow_unauthenticated_public:
        return
    missing = [name for name, token in tokens.items() if not str(token or '').strip()]
    if not missing:
        return
    missing_text = ', '.join(missing)
    raise ValueError(
        f'Refusing to bind an unauthenticated API to public host {host!r}. '
        f'Missing token(s): {missing_text}. Bind to a loopback host, configure the token(s), '
        'or explicitly pass --allow-unauthenticated-public.'
    )
