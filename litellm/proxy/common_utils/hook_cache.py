from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Final

from litellm.caching.dual_cache import DualCache

if TYPE_CHECKING:
    from opentelemetry.trace import Span

    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache


class HookCache(DualCache):
    """
    The ``cache`` a pre-call hook is handed. Hooks used to be handed the auth cache itself, and a hook written against
    that reads keys, users and teams out of it by cache key and deletes them from it. So a read answers with this
    worker's copy of an auth object when it holds one and with the hook's own state otherwise, and a delete drops the
    key from both. A write is the hook's own state.

    The two share one Redis keyspace, which the auth cache owns: an auth object this worker does not hold is read from
    Redis as any other key of this cache, and a delete reaches Redis through the auth cache alone.
    """

    def __init__(self, auth_cache: Callable[[], UserApiKeyCache], default_in_memory_ttl: float | None = None) -> None:
        super().__init__(default_in_memory_ttl=default_in_memory_ttl)
        self._auth_cache: Final = auth_cache

    def get_cache(
        self, key: str, parent_otel_span: Span | None = None, local_only: bool = False, **kwargs: object
    ) -> object:
        held: Final[object] = self._auth_cache().get_local_by_key(key)
        if held is not None:
            return held
        return super().get_cache(key, parent_otel_span=parent_otel_span, local_only=local_only, **kwargs)

    async def async_get_cache(
        self, key: str, parent_otel_span: Span | None = None, local_only: bool = False, **kwargs: object
    ) -> object:
        held: Final[object] = self._auth_cache().get_local_by_key(key)
        if held is not None:
            return held
        return await super().async_get_cache(key, parent_otel_span=parent_otel_span, local_only=local_only, **kwargs)

    def delete_cache(self, key: str) -> None:
        self.in_memory_cache.delete_cache(key)
        self._auth_cache().delete_cache(key)

    async def async_delete_cache(self, key: str) -> None:
        self.in_memory_cache.delete_cache(key)
        await self._auth_cache().async_delete_cache(key)

    async def async_delete_cache_keys(self, keys: Sequence[str]) -> None:
        for key in keys:
            self.in_memory_cache.delete_cache(key)
        await self._auth_cache().async_delete_cache_keys(keys)
