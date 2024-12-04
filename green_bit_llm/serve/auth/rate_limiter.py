from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from fastapi import HTTPException
from starlette.status import HTTP_429_TOO_MANY_REQUESTS
import asyncio
from collections import defaultdict


class RateLimiter:
    def __init__(self):
        self._request_times: Dict[str, List[datetime]] = defaultdict(list)
        self._token_counts: Dict[str, List[Tuple[datetime, int]]] = defaultdict(list)
        self._concurrent_requests: Dict[str, int] = defaultdict(int)
        self._locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def acquire_concurrent_request(self, api_key: str, limit: int):
        """Try to acquire a concurrent request slot."""
        async with self._locks[api_key]:
            if self._concurrent_requests[api_key] >= limit:
                raise HTTPException(
                    status_code=HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Maximum concurrent requests ({limit}) exceeded"
                )
            self._concurrent_requests[api_key] += 1

    async def release_concurrent_request(self, api_key: str):
        """Release a concurrent request slot."""
        async with self._locks[api_key]:
            if self._concurrent_requests[api_key] > 0:
                self._concurrent_requests[api_key] -= 1

    def check_rate_limit(self, api_key: str, rpm_limit: int):
        """Check requests per minute limit."""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)

        # Clean old entries
        self._request_times[api_key] = [
            time for time in self._request_times[api_key]
            if time > minute_ago
        ]

        # Check RPM limit
        if len(self._request_times[api_key]) >= rpm_limit:
            raise HTTPException(
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Maximum {rpm_limit} requests per minute."
            )

        self._request_times[api_key].append(now)

    def check_token_limit(self, api_key: str, new_tokens: int, tpm_limit: int):
        """Check tokens per minute limit."""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)

        # Clean old entries
        self._token_counts[api_key] = [
            (time, count) for time, count in self._token_counts[api_key]
            if time > minute_ago
        ]

        # Calculate current token usage
        current_tpm = sum(count for _, count in self._token_counts[api_key])

        # Check TPM limit
        if current_tpm + new_tokens > tpm_limit:
            raise HTTPException(
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Token rate limit exceeded. Maximum {tpm_limit} tokens per minute."
            )

        self._token_counts[api_key].append((now, new_tokens))