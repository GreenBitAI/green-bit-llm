import sqlite3
import hashlib
from typing import Optional
from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN
import logging
from pathlib import Path
from dotenv import load_dotenv
import os

from .rate_limiter import RateLimiter

# API key header
API_KEY_HEADER = APIKeyHeader(name="X-Api-Key", auto_error=True)


class APIKeyAuth:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.rate_limiter = RateLimiter()
        self.logger = logging.getLogger("greenbit_server")

    def get_db_connection(self):
        return sqlite3.connect(self.db_path)

    def _hash_key(self, api_key: str) -> str:
        """Hash the API key for database lookup."""
        return hashlib.blake2b(
            api_key.encode(),
            digest_size=32,
            salt=b"greenbit_storage",
            person=b"api_key_storage"
        ).hexdigest()

    def validate_api_key(self, api_key: str) -> dict:
        """Validate API key and return user info if valid."""
        try:
            hashed_key = self._hash_key(api_key)

            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        ak.user_id,
                        u.name,
                        u.email,
                        u.organization,
                        ak.tier,
                        ak.rpm_limit,
                        ak.tpm_limit,
                        ak.concurrent_requests,
                        ak.max_tokens,
                        ak.permissions,
                        ak.is_active
                    FROM api_keys ak
                    JOIN users u ON ak.user_id = u.id
                    WHERE ak.api_key_hash = ?
                """, (hashed_key,))
                result = cursor.fetchone()

                if not result:
                    raise HTTPException(
                        status_code=HTTP_403_FORBIDDEN,
                        detail="Invalid API key"
                    )

                user_info = {
                    "user_id": result[0],
                    "name": result[1],
                    "email": result[2],
                    "organization": result[3],
                    "tier": result[4],
                    "rpm_limit": result[5],
                    "tpm_limit": result[6],
                    "concurrent_requests": result[7],
                    "max_tokens": result[8],
                    "permissions": result[9].split(','),
                    "is_active": bool(result[10])
                }

                if not user_info["is_active"]:
                    raise HTTPException(
                        status_code=HTTP_403_FORBIDDEN,
                        detail="API key is inactive"
                    )

                # Update last_used_at
                cursor.execute("""
                    UPDATE api_keys 
                    SET last_used_at = CURRENT_TIMESTAMP 
                    WHERE api_key_hash = ?
                """, (hashed_key,))
                conn.commit()

                return user_info

        except sqlite3.Error as e:
            self.logger.error(f"Database error during API key validation: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")

    async def check_rate_limits(self, api_key: str, user_info: dict, estimated_tokens: Optional[int] = None):
        """Check all rate limits."""
        try:
            # Check RPM
            self.rate_limiter.check_rate_limit(api_key, user_info["rpm_limit"])

            # Check TPM if tokens are provided
            if estimated_tokens is not None:
                self.rate_limiter.check_token_limit(
                    api_key,
                    estimated_tokens,
                    user_info["tpm_limit"]
                )

            # Try to acquire concurrent request slot
            await self.rate_limiter.acquire_concurrent_request(
                api_key,
                user_info["concurrent_requests"]
            )

        except Exception as e:
            self.logger.error(f"Rate limit check failed: {str(e)}")
            raise

    def check_permissions(self, user_info: dict, endpoint_type: str):
        """Check if the user has permission to access the endpoint."""
        if endpoint_type not in user_info["permissions"]:
            raise HTTPException(
                status_code=HTTP_403_FORBIDDEN,
                detail=f"No permission to access {endpoint_type} endpoint"
            )

    def check_token_limit(self, user_info: dict, requested_tokens: int):
        """Check if the requested tokens are within the user's limit."""
        if requested_tokens > user_info["max_tokens"]:
            raise HTTPException(
                status_code=HTTP_403_FORBIDDEN,
                detail=f"Requested tokens ({requested_tokens}) exceed maximum allowed ({user_info['max_tokens']})"
            )


def load_api_key(env_file: str = None) -> str:
    """Load API key from environment file or environment variables."""
    if env_file and Path(env_file).exists():
        load_dotenv(env_file)

    api_key = os.getenv("LIBRA_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="API key not found in environment"
        )

    return api_key


async def get_api_key_auth(
        api_key: str = Security(API_KEY_HEADER),
        env_file: str = None,
        estimated_tokens: Optional[int] = None
) -> dict:
    """FastAPI dependency for API key authentication."""
    # If no API key in header, try to load from environment
    if not api_key:
        api_key = load_api_key(env_file)

    db_path = os.getenv("LIBRA_DB_PATH", str(Path(__file__).parent.parent.parent.parent / "db" / "greenbit.db"))
    auth_handler = APIKeyAuth(db_path)
    user_info = auth_handler.validate_api_key(api_key)
    await auth_handler.check_rate_limits(api_key, user_info, estimated_tokens)
    return user_info