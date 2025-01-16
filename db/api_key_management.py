import sqlite3
import secrets
import base64
import time
import hashlib
import argparse
from typing import Tuple, Optional
from datetime import datetime
from pathlib import Path
import os
from dotenv import load_dotenv


class APIKeyGenerator:
    # Key format: sk-gb-{api_version}-{base64_id}-{signature}
    API_VERSION = "v1"
    PREFIX = "sk-gb"  # sk = secret key, gb = greenbit
    KEY_LENGTH = 128  # Total approximate length after encoding

    @classmethod
    def generate_api_key(cls, user_id: Optional[int] = None) -> Tuple[str, str]:
        """
        Generate a new API key and its hashed version for storage.
        Returns (api_key, hashed_key)
        """
        # Generate timestamp component
        timestamp = int(time.time())

        # Generate random bytes for uniqueness (32 bytes)
        random_bytes = secrets.token_bytes(32)

        # Create unique identifier component
        identifier = base64.urlsafe_b64encode(
            random_bytes + timestamp.to_bytes(8, 'big')
        ).decode('utf-8').rstrip('=')

        # Generate a secure random component (48 bytes)
        secret = secrets.token_bytes(48)
        secret_b64 = base64.urlsafe_b64encode(secret).decode('utf-8').rstrip('=')

        # Create signature component
        signature_data = f"{identifier}{timestamp}{user_id or ''}"
        signature = cls._create_signature(signature_data, secret)

        # Construct the full API key
        api_key = f"{cls.PREFIX}-{cls.API_VERSION}-{identifier}-{signature}"

        # Create hash for storage
        hashed_key = cls._hash_key(api_key)

        return api_key, hashed_key

    @staticmethod
    def _create_signature(data: str, secret: bytes) -> str:
        """Create a signature for the API key components."""
        hmac = hashlib.blake2b(
            key=secret,
            digest_size=32,
            person=b"greenbit_api_key"
        )
        hmac.update(data.encode())
        return base64.urlsafe_b64encode(hmac.digest()).decode('utf-8').rstrip('=')

    @staticmethod
    def _hash_key(api_key: str) -> str:
        """Create a hash of the API key for storage."""
        return hashlib.blake2b(
            api_key.encode(),
            digest_size=32,
            salt=b"greenbit_storage",
            person=b"api_key_storage"
        ).hexdigest()


# Define tier limits
TIER_LIMITS = {
    'free': {
        'rpm_limit': 10,        # Requests per minute
        'tpm_limit': 10000,     # Tokens per minute
        'concurrent_requests': 100,# Concurrent requests allowed
        'max_tokens': 4096      # Maximum tokens per request
    },
    'basic': {
        'rpm_limit': 60,        # Requests per minute
        'tpm_limit': 40000,     # Tokens per minute
        'concurrent_requests': 500,# Concurrent requests allowed
        'max_tokens': 8192      # Maximum tokens per request
    },
    'standard': {
        'rpm_limit': 250,       # Requests per minute
        'tpm_limit': 100000,    # Tokens per minute
        'concurrent_requests': 1500,# Concurrent requests allowed
        'max_tokens': 16384     # Maximum tokens per request
    },
    'premium': {
        'rpm_limit': 100000,      # Requests per minute
        'tpm_limit': 40000000,    # Tokens per minute
        'concurrent_requests': 500000,# Concurrent requests allowed
        'max_tokens': 32768000     # Maximum tokens per request
    }
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a GreenBit API key for a user",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Example usage:
                python api_key_management.py --name "John Doe" --email "john@example.com" --org "Example Inc" --tier premium
                or add the user information into an .env file:
                # .env
                LIBRA_USER_NAME="Haojin Yang"
                LIBRA_USER_EMAIL="haojin.yang@greenbit.ai"
                LIBRA_ORGANIZATION="GreenBitAI"
                LIBRA_API_TIER="standard"
                LIBRA_DB_PATH="db/greenbit.db"
                # load
                python api_key_management.py --env-file .env  # Load from environment file
            
            Available tiers:
              - free     (4K tokens, 10 RPM)
              - basic    (8K tokens, 60 RPM)
              - standard (16K tokens, 250 RPM)
              - premium  (32K tokens, 1000 RPM)
        """
    )

    # Add env file argument
    parser.add_argument(
        "--env-file",
        type=str,
        default=".env",
        help="Path to .env file (default: .env)"
    )

    parser.add_argument(
        "--db",
        type=str,
        default="db/greenbit.db",
        help="Path to database file (default:greenbit.db)"
    )

    # Required arguments
    parser.add_argument(
        "--name",
        required=True,
        help="User's full name"
    )
    parser.add_argument(
        "--email",
        required=True,
        help="User's email address"
    )
    parser.add_argument(
        "--org",
        required=True,
        help="User's organization name"
    )

    # Optional arguments
    parser.add_argument(
        "--tier",
        choices=['free', 'basic', 'standard', 'premium'],
        default='basic',
        help="API usage tier (default: basic)"
    )

    return parser.parse_args()


def create_api_key(name: str, email: str, organization: str, tier: str = 'basic', db_path: str = 'db/greenbit.db') -> str:
    """Create a new API key with user information and store it in the database."""
    if tier not in TIER_LIMITS:
        raise ValueError(f"Invalid tier: {tier}. Must be one of: {', '.join(TIER_LIMITS.keys())}")

    try:
        # Ensure database directory exists
        db_dir = Path("db")
        db_dir.mkdir(exist_ok=True)

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Check if email already exists
            cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
            existing_user = cursor.fetchone()

            if existing_user:
                user_id = existing_user[0]
                # Update existing user information
                cursor.execute("""
                    UPDATE users 
                    SET name = ?, organization = ?
                    WHERE id = ?
                """, (name, organization, user_id))
            else:
                # Insert new user
                cursor.execute("""
                    INSERT INTO users (name, email, organization)
                    VALUES (?, ?, ?)
                    RETURNING id
                """, (name, email, organization))
                user_id = cursor.fetchone()[0]

            # Generate API key
            api_key, hashed_key = APIKeyGenerator.generate_api_key(user_id)

            # Get tier limits
            tier_limits = TIER_LIMITS[tier]

            # Insert API key information
            cursor.execute("""
                INSERT INTO api_keys (
                    user_id,
                    api_key_hash,
                    name,
                    email,
                    organization,
                    tier,
                    rpm_limit,
                    tpm_limit,
                    concurrent_requests,
                    max_tokens,
                    permissions,
                    is_active,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                hashed_key,
                name,
                email,
                organization,
                tier,
                tier_limits['rpm_limit'],
                tier_limits['tpm_limit'],
                tier_limits['concurrent_requests'],
                tier_limits['max_tokens'],
                'completion,chat',  # Default permissions
                True,              # Active by default
                datetime.utcnow().isoformat()
            ))

            conn.commit()
            return api_key

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        raise
    except Exception as e:
        print(f"Error: {e}")
        raise


def display_key_info(api_key: str, tier: str):
    """Display formatted API key information."""
    tier_limits = TIER_LIMITS[tier]

    print("\n" + "=" * 50)
    print("API KEY GENERATED SUCCESSFULLY")
    print("=" * 50)
    print(f"\nAPI Key: {api_key}")
    print(f"\nTier: {tier.upper()}")
    print("\nLimits:")
    print(f"  - Max tokens per request: {tier_limits['max_tokens']}")
    print(f"  - Requests per minute: {tier_limits['rpm_limit']}")
    print(f"  - Tokens per minute: {tier_limits['tpm_limit']:,}")
    print(f"  - Concurrent requests: {tier_limits['concurrent_requests']}")
    print("\nIMPORTANT: Store this API key safely. It won't be shown again.")
    print("=" * 50 + "\n")


def load_config(args):
    """Load configuration from env file and/or command line arguments."""
    # Load .env file if it exists
    env_path = Path(args.env_file)
    if env_path.exists():
        load_dotenv(env_path)

    # Priority: command line args > env vars > defaults
    config = {
        'name': args.name or os.getenv('LIBRA_USER_NAME'),
        'email': args.email or os.getenv('LIBRA_USER_EMAIL'),
        'organization': args.org or os.getenv('LIBRA_ORGANIZATION'),
        'tier': args.tier or os.getenv('LIBRA_API_TIER', 'basic'),
        'db_path': args.db or os.getenv('LIBRA_DB_PATH', 'db/greenbit.db')
    }

    # Validate required fields
    missing_fields = [k for k, v in config.items() if v is None]
    if missing_fields:
        raise ValueError(f"Missing required configuration: {', '.join(missing_fields)}")

    return config


def main():
    """Main function for API key creation"""
    try:
        # Parse command line arguments
        args = parse_args()

        # Load configuration
        config = load_config(args)

        # Create a new API key
        api_key = create_api_key(
            name=config['name'],
            email=config['email'],
            organization=config['organization'],
            tier=config['tier'],
            db_path=config['db_path'],
        )

        # Display API key information
        display_key_info(api_key, config['tier'])

        # Optionally save to .env file
        if args.env_file:
            with open(args.env_file, 'a') as f:
                f.write(f"\nLIBRA_API_KEY={api_key}")
                print(f"\nAPI key has been added to {args.env_file}")

    except Exception as e:
        print(f"\nError: Failed to create API key: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
