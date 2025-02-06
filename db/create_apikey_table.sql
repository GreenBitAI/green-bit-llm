-- z.B. sqlite3 greenbit.db < create_tables.sql

-- drop tables if exists
--DROP TABLE IF EXISTS api_keys;
--DROP TABLE IF EXISTS users;

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    organization TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create API keys table
CREATE TABLE IF NOT EXISTS api_keys (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    api_key_hash TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    organization TEXT,
    tier VARCHAR(20) DEFAULT 'basic',
    rpm_limit INTEGER DEFAULT 60,
    tpm_limit INTEGER DEFAULT 40000,
    concurrent_requests INTEGER DEFAULT 5,
    max_tokens INTEGER DEFAULT 32768,
    permissions TEXT DEFAULT 'completion,chat',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_used_at TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id),
    CHECK (is_active IN (0, 1))
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_api_key_hash ON api_keys(api_key_hash);
CREATE INDEX IF NOT EXISTS idx_user_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_last_used ON api_keys(last_used_at);