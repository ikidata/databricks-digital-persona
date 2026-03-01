"""Database connectivity and SQL query execution against Databricks."""

import logging
import os
from databricks import sql as databricks_sql

logger = logging.getLogger(__name__)
from databricks.sdk.core import Config, oauth_service_principal
from app.config import (
    get_databricks_host,
    get_auth_method,
    get_token,
    get_client_id,
    get_client_secret,
    get_warehouse_http_path,
)


def _make_credentials_provider():
    """Create an OAuth M2M credentials provider using service principal.

    Only used when auth_method is 'service_principal'.
    """
    config = Config(
        host=get_databricks_host(),
        client_id=get_client_id(),
        client_secret=get_client_secret(),
    )
    return oauth_service_principal(config)


def get_auth_token():
    """Get a Bearer token string for REST API calls.

    Returns the token value (string) regardless of auth method.
    Priority: DATABRICKS_TOKEN env var (injected by Apps runtime) → config.yaml.
    For service_principal: exchanges client credentials for an OAuth token.
    For pat: returns the PAT directly.
    """
    # Databricks Apps runtime injects DATABRICKS_TOKEN for the app service principal
    env_token = os.environ.get("DATABRICKS_TOKEN")
    if env_token:
        return env_token
    try:
        if get_auth_method() == "pat":
            return get_token()
        # Service principal: use the credentials provider to get a fresh token
        creds_provider = _make_credentials_provider()
        oauth_token = creds_provider()
        return oauth_token.token
    except Exception:
        raise RuntimeError(
            "No auth token available. Set DATABRICKS_TOKEN env var or configure config.yaml."
        )


def get_connection(user_token=None, timeout=None):
    """Get a Databricks SQL connection.

    Args:
        user_token: If provided, use this token (on-behalf-of-user).
                   Otherwise, use the configured auth method from config.yaml.
        timeout:   Optional socket timeout in seconds for this connection.
    """
    http_path = get_warehouse_http_path()
    extra = {"_socket_timeout": timeout} if timeout else {}

    if user_token:
        # On-behalf-of-user: use Databricks SDK Config to resolve the host
        # correctly in Databricks Apps runtime (reads DATABRICKS_HOST env var)
        cfg = Config()
        server_hostname = cfg.host.replace("https://", "").replace("http://", "")
        logger.info("OBO connection: host=%s http_path=%s token_len=%d",
                     server_hostname, http_path, len(user_token))
        return databricks_sql.connect(
            server_hostname=server_hostname,
            http_path=http_path,
            access_token=user_token,
            **extra,
        )

    # Non-OBO path: resolve host and pick the best auth method.
    host = get_databricks_host()
    server_hostname = host.replace("https://", "").replace("http://", "")

    # Databricks Apps runtime: DATABRICKS_TOKEN is set for the app service principal.
    # Use it directly — this takes priority over config.yaml settings.
    env_token = os.environ.get("DATABRICKS_TOKEN")
    if env_token:
        logger.info("SP connection via DATABRICKS_TOKEN: host=%s http_path=%s",
                     server_hostname, http_path)
        return databricks_sql.connect(
            server_hostname=server_hostname,
            http_path=http_path,
            access_token=env_token,
            **extra,
        )

    if get_auth_method() == "pat":
        return databricks_sql.connect(
            server_hostname=server_hostname,
            http_path=http_path,
            access_token=get_token(),
            **extra,
        )

    # Service principal OAuth M2M (requires client_id/client_secret in config.yaml)
    return databricks_sql.connect(
        server_hostname=server_hostname,
        http_path=http_path,
        credentials_provider=_make_credentials_provider,
        **extra,
    )


def execute_query(query, params=None, user_token=None):
    """Execute a SQL query and return results as list of dicts."""
    auth_mode = "obo_token" if user_token else ("pat" if get_auth_method() == "pat" else "service_principal")
    logger.info("execute_query auth_mode=%s has_token=%s | query_preview=%.80s...",
                auth_mode, bool(user_token), query.strip())
    conn = get_connection(user_token=user_token)
    try:
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            logger.debug("execute_query returned %d rows", len(rows))
            return [dict(zip(columns, row)) for row in rows]
    finally:
        conn.close()


def check_table_access(table_name, user_token=None):
    """Check if we can access a specific system table. Returns (accessible, row_count, error)."""
    try:
        results = execute_query(
            f"SELECT 1 AS ok FROM {table_name} LIMIT 1",
            user_token=user_token,
        )
        return True, 1 if results else 0, None
    except Exception as e:
        return False, 0, str(e)


def check_all_tables_access(table_names, user_token=None):
    """Check access to multiple system tables in parallel.

    Runs all checks concurrently (max 60s timeout per check) so the total
    wall-clock time ≈ slowest single check, not the sum of all checks.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _check_one(table):
        try:
            conn = get_connection(user_token=user_token, timeout=60)
            try:
                with conn.cursor() as cursor:
                    cursor.execute(f"SELECT 1 AS ok FROM {table} LIMIT 1")
                    rows = cursor.fetchall()
                    return {"table": table, "accessible": True,
                            "row_count": len(rows), "error": None}
            finally:
                conn.close()
        except Exception as e:
            return {"table": table, "accessible": False,
                    "row_count": 0, "error": str(e)}

    with ThreadPoolExecutor(max_workers=len(table_names)) as pool:
        futures = {pool.submit(_check_one, t): t for t in table_names}
        result_map = {}
        for future in as_completed(futures):
            r = future.result()
            result_map[r["table"]] = r

    # Preserve input order
    return [result_map[t] for t in table_names]
