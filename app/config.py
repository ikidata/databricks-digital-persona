"""Load configuration from config.yaml."""

import os
import yaml

_config = None


def get_config():
    """Load and cache config from config.yaml.

    Returns an empty dict when the file does not exist (e.g. Databricks Apps
    runtime where configuration comes from env vars and app.yaml).
    """
    global _config
    if _config is not None:
        return _config

    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    if not os.path.exists(config_path):
        _config = {}
        return _config
    with open(config_path, "r", encoding="utf-8") as f:
        _config = yaml.safe_load(f)
    return _config


def _ensure_scheme(host):
    """Ensure the host URL has an https:// scheme."""
    if host and not host.startswith(("https://", "http://")):
        host = "https://" + host
    return host


def get_databricks_host():
    # Prefer env var (auto-injected by Databricks Apps runtime)
    env_host = os.environ.get("DATABRICKS_HOST")
    if env_host:
        return _ensure_scheme(env_host.rstrip("/"))
    cfg = get_config()
    return _ensure_scheme(cfg.get("databricks", {}).get("host", "").rstrip("/"))


def get_auth_method():
    """Return the configured auth method: 'service_principal' or 'pat'.

    Defaults to 'service_principal' when config.yaml is absent (Databricks Apps).
    """
    cfg = get_config()
    method = cfg.get("databricks", {}).get("auth_method", "service_principal")
    if method not in ("service_principal", "pat"):
        raise ValueError(
            f"Invalid auth_method '{method}' in config.yaml. "
            "Must be 'service_principal' or 'pat'."
        )
    return method


def get_token():
    """Return the PAT token from config.yaml or DATABRICKS_TOKEN env var."""
    env_token = os.environ.get("DATABRICKS_TOKEN")
    if env_token:
        return env_token
    cfg = get_config()
    token = cfg.get("databricks", {}).get("token", "")
    if not token:
        raise ValueError(
            "auth_method is 'pat' but no token found. "
            "Set databricks.token in config.yaml or DATABRICKS_TOKEN env var."
        )
    return token


def get_client_id():
    cfg = get_config()
    return cfg.get("databricks", {}).get("client_id", "")


def get_client_secret():
    cfg = get_config()
    return cfg.get("databricks", {}).get("client_secret", "")


def get_warehouse_http_path():
    # WAREHOUSE_ID is populated by Databricks Apps runtime from the
    # resource declared in app.yaml / databricks.yml (valueFrom).
    warehouse_id = os.environ.get("WAREHOUSE_ID")
    if warehouse_id:
        return f"/sql/1.0/warehouses/{warehouse_id}"
    # Legacy fallback: full HTTP path env var or config.yaml
    env_path = os.environ.get("DATABRICKS_WAREHOUSE_HTTP_PATH")
    if env_path:
        return env_path
    cfg = get_config()
    return cfg.get("databricks", {}).get("sql_warehouse_http_path", "")


def get_warehouse_id():
    cfg = get_config()
    return cfg.get("databricks", {}).get("sql_warehouse_id", "")


_DEFAULT_LOOKBACK_DAYS = 90

_DEFAULT_SYSTEM_TABLES = [
    "system.access.audit",
    "system.query.history",
    "system.compute.clusters",
    "system.lakeflow.jobs",
    "system.lakeflow.job_run_timeline",
    "system.access.table_lineage",
    "system.access.column_lineage",
    "system.billing.usage",
]


def get_lookback_days():
    cfg = get_config()
    return cfg.get("persona", {}).get("lookback_days", _DEFAULT_LOOKBACK_DAYS)


def get_system_tables():
    cfg = get_config()
    return cfg.get("system_tables", _DEFAULT_SYSTEM_TABLES)


def get_chat_model_endpoint():
    # Prefer env var (set by databricks.yml / app.yaml), fall back to config.yaml
    env_ep = os.environ.get("MODEL_SERVING_ENDPOINT")
    if env_ep:
        return env_ep
    cfg = get_config()
    chat = cfg.get("chat", {})
    return chat.get("model_serving_endpoint", "databricks-claude-sonnet-4-6")


_DEFAULT_CHAT_MAX_TOKENS = 1024


def get_chat_max_tokens():
    cfg = get_config()
    chat = cfg.get("chat", {})
    return chat.get("max_tokens", _DEFAULT_CHAT_MAX_TOKENS)
