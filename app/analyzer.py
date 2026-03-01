"""Persona analysis engine - collects and processes data from system tables."""

import logging
import os
import threading

import requests as _requests

from app.db import execute_query
from app.config import get_lookback_days, get_databricks_host
from app.store import store

# ---------------------------------------------------------------------------
# Audit event filter: whitelist of services that represent genuine human work.
# Excludes automated background noise (tokenLogin, authzEval, getTable, etc.)
# that inflate counts by orders of magnitude and hide real patterns.
# ---------------------------------------------------------------------------
_MEANINGFUL_AUDIT_FILTER = """
  AND (
    service_name IN (
      'notebook', 'jobs', 'clusters', 'dashboards', 'alerts',
      'repos', 'workspace', 'sql', 'mlflowExperiment', 'mlflowModelRegistry',
      'aibiGenie', 'agentFramework', 'serving', 'featureStore',
      'vectorSearch', 'lakehouseMonitoring', 'permissions', 'secrets',
      'filesystem', 'files'
    )
    OR (
      service_name = 'unityCatalog'
      AND action_name IN (
        'createTable', 'deleteTable', 'alterTable',
        'createSchema', 'deleteSchema', 'createCatalog', 'deleteCatalog',
        'createVolume', 'deleteVolume', 'createFunction', 'deleteFunction',
        'updatePermissions', 'shareAsset', 'createConnection', 'deleteConnection'
      )
    )
  )
"""

logger = logging.getLogger(__name__)


def run_analysis(user_email, user_token=None):
    """Run full persona analysis in a background thread."""
    t = threading.Thread(
        target=_run_analysis_worker,
        args=(user_email, user_token),
        daemon=True,
    )
    t.start()
    return t


def _get_databricks_user_id(user_email, token=None):
    """Resolve Databricks numeric user ID for an email via SCIM API.

    Returns the string ID (e.g. "3502498566132612") or None on failure.
    Used to filter system.lakeflow.jobs which stores creator_id (numeric),
    not creator_user_name (null in practice).
    """
    host = get_databricks_host()
    if not token:
        try:
            from app.db import get_auth_token
            token = get_auth_token()
        except Exception:
            return None
    try:
        resp = _requests.get(
            f"{host}/api/2.0/preview/scim/v2/Users",
            params={"filter": f"userName eq '{user_email}'"},
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        if resp.ok:
            resources = resp.json().get("Resources", [])
            if resources:
                uid = resources[0].get("id")
                logger.info("Resolved user_id=%s for %s", uid, user_email)
                return uid
    except Exception as exc:
        logger.warning("SCIM user ID lookup failed for %s: %s", user_email, exc)
    return None


def _resolve_dashboard_names(ids, token=None):
    """Build a dashboard_id → display_name map using the Lakeview list API.

    Calls GET /api/2.0/lakeview/dashboards (paginated) to get all active
    dashboards, then returns a dict keyed by dashboard ID.  Dashboards that
    have been deleted will not appear in the list; callers should fall back
    to the raw ID for those.

    Args:
        ids: iterable of dashboard ID strings (used only for logging)
        token: Databricks auth token

    Returns:
        dict[str, str] mapping dashboard_id → display_name
    """
    host = get_databricks_host()
    if not token:
        try:
            from app.db import get_auth_token
            token = get_auth_token()
        except Exception:
            return {}
    name_map = {}
    params = {"page_size": 100}
    try:
        while True:
            resp = _requests.get(
                f"{host}/api/2.0/lakeview/dashboards",
                params=params,
                headers={"Authorization": f"Bearer {token}"},
                timeout=10,
            )
            if not resp.ok:
                logger.warning("Lakeview dashboard list returned HTTP %s", resp.status_code)
                break
            data = resp.json()
            for d in data.get("dashboards", []):
                did  = d.get("dashboard_id") or d.get("id", "")
                name = d.get("display_name") or d.get("name", "")
                if did and name:
                    name_map[did] = name
            next_token = data.get("next_page_token")
            if not next_token:
                break
            params = {"page_size": 100, "page_token": next_token}
        logger.info("Resolved %d dashboard names from Lakeview API", len(name_map))
    except Exception as exc:
        logger.warning("Could not resolve dashboard names: %s", exc)
    return name_map


def _run_analysis_worker(user_email, user_token=None):
    """Worker that collects all persona data from system tables."""
    lookback = get_lookback_days()
    logger.info("Starting analysis for %s | lookback=%d | has_obo_token=%s | token_len=%d",
                user_email, lookback, user_token is not None,
                len(user_token) if user_token else 0)
    persona = {}

    # Resolve the user's numeric Databricks ID once (needed for lakeflow.jobs)
    store.set_analysis_status(user_email, "running", 2, "Resolving user identity...")
    user_id = _get_databricks_user_id(user_email, token=user_token)
    if user_id:
        logger.info("Using Databricks user_id=%s for jobs queries", user_id)
    else:
        logger.warning("Could not resolve Databricks user_id for %s — jobs data may be empty", user_email)

    # Steps ordered lightest → heaviest so progress fills quickly at the start
    # and the heaviest queries run last (near 85-95%)
    # Weight roughly correlates to expected query time
    # Each step: (message, function, weight, error_key)
    # error_key is used to store errors under the correct persona key
    # so _build_data_health can show the actual error message
    steps = [
        ("Profiling compute and cluster usage...", _collect_compute_profile, 5, "compute"),
        ("Discovering job ownership & reliability...", _collect_job_profile, 8, "jobs"),
        ("Mapping collaboration network...", _collect_collaboration_profile, 7, "collaboration"),
        ("Analyzing SQL query patterns & complexity...", _collect_query_profile, 18, "queries"),
        ("Tracing data lineage & dependencies...", _collect_lineage_profile, 18, "lineage"),
        ("Scanning audit logs for activity patterns...", _collect_activity_profile, 35, "activity"),
        ("Comparing with workspace averages...", _collect_workspace_comparison, 5, "comparison"),
        ("Deep-diving into job runs & failures...", _collect_job_deep_dive, 8, "job_deep_dive"),
        ("Analyzing table lifecycle & governance...", _collect_table_governance_profile, 7, "table_governance"),
        ("Profiling GenAI & LLM usage...", _collect_genai_profile, 6, "genai"),
        ("Scanning Git & version control activity...", _collect_git_profile, 4, "git_activity"),
        ("Calculating cost & efficiency metrics...", _collect_cost_profile, 8, "cost"),
        ("Mapping dashboard activity...", _collect_dashboard_profile, 5, "dashboard_activity"),
        ("Building engagement timeline & streaks...", _collect_engagement_timeline, 8, "engagement"),
        ("Classifying persona archetype...", _build_summary, 9, "summary"),
    ]
    total_weight = sum(w for _, _, w, _ in steps)

    store.set_analysis_status(user_email, "running", 3, "Connecting to Databricks...")

    try:
        cumulative_weight = 0
        errors = []
        for i, (message, func, weight, error_key) in enumerate(steps):
            pct = int((cumulative_weight / total_weight) * 100)
            store.set_analysis_status(
                user_email, "running",
                max(pct, 3),  # never show less than 3%
                message,
            )
            try:
                if func == _build_summary:
                    result = func(user_email, lookback, token=user_token, accumulated=persona)
                elif func in (_collect_job_profile, _collect_job_deep_dive):
                    # Pass resolved user_id for jobs queries
                    result = func(user_email, lookback, user_token, user_id=user_id)
                else:
                    result = func(user_email, lookback, user_token)
                persona.update(result)
                logger.info("Step '%s' OK for %s — keys returned: %s",
                            func.__name__, user_email, list(result.keys()) if result else "none")
            except Exception as e:
                # Continue even if one section fails — store error under the
                # correct persona key so data_health can display it
                logger.warning("Step '%s' FAILED for %s: %s", func.__name__, user_email, e)
                persona[error_key] = {"error": str(e)}
                errors.append({"step": func.__name__, "key": error_key, "error": str(e)})
            cumulative_weight += weight

        # Build data health summary so the UI can show what worked and what didn't
        persona["data_health"] = _build_data_health(persona, errors)

        store.set_persona_data(user_email, persona)
        store.set_analysis_status(user_email, "complete", 100, "Analysis complete!")
    except Exception as e:
        store.set_analysis_status(user_email, "error", 0, f"Analysis failed: {e}")


def _collect_activity_profile(user_email, lookback, token=None):
    """Collect overall activity profile from audit logs.

    Visualisation queries (hourly, DoW, service breakdown, client tools) use
    _MEANINGFUL_AUDIT_FILTER to strip out automated background noise such as
    tokenLogin, authzEval, getTable, generateTemporaryTableCredential, etc.
    that can inflate counts by 100x and hide genuine work patterns.

    The raw summary (total events, active days) still counts everything so the
    full workspace percentile comparison remains accurate.
    """
    # Service distribution — meaningful events only
    services = execute_query(f"""
        SELECT service_name, COUNT(*) AS event_count
        FROM system.access.audit
        WHERE event_date >= CURRENT_DATE - INTERVAL {lookback} DAYS
          AND user_identity.email = '{user_email}'
          {_MEANINGFUL_AUDIT_FILTER}
        GROUP BY service_name
        ORDER BY event_count DESC
        LIMIT 20
    """, user_token=token)

    # Daily activity pattern — use event_date (partition column) instead of
    # DATE_TRUNC on event_time to avoid materialising timestamps
    daily_activity = execute_query(f"""
        SELECT event_date AS activity_date,
               COUNT(*) AS event_count
        FROM system.access.audit
        WHERE event_date >= CURRENT_DATE - INTERVAL {lookback} DAYS
          AND user_identity.email = '{user_email}'
          {_MEANINGFUL_AUDIT_FILTER}
        GROUP BY event_date
        ORDER BY event_date
    """, user_token=token)

    # Hourly pattern — meaningful events only
    hourly_pattern = execute_query(f"""
        SELECT HOUR(event_time) AS hour_of_day,
               COUNT(*) AS event_count
        FROM system.access.audit
        WHERE event_date >= CURRENT_DATE - INTERVAL {lookback} DAYS
          AND user_identity.email = '{user_email}'
          {_MEANINGFUL_AUDIT_FILTER}
        GROUP BY 1
        ORDER BY 1
    """, user_token=token)

    # Day of week pattern — meaningful events only
    dow_pattern = execute_query(f"""
        SELECT DAYOFWEEK(event_time) AS day_of_week,
               COUNT(*) AS event_count
        FROM system.access.audit
        WHERE event_date >= CURRENT_DATE - INTERVAL {lookback} DAYS
          AND user_identity.email = '{user_email}'
          {_MEANINGFUL_AUDIT_FILTER}
        GROUP BY 1
        ORDER BY 1
    """, user_token=token)

    # Top actions — meaningful events only
    top_actions = execute_query(f"""
        SELECT service_name, action_name, COUNT(*) AS action_count
        FROM system.access.audit
        WHERE event_date >= CURRENT_DATE - INTERVAL {lookback} DAYS
          AND user_identity.email = '{user_email}'
          {_MEANINGFUL_AUDIT_FILTER}
        GROUP BY service_name, action_name
        ORDER BY action_count DESC
        LIMIT 30
    """, user_token=token)

    # Client tools — meaningful events only (user_agent still available there)
    clients = execute_query(f"""
        SELECT
          CASE
            WHEN user_agent LIKE '%python%' OR user_agent LIKE '%sdk%' THEN 'SDK/Python'
            WHEN user_agent LIKE '%curl%' THEN 'cURL/CLI'
            WHEN user_agent LIKE '%Mozilla%' OR user_agent LIKE '%Chrome%' THEN 'Web Browser'
            WHEN user_agent LIKE '%DBConnect%' THEN 'DB Connect'
            WHEN user_agent LIKE '%spark%' THEN 'Spark'
            ELSE COALESCE(SUBSTRING(user_agent, 1, 30), 'Unknown')
          END AS client_type,
          COUNT(*) AS usage_count
        FROM system.access.audit
        WHERE event_date >= CURRENT_DATE - INTERVAL {lookback} DAYS
          AND user_identity.email = '{user_email}'
          AND user_agent IS NOT NULL
          {_MEANINGFUL_AUDIT_FILTER}
        GROUP BY 1
        ORDER BY usage_count DESC
        LIMIT 10
    """, user_token=token)

    # Total events & active days — ALL events (for accurate workspace comparison)
    # Use event_date (partition column) for active_days instead of DATE(event_time)
    summary_stats = execute_query(f"""
        SELECT
          COUNT(*) AS total_events,
          COUNT(DISTINCT event_date) AS active_days,
          MIN(event_time) AS first_event,
          MAX(event_time) AS last_event
        FROM system.access.audit
        WHERE event_date >= CURRENT_DATE - INTERVAL {lookback} DAYS
          AND user_identity.email = '{user_email}'
    """, user_token=token)

    return {
        "activity": {
            "services": _serialize(services),
            "daily_activity": _serialize(daily_activity),
            "hourly_pattern": _serialize(hourly_pattern),
            "dow_pattern": _serialize(dow_pattern),
            "top_actions": _serialize(top_actions),
            "clients": _serialize(clients),
            "summary": _serialize(summary_stats),
        }
    }


def _collect_query_profile(user_email, lookback, token=None):
    """Collect SQL query patterns from query history."""
    # Statement type distribution
    stmt_types = execute_query(f"""
        SELECT statement_type, COUNT(*) AS query_count,
               AVG(total_duration_ms) AS avg_duration_ms,
               SUM(produced_rows) AS total_rows
        FROM system.query.history
        WHERE executed_by = '{user_email}'
          AND start_time >= CURRENT_DATE - INTERVAL {lookback} DAYS
        GROUP BY statement_type
        ORDER BY query_count DESC
    """, user_token=token)

    # Query complexity distribution
    # ORDER BY uses MIN(total_duration_ms) as sort_key because Spark SQL does not
    # allow referencing a non-aggregated, non-grouped column in ORDER BY after GROUP BY.
    complexity = execute_query(f"""
        SELECT
          CASE
            WHEN total_duration_ms < 1000 THEN 'Quick (<1s)'
            WHEN total_duration_ms < 10000 THEN 'Moderate (1-10s)'
            WHEN total_duration_ms < 60000 THEN 'Heavy (10-60s)'
            WHEN total_duration_ms < 300000 THEN 'Very Heavy (1-5min)'
            ELSE 'Extreme (>5min)'
          END AS complexity,
          COUNT(*) AS count,
          MIN(total_duration_ms) AS sort_key
        FROM system.query.history
        WHERE executed_by = '{user_email}'
          AND start_time >= CURRENT_DATE - INTERVAL {lookback} DAYS
        GROUP BY 1
        ORDER BY sort_key ASC
    """, user_token=token)

    # Daily query volume
    daily_queries = execute_query(f"""
        SELECT DATE_TRUNC('day', start_time) AS query_date,
               COUNT(*) AS query_count,
               AVG(total_duration_ms) AS avg_duration_ms
        FROM system.query.history
        WHERE executed_by = '{user_email}'
          AND start_time >= CURRENT_DATE - INTERVAL {lookback} DAYS
        GROUP BY 1
        ORDER BY 1
    """, user_token=token)

    # Query source distribution — derive type from which struct field is populated
    # (query_source has no .type field; it contains nullable fields like
    #  notebook_id, job_info, dashboard_id, legacy_dashboard_id, alert_id)
    query_sources = execute_query(f"""
        SELECT
          CASE
            WHEN query_source.notebook_id IS NOT NULL THEN 'notebook'
            WHEN query_source.job_info IS NOT NULL THEN 'job'
            WHEN query_source.dashboard_id IS NOT NULL THEN 'dashboard'
            WHEN query_source.legacy_dashboard_id IS NOT NULL THEN 'dashboard'
            WHEN query_source.alert_id IS NOT NULL THEN 'alert'
            ELSE 'other'
          END AS source_type,
          COUNT(*) AS count
        FROM system.query.history
        WHERE executed_by = '{user_email}'
          AND start_time >= CURRENT_DATE - INTERVAL {lookback} DAYS
        GROUP BY 1
        ORDER BY count DESC
    """, user_token=token)

    # Error rate
    error_stats = execute_query(f"""
        SELECT
          COUNT(*) AS total_queries,
          SUM(CASE WHEN execution_status = 'FINISHED' THEN 1 ELSE 0 END) AS successful,
          SUM(CASE WHEN execution_status != 'FINISHED' THEN 1 ELSE 0 END) AS failed
        FROM system.query.history
        WHERE executed_by = '{user_email}'
          AND start_time >= CURRENT_DATE - INTERVAL {lookback} DAYS
    """, user_token=token)

    return {
        "queries": {
            "statement_types": _serialize(stmt_types),
            "complexity": _serialize(complexity),
            "daily_queries": _serialize(daily_queries),
            "query_sources": _serialize(query_sources),
            "error_stats": _serialize(error_stats),
        }
    }


def _collect_compute_profile(user_email, lookback, token=None):
    """Collect compute/cluster usage patterns."""
    # Column reference (verified against system.compute.clusters schema):
    #   cluster_source (not cluster_type), driver_node_type + worker_node_type (not node_type_id),
    #   min_autoscale_workers / max_autoscale_workers, dbr_version (not spark_version),
    #   change_time (not create_time for filtering — create_time can be null)
    clusters = execute_query(f"""
        SELECT cluster_id, cluster_name, cluster_source,
               driver_node_type, worker_node_type,
               worker_count, min_autoscale_workers, max_autoscale_workers,
               dbr_version, data_security_mode, change_date
        FROM system.compute.clusters
        WHERE owned_by = '{user_email}'
          AND change_date >= CURRENT_DATE - INTERVAL {lookback} DAYS
        ORDER BY change_date DESC
        LIMIT 50
    """, user_token=token)

    return {
        "compute": {
            "clusters": _serialize(clusters),
        }
    }


def _collect_job_profile(user_email, lookback, token=None, user_id=None):
    """Collect job ownership and execution patterns.

    system.lakeflow.jobs stores creator_id / run_as (numeric principal IDs).
    creator_user_name and run_as_user_name exist in the schema but are always
    NULL in practice, so we filter by the resolved numeric user_id.
    If user_id is unavailable we fall back to a name-based filter (may return
    zero rows on most workspaces but won't crash).
    """
    if user_id:
        user_filter = f"(j.creator_id = '{user_id}' OR j.run_as = '{user_id}')"
        jobs_filter = f"(creator_id = '{user_id}' OR run_as = '{user_id}')"
    else:
        user_filter = f"(j.creator_user_name = '{user_email}' OR j.run_as_user_name = '{user_email}')"
        jobs_filter = f"(creator_user_name = '{user_email}' OR run_as_user_name = '{user_email}')"

    jobs = execute_query(f"""
        SELECT job_id, name, creator_id, run_as, paused, tags, change_time
        FROM system.lakeflow.jobs
        WHERE {jobs_filter}
        ORDER BY change_time DESC
        LIMIT 100
    """, user_token=token)

    # Job run success rates — duration computed from period timestamps
    job_runs = execute_query(f"""
        SELECT
          j.name AS job_name,
          COUNT(*) AS total_runs,
          SUM(CASE WHEN jrt.result_state = 'SUCCEEDED' THEN 1 ELSE 0 END) AS successes,
          SUM(CASE WHEN jrt.result_state != 'SUCCEEDED' THEN 1 ELSE 0 END) AS failures,
          AVG(TIMESTAMPDIFF(SECOND, jrt.period_start_time, jrt.period_end_time)) AS avg_duration_secs
        FROM system.lakeflow.job_run_timeline jrt
        JOIN system.lakeflow.jobs j ON j.job_id = jrt.job_id
        WHERE {user_filter}
          AND jrt.period_start_time >= CURRENT_DATE - INTERVAL {lookback} DAYS
        GROUP BY j.name
        ORDER BY total_runs DESC
        LIMIT 30
    """, user_token=token)

    return {
        "jobs": {
            "owned_jobs": _serialize(jobs),
            "job_runs": _serialize(job_runs),
        }
    }


def _collect_lineage_profile(user_email, lookback, token=None):
    """Collect data lineage information.

    system.access.table_lineage uses `created_by` (not `executed_by`) as the
    user identity column.
    """
    # Tables read
    tables_read = execute_query(f"""
        SELECT source_table_full_name AS table_name,
               COUNT(*) AS read_count
        FROM system.access.table_lineage
        WHERE created_by = '{user_email}'
          AND event_date >= CURRENT_DATE - INTERVAL {lookback} DAYS
          AND source_table_full_name IS NOT NULL
        GROUP BY 1
        ORDER BY read_count DESC
        LIMIT 30
    """, user_token=token)

    # Tables written
    tables_written = execute_query(f"""
        SELECT target_table_full_name AS table_name,
               COUNT(*) AS write_count
        FROM system.access.table_lineage
        WHERE created_by = '{user_email}'
          AND event_date >= CURRENT_DATE - INTERVAL {lookback} DAYS
          AND target_table_full_name IS NOT NULL
        GROUP BY 1
        ORDER BY write_count DESC
        LIMIT 30
    """, user_token=token)

    # Catalogs and schemas used
    catalog_usage = execute_query(f"""
        SELECT
          SPLIT(COALESCE(source_table_full_name, target_table_full_name), '\\\\.')[0] AS catalog_name,
          COUNT(*) AS usage_count
        FROM system.access.table_lineage
        WHERE created_by = '{user_email}'
          AND event_date >= CURRENT_DATE - INTERVAL {lookback} DAYS
        GROUP BY 1
        ORDER BY usage_count DESC
        LIMIT 15
    """, user_token=token)

    return {
        "lineage": {
            "tables_read": _serialize(tables_read),
            "tables_written": _serialize(tables_written),
            "catalog_usage": _serialize(catalog_usage),
        }
    }


def _collect_collaboration_profile(user_email, lookback, token=None):
    """Collect collaboration patterns - who depends on this user's data.

    system.access.table_lineage uses `created_by` (not `executed_by`) as the
    user identity column.
    """
    # Downstream consumers (who reads tables this user has written).
    # Pre-filter each side by user/date BEFORE joining to avoid a full
    # cross-product scan of the lineage table.
    downstream = execute_query(f"""
        WITH my_writes AS (
            SELECT DISTINCT target_table_full_name
            FROM system.access.table_lineage
            WHERE created_by = '{user_email}'
              AND event_date >= CURRENT_DATE - INTERVAL {lookback} DAYS
        )
        SELECT tl.created_by AS downstream_user,
               COUNT(*) AS interaction_count
        FROM system.access.table_lineage tl
        JOIN my_writes ON tl.source_table_full_name = my_writes.target_table_full_name
        WHERE tl.created_by != '{user_email}'
          AND tl.event_date >= CURRENT_DATE - INTERVAL {lookback} DAYS
        GROUP BY 1
        ORDER BY interaction_count DESC
        LIMIT 20
    """, user_token=token)

    # Upstream dependencies (whose tables does this user read/use).
    upstream = execute_query(f"""
        WITH my_reads AS (
            SELECT DISTINCT source_table_full_name
            FROM system.access.table_lineage
            WHERE created_by = '{user_email}'
              AND event_date >= CURRENT_DATE - INTERVAL {lookback} DAYS
        )
        SELECT tl.created_by AS upstream_user,
               COUNT(*) AS interaction_count
        FROM system.access.table_lineage tl
        JOIN my_reads ON tl.target_table_full_name = my_reads.source_table_full_name
        WHERE tl.created_by != '{user_email}'
          AND tl.event_date >= CURRENT_DATE - INTERVAL {lookback} DAYS
        GROUP BY 1
        ORDER BY interaction_count DESC
        LIMIT 20
    """, user_token=token)

    return {
        "collaboration": {
            "downstream_consumers": _serialize(downstream),
            "upstream_dependencies": _serialize(upstream),
        }
    }


def _collect_workspace_comparison(user_email, lookback, token=None):
    """Collect workspace-wide aggregates for comparison (privacy-safe: only averages).

    Avoids the expensive PERCENT_RANK() window function that scans all users.
    Instead, counts how many users have more/fewer events than the current user
    which needs only a single aggregation pass, not a full sort.
    """
    try:
        # Single query: workspace averages + this user's counts + approximate
        # percentile via counting users above/below (no window function needed).
        comparison = execute_query(f"""
            WITH per_user AS (
                SELECT user_identity.email AS email,
                       COUNT(*) AS event_cnt,
                       COUNT(DISTINCT event_date) AS active_days
                FROM system.access.audit
                WHERE event_date >= CURRENT_DATE - INTERVAL {lookback} DAYS
                  AND user_identity.email IS NOT NULL
                  AND user_identity.email NOT LIKE '%service-principal%'
                GROUP BY user_identity.email
            ),
            my_cnt AS (
                SELECT COALESCE(MAX(event_cnt), 0) AS val
                FROM per_user WHERE email = '{user_email}'
            )
            SELECT
                COUNT(*)                            AS total_users,
                ROUND(AVG(per_user.event_cnt))      AS avg_events_per_user,
                ROUND(AVG(per_user.active_days))    AS avg_active_days,
                MAX(my_cnt.val)                     AS my_events,
                ROUND(SUM(CASE WHEN per_user.event_cnt <= my_cnt.val
                               THEN 1 ELSE 0 END) * 100.0 / GREATEST(COUNT(*), 1)) AS event_percentile
            FROM per_user
            CROSS JOIN my_cnt
        """, user_token=token)

        query_comparison = execute_query(f"""
            WITH per_user AS (
                SELECT executed_by AS email, COUNT(*) AS query_cnt
                FROM system.query.history
                WHERE start_time >= CURRENT_DATE - INTERVAL {lookback} DAYS
                  AND executed_by IS NOT NULL
                GROUP BY executed_by
            ),
            my_cnt AS (
                SELECT COALESCE(MAX(query_cnt), 0) AS val
                FROM per_user WHERE email = '{user_email}'
            )
            SELECT
                ROUND(AVG(per_user.query_cnt))      AS avg_queries_per_user,
                COUNT(*)                            AS total_querying_users,
                ROUND(SUM(CASE WHEN per_user.query_cnt <= my_cnt.val
                               THEN 1 ELSE 0 END) * 100.0 / GREATEST(COUNT(*), 1)) AS query_percentile
            FROM per_user
            CROSS JOIN my_cnt
        """, user_token=token)

        job_stats = execute_query(f"""
            SELECT
                COUNT(*) / GREATEST(COUNT(DISTINCT j.creator_id), 1) AS avg_jobs_per_user,
                COUNT(DISTINCT j.creator_id) AS total_job_owners
            FROM system.lakeflow.jobs j
            WHERE j.creator_id IS NOT NULL
              AND j.change_time >= CURRENT_DATE - INTERVAL {lookback} DAYS
        """, user_token=token)

        c  = comparison[0] if comparison else {}
        qc = query_comparison[0] if query_comparison else {}
        js = job_stats[0] if job_stats else {}

        return {
            "comparison": {
                "workspace_users": int(c.get("total_users", 0)),
                "avg_events_per_user": round(float(c.get("avg_events_per_user", 0))),
                "avg_active_days": round(float(c.get("avg_active_days", 0))),
                "avg_queries_per_user": round(float(qc.get("avg_queries_per_user", 0))),
                "avg_query_duration_ms": 0,  # dropped: requires separate full scan
                "avg_jobs_per_user": round(float(js.get("avg_jobs_per_user", 0))),
                "event_percentile": int(c.get("event_percentile", 0)),
                "query_percentile": int(qc.get("query_percentile", 0)),
            }
        }
    except Exception:
        return {"comparison": {}}


def _collect_job_deep_dive(user_email, lookback, token=None, user_id=None):
    """Deep dive into job runs: failures, trigger types, timing patterns.

    Uses numeric user_id (creator_id / run_as) when available since
    creator_user_name / run_as_user_name are NULL in practice.
    Duration is computed from period timestamps as run_duration_seconds is 0.
    """
    if user_id:
        user_filter = f"(j.creator_id = '{user_id}' OR j.run_as = '{user_id}')"
    else:
        user_filter = f"(j.creator_user_name = '{user_email}' OR j.run_as_user_name = '{user_email}')"

    # Per-job run breakdown with trigger type and result state
    run_breakdown = execute_query(f"""
        SELECT
          j.name AS job_name,
          jrt.result_state,
          jrt.trigger_type,
          COUNT(*) AS run_count,
          AVG(TIMESTAMPDIFF(SECOND, jrt.period_start_time, jrt.period_end_time)) AS avg_duration_secs,
          MIN(jrt.period_start_time) AS first_run,
          MAX(jrt.period_start_time) AS last_run
        FROM system.lakeflow.job_run_timeline jrt
        JOIN system.lakeflow.jobs j ON j.job_id = jrt.job_id
        WHERE {user_filter}
          AND jrt.period_start_time >= CURRENT_DATE - INTERVAL {lookback} DAYS
        GROUP BY j.name, jrt.result_state, jrt.trigger_type
        ORDER BY j.name, run_count DESC
    """, user_token=token)

    # Failure analysis
    failures = execute_query(f"""
        SELECT
          j.name AS job_name,
          jrt.result_state,
          jrt.termination_code,
          COUNT(*) AS fail_count,
          MAX(jrt.period_start_time) AS last_failure
        FROM system.lakeflow.job_run_timeline jrt
        JOIN system.lakeflow.jobs j ON j.job_id = jrt.job_id
        WHERE {user_filter}
          AND jrt.result_state != 'SUCCEEDED'
          AND jrt.period_start_time >= CURRENT_DATE - INTERVAL {lookback} DAYS
        GROUP BY j.name, jrt.result_state, jrt.termination_code
        ORDER BY fail_count DESC
    """, user_token=token)

    # Job run timing — what hour do jobs typically run
    run_timing = execute_query(f"""
        SELECT
          j.name AS job_name,
          HOUR(jrt.period_start_time) AS run_hour,
          COUNT(*) AS run_count
        FROM system.lakeflow.job_run_timeline jrt
        JOIN system.lakeflow.jobs j ON j.job_id = jrt.job_id
        WHERE {user_filter}
          AND jrt.period_start_time >= CURRENT_DATE - INTERVAL {lookback} DAYS
        GROUP BY j.name, HOUR(jrt.period_start_time)
        ORDER BY j.name, run_hour
    """, user_token=token)

    return {
        "job_deep_dive": {
            "run_breakdown": _serialize(run_breakdown),
            "failures": _serialize(failures),
            "run_timing": _serialize(run_timing),
        }
    }


def _collect_table_governance_profile(user_email, lookback, token=None):
    """Collect table lifecycle and Unity Catalog governance activity."""
    # Table lifecycle summary (create/delete/alter counts)
    table_lifecycle = execute_query(f"""
        SELECT action_name, COUNT(*) AS cnt
        FROM system.access.audit
        WHERE event_date >= CURRENT_DATE - INTERVAL {lookback} DAYS
          AND user_identity.email = '{user_email}'
          AND service_name = 'unityCatalog'
          AND action_name IN ('createTable', 'deleteTable', 'alterTable')
        GROUP BY action_name
        ORDER BY cnt DESC
    """, user_token=token)

    # Table names for created tables
    tables_created = execute_query(f"""
        SELECT
          CONCAT(
            request_params['catalog_name'], '.',
            request_params['schema_name'], '.',
            request_params['name']
          ) AS table_name,
          COUNT(*) AS create_count,
          MIN(event_time) AS first_created,
          MAX(event_time) AS last_created
        FROM system.access.audit
        WHERE event_date >= CURRENT_DATE - INTERVAL {lookback} DAYS
          AND user_identity.email = '{user_email}'
          AND service_name = 'unityCatalog'
          AND action_name = 'createTable'
          AND request_params['catalog_name'] IS NOT NULL
        GROUP BY 1
        ORDER BY last_created DESC
        LIMIT 20
    """, user_token=token)

    # Table names for deleted tables
    tables_deleted = execute_query(f"""
        SELECT
          request_params['full_name_arg'] AS table_name,
          COUNT(*) AS delete_count,
          MAX(event_time) AS deleted_at
        FROM system.access.audit
        WHERE event_date >= CURRENT_DATE - INTERVAL {lookback} DAYS
          AND user_identity.email = '{user_email}'
          AND service_name = 'unityCatalog'
          AND action_name = 'deleteTable'
          AND request_params['full_name_arg'] IS NOT NULL
        GROUP BY 1
        ORDER BY deleted_at DESC
        LIMIT 20
    """, user_token=token)

    # Governance actions (permissions, catalog/schema creation)
    governance = execute_query(f"""
        SELECT action_name, COUNT(*) AS cnt
        FROM system.access.audit
        WHERE event_date >= CURRENT_DATE - INTERVAL {lookback} DAYS
          AND user_identity.email = '{user_email}'
          AND service_name = 'unityCatalog'
          AND action_name IN (
            'updatePermissions', 'getPermissions',
            'createCatalog', 'deleteCatalog',
            'createSchema', 'deleteSchema',
            'createVolume', 'deleteVolume',
            'createFunction', 'deleteFunction'
          )
        GROUP BY action_name
        ORDER BY cnt DESC
    """, user_token=token)

    return {
        "table_governance": {
            "table_lifecycle": _serialize(table_lifecycle),
            "tables_created": _serialize(tables_created),
            "tables_deleted": _serialize(tables_deleted),
            "governance": _serialize(governance),
        }
    }


def _collect_genai_profile(user_email, lookback, token=None):
    """Collect GenAI/LLM usage from three sources:

    1. system.access.audit (aibiGenie) — AI/BI Genie conversations, directly attributed.
    2. system.access.audit (agentFramework) — Agent deployments/queries, directly attributed.
    3. system.billing.usage — Model endpoint billing costs. Per-user attribution only works
       for custom endpoints; Anthropic/foundation model run_as is NULL so workspace totals
       are returned for those.
    """
    try:
        # ── AI/BI Genie conversations ──────────────────────────────────────────
        genie_activity = execute_query(f"""
            SELECT action_name, COUNT(*) AS cnt,
                   COUNT(DISTINCT DATE(event_time)) AS active_days
            FROM system.access.audit
            WHERE user_identity.email = '{user_email}'
              AND service_name = 'aibiGenie'
              AND event_date >= CURRENT_DATE - INTERVAL {lookback} DAYS
            GROUP BY action_name
            ORDER BY cnt DESC
        """, user_token=token)

        genie_summary = execute_query(f"""
            SELECT
              COUNT(CASE WHEN action_name IN (
                'genieStartConversationMessage','genieCreateConversationMessage',
                'createConversationMessage','createConversation'
              ) THEN 1 END) AS conversations_started,
              COUNT(CASE WHEN action_name IN (
                'genieGetConversationMessage','getConversationMessage',
                'listGenieSpaceMessages','listConversationMessageComments'
              ) THEN 1 END) AS messages_read,
              COUNT(DISTINCT DATE(event_time)) AS active_days
            FROM system.access.audit
            WHERE user_identity.email = '{user_email}'
              AND service_name = 'aibiGenie'
              AND event_date >= CURRENT_DATE - INTERVAL {lookback} DAYS
        """, user_token=token)

        # ── Agent Framework ────────────────────────────────────────────────────
        agent_activity = execute_query(f"""
            SELECT action_name, COUNT(*) AS cnt,
                   COUNT(DISTINCT DATE(event_time)) AS active_days
            FROM system.access.audit
            WHERE user_identity.email = '{user_email}'
              AND service_name = 'agentFramework'
              AND event_date >= CURRENT_DATE - INTERVAL {lookback} DAYS
            GROUP BY action_name
            ORDER BY cnt DESC
        """, user_token=token)

        agent_summary = execute_query(f"""
            SELECT
              COUNT(CASE WHEN action_name = 'deployChain' THEN 1 END) AS agents_deployed,
              COUNT(CASE WHEN action_name IN ('getChainDeployments','listChainDeployments') THEN 1 END) AS agent_queries,
              COUNT(DISTINCT DATE(event_time)) AS active_days
            FROM system.access.audit
            WHERE user_identity.email = '{user_email}'
              AND service_name = 'agentFramework'
              AND event_date >= CURRENT_DATE - INTERVAL {lookback} DAYS
        """, user_token=token)

        # ── Billing: endpoints directly attributed to this user ────────────────
        user_endpoints = execute_query(f"""
            SELECT
              COALESCE(usage_metadata.endpoint_name, sku_name) AS endpoint_name,
              sku_name,
              SUM(usage_quantity) AS total_dbu,
              COUNT(*) AS billing_records,
              MIN(usage_date) AS first_used,
              MAX(usage_date) AS last_used
            FROM system.billing.usage
            WHERE usage_date >= CURRENT_DATE - INTERVAL {lookback} DAYS
              AND (
                sku_name LIKE '%INFERENCE%'
                OR sku_name LIKE '%MODEL_SERVING%'
                OR sku_name LIKE '%ANTHROPIC%'
                OR sku_name LIKE '%SERVING%'
              )
              AND identity_metadata.run_as = '{user_email}'
            GROUP BY COALESCE(usage_metadata.endpoint_name, sku_name), sku_name
            ORDER BY total_dbu DESC
            LIMIT 20
        """, user_token=token)

        # ── Billing: workspace-wide foundation model usage ─────────────────────
        workspace_endpoints = execute_query(f"""
            SELECT
              COALESCE(usage_metadata.endpoint_name, sku_name) AS endpoint_name,
              sku_name,
              SUM(usage_quantity) AS total_dbu,
              COUNT(*) AS billing_records,
              MIN(usage_date) AS first_used,
              MAX(usage_date) AS last_used
            FROM system.billing.usage
            WHERE usage_date >= CURRENT_DATE - INTERVAL {lookback} DAYS
              AND (
                sku_name LIKE '%INFERENCE%'
                OR sku_name LIKE '%MODEL_SERVING%'
                OR sku_name LIKE '%ANTHROPIC%'
                OR sku_name LIKE '%SERVING%'
              )
            GROUP BY COALESCE(usage_metadata.endpoint_name, sku_name), sku_name
            ORDER BY total_dbu DESC
            LIMIT 20
        """, user_token=token)

        # ── Build modality summary (drives the constellation viz) ─────────────
        gs = genie_summary[0] if genie_summary else {}
        ags = agent_summary[0] if agent_summary else {}
        genie_total = sum(int(r.get("cnt") or 0) for r in (genie_activity or []))
        agent_total = sum(int(r.get("cnt") or 0) for r in (agent_activity or []))
        endpoint_dbu = sum(float(r.get("total_dbu") or 0) for r in (user_endpoints or []))
        workspace_dbu = sum(float(r.get("total_dbu") or 0) for r in (workspace_endpoints or []))

        modality_summary = [
            {
                "modality": "AI/BI Genie",
                "icon": "✨",
                "description": "Natural language data conversations",
                "conversations_started": int(gs.get("conversations_started") or 0),
                "messages_read": int(gs.get("messages_read") or 0),
                "total_interactions": genie_total,
                "active_days": int(gs.get("active_days") or 0),
            },
            {
                "modality": "Agent Framework",
                "icon": "🦾",
                "description": "Deployed AI agent pipelines",
                "agents_deployed": int(ags.get("agents_deployed") or 0),
                "agent_queries": int(ags.get("agent_queries") or 0),
                "total_interactions": agent_total,
                "active_days": int(ags.get("active_days") or 0),
            },
            {
                "modality": "Model Endpoints",
                "icon": "🧠",
                "description": "Foundation model API usage (workspace-wide)",
                "endpoints_count": len(workspace_endpoints or []),
                "total_dbu_attributed": round(endpoint_dbu, 2),
                "total_dbu_workspace": round(workspace_dbu, 2),
                "active_days": len(set(
                    str(r.get("first_used", ""))[:10]
                    for r in (workspace_endpoints or [])
                    if r.get("first_used")
                )),
            },
        ]

        return {
            "genai": {
                "modality_summary": modality_summary,
                "genie_activity": _serialize(genie_activity),
                "agent_activity": _serialize(agent_activity),
                "user_endpoints": _serialize(user_endpoints),
                "workspace_endpoints": _serialize(workspace_endpoints),
                "attribution_note": (
                    "Foundation model & Anthropic endpoints do not expose per-user "
                    "attribution in billing data (identity_metadata.run_as is null). "
                    "Workspace totals are shown for those endpoints."
                ),
            }
        }
    except Exception:
        return {"genai": {}}


def _collect_git_profile(user_email, lookback, token=None):
    """Collect Git/Repos version control activity."""
    # Git action breakdown
    git_actions = execute_query(f"""
        SELECT action_name, COUNT(*) AS cnt
        FROM system.access.audit
        WHERE event_date >= CURRENT_DATE - INTERVAL {lookback} DAYS
          AND user_identity.email = '{user_email}'
          AND service_name = 'repos'
        GROUP BY action_name
        ORDER BY cnt DESC
    """, user_token=token)

    # Repos created
    repos_created = execute_query(f"""
        SELECT
          request_params['url'] AS repo_url,
          request_params['path'] AS repo_path,
          MIN(event_time) AS created_at
        FROM system.access.audit
        WHERE event_date >= CURRENT_DATE - INTERVAL {lookback} DAYS
          AND user_identity.email = '{user_email}'
          AND service_name = 'repos'
          AND action_name = 'createRepo'
        GROUP BY request_params['url'], request_params['path']
        ORDER BY created_at DESC
        LIMIT 10
    """, user_token=token)

    return {
        "git_activity": {
            "actions": _serialize(git_actions),
            "repos_created": _serialize(repos_created),
        }
    }


def _collect_cost_profile(user_email, lookback, token=None):
    """Collect cost and efficiency metrics from billing data."""
    try:
        # Per-user cost by SKU
        sku_costs = execute_query(f"""
            SELECT sku_name, SUM(usage_quantity) AS total_dbu
            FROM system.billing.usage
            WHERE usage_date >= CURRENT_DATE - INTERVAL {lookback} DAYS
              AND identity_metadata.run_as = '{user_email}'
              AND usage_unit = 'DBU'
            GROUP BY sku_name
            ORDER BY total_dbu DESC
        """, user_token=token)

        # Weekly cost trend
        weekly_trend = execute_query(f"""
            SELECT
              DATE_TRUNC('week', usage_date) AS week,
              SUM(usage_quantity) AS weekly_dbu
            FROM system.billing.usage
            WHERE usage_date >= CURRENT_DATE - INTERVAL {lookback} DAYS
              AND identity_metadata.run_as = '{user_email}'
              AND usage_unit = 'DBU'
            GROUP BY DATE_TRUNC('week', usage_date)
            ORDER BY week
        """, user_token=token)

        # User's share of workspace total
        cost_share = execute_query(f"""
            SELECT
              SUM(CASE WHEN identity_metadata.run_as = '{user_email}' THEN usage_quantity ELSE 0 END) AS user_dbu,
              SUM(usage_quantity) AS workspace_total_dbu
            FROM system.billing.usage
            WHERE usage_date >= CURRENT_DATE - INTERVAL {lookback} DAYS
              AND usage_unit = 'DBU'
        """, user_token=token)

        # Cost by compute category
        cost_categories = execute_query(f"""
            SELECT
              CASE
                WHEN sku_name LIKE '%JOBS%' THEN 'Jobs Compute'
                WHEN sku_name LIKE '%SQL%' THEN 'SQL Warehouses'
                WHEN sku_name LIKE '%ALL_PURPOSE%' THEN 'Interactive Compute'
                WHEN sku_name LIKE '%INFERENCE%' OR sku_name LIKE '%MODEL_SERVING%' OR sku_name LIKE '%ANTHROPIC%' THEN 'Model Serving / AI'
                WHEN sku_name LIKE '%STORAGE%' THEN 'Storage'
                ELSE 'Other'
              END AS cost_category,
              SUM(usage_quantity) AS total_dbu
            FROM system.billing.usage
            WHERE usage_date >= CURRENT_DATE - INTERVAL {lookback} DAYS
              AND identity_metadata.run_as = '{user_email}'
              AND usage_unit = 'DBU'
            GROUP BY 1
            ORDER BY total_dbu DESC
        """, user_token=token)

        return {
            "cost": {
                "sku_costs": _serialize(sku_costs),
                "weekly_trend": _serialize(weekly_trend),
                "cost_share": _serialize(cost_share),
                "cost_categories": _serialize(cost_categories),
            }
        }
    except Exception:
        return {"cost": {}}


def _collect_dashboard_profile(user_email, lookback, token=None):
    """Collect dashboard activity from audit logs.

    Returns meaningful engagement metrics instead of raw API action counts:
    - sessions: per-day quality breakdown (queries run, edits, views per session)
    - top_dashboards: engagement depth per dashboard (query intensity, builder vs consumer)
    - behavior: overall builder/consumer/publisher role summary
    """
    # ── Per-session (per-day) quality breakdown ────────────────────────────────
    # Use event_date (partition col) for grouping instead of DATE(event_time)
    sessions = execute_query(f"""
        SELECT
          event_date AS session_date,
          COUNT(DISTINCT request_params['dashboard_id']) AS dashboards_touched,
          SUM(CASE WHEN action_name = 'executeQuery' THEN 1 ELSE 0 END) AS queries_run,
          SUM(CASE WHEN action_name = 'cancelQuery' THEN 1 ELSE 0 END) AS queries_cancelled,
          SUM(CASE WHEN action_name IN (
            'createDashboard','updateDashboard','cloneDashboard','publishDashboard'
          ) THEN 1 ELSE 0 END) AS edits_made,
          SUM(CASE WHEN action_name IN (
            'getDashboard','getDashboardDetails','getPublishedDashboard'
          ) THEN 1 ELSE 0 END) AS views,
          COUNT(*) AS total_actions
        FROM system.access.audit
        WHERE user_identity.email = '{user_email}'
          AND service_name = 'dashboards'
          AND event_date >= CURRENT_DATE - INTERVAL {lookback} DAYS
        GROUP BY event_date
        ORDER BY session_date DESC
        LIMIT 30
    """, user_token=token)

    # ── Top dashboards by engagement depth ────────────────────────────────────
    top_dashboards = execute_query(f"""
        SELECT
          request_params['dashboard_id'] AS dashboard_id,
          COUNT(*) AS total_actions,
          SUM(CASE WHEN action_name = 'executeQuery' THEN 1 ELSE 0 END) AS query_runs,
          SUM(CASE WHEN action_name IN (
            'createDashboard','updateDashboard','cloneDashboard'
          ) THEN 1 ELSE 0 END) AS edit_count,
          SUM(CASE WHEN action_name = 'cancelQuery' THEN 1 ELSE 0 END) AS cancelled_queries,
          MIN(event_date) AS first_seen,
          MAX(event_date) AS last_seen,
          COUNT(DISTINCT event_date) AS active_days
        FROM system.access.audit
        WHERE user_identity.email = '{user_email}'
          AND service_name = 'dashboards'
          AND event_date >= CURRENT_DATE - INTERVAL {lookback} DAYS
          AND request_params['dashboard_id'] IS NOT NULL
        GROUP BY request_params['dashboard_id']
        ORDER BY total_actions DESC
        LIMIT 15
    """, user_token=token)

    # ── Overall behavior summary (builder vs consumer) ────────────────────────
    behavior = execute_query(f"""
        SELECT
          SUM(CASE WHEN action_name IN ('createDashboard','cloneDashboard') THEN 1 ELSE 0 END) AS dashboards_created,
          SUM(CASE WHEN action_name = 'updateDashboard' THEN 1 ELSE 0 END) AS dashboards_edited,
          SUM(CASE WHEN action_name IN ('publishDashboard','triggerDashboardSnapshot') THEN 1 ELSE 0 END) AS publishing_actions,
          SUM(CASE WHEN action_name IN ('executeQuery','getQueryResult') THEN 1 ELSE 0 END) AS query_interactions,
          SUM(CASE WHEN action_name IN ('getDashboard','getDashboardDetails','getPublishedDashboard') THEN 1 ELSE 0 END) AS dashboard_views,
          COUNT(DISTINCT event_date) AS active_days,
          COUNT(DISTINCT request_params['dashboard_id']) AS unique_dashboards
        FROM system.access.audit
        WHERE user_identity.email = '{user_email}'
          AND service_name = 'dashboards'
          AND event_date >= CURRENT_DATE - INTERVAL {lookback} DAYS
    """, user_token=token)

    # ── Resolve human-readable dashboard names from Lakeview API ──────────────
    # Deleted dashboards will not be found; their dashboard_name stays empty
    # so the UI falls back to displaying the short ID.
    if top_dashboards:
        all_ids = [r.get("dashboard_id") for r in top_dashboards if r.get("dashboard_id")]
        name_map = _resolve_dashboard_names(all_ids, token=token)
        for row in top_dashboards:
            did = row.get("dashboard_id", "")
            row["dashboard_name"] = name_map.get(did, "")

    return {
        "dashboard_activity": {
            "sessions": _serialize(sessions),
            "top_dashboards": _serialize(top_dashboards),
            "behavior": _serialize(behavior),
        }
    }


def _collect_engagement_timeline(user_email, lookback, token=None):
    """Collect engagement heatmap, streaks, and weekly trends."""
    # 24x7 heatmap matrix
    heatmap = execute_query(f"""
        SELECT
          HOUR(event_time) AS hour_of_day,
          DAYOFWEEK(event_time) AS day_of_week,
          COUNT(*) AS event_count
        FROM system.access.audit
        WHERE event_date >= CURRENT_DATE - INTERVAL {lookback} DAYS
          AND user_identity.email = '{user_email}'
        GROUP BY HOUR(event_time), DAYOFWEEK(event_time)
        ORDER BY day_of_week, hour_of_day
    """, user_token=token)

    # Daily activity for streak calculation — use event_date (partition col)
    daily_volumes = execute_query(f"""
        SELECT
          event_date AS activity_date,
          COUNT(*) AS event_count,
          COUNT(DISTINCT service_name) AS services_used
        FROM system.access.audit
        WHERE event_date >= CURRENT_DATE - INTERVAL {lookback} DAYS
          AND user_identity.email = '{user_email}'
        GROUP BY event_date
        ORDER BY event_date
    """, user_token=token)

    # Weekly trend — use event_date to avoid materialising timestamps
    weekly_trend = execute_query(f"""
        SELECT
          DATE_TRUNC('week', event_date) AS week,
          COUNT(*) AS event_count
        FROM system.access.audit
        WHERE event_date >= CURRENT_DATE - INTERVAL {lookback} DAYS
          AND user_identity.email = '{user_email}'
        GROUP BY DATE_TRUNC('week', event_date)
        ORDER BY week
    """, user_token=token)

    # Busiest single day — use event_date (partition col) for grouping
    busiest_day = execute_query(f"""
        SELECT
          event_date AS activity_date,
          COUNT(*) AS event_count,
          COUNT(DISTINCT service_name) AS services_used,
          MIN(event_time) AS first_event,
          MAX(event_time) AS last_event
        FROM system.access.audit
        WHERE event_date >= CURRENT_DATE - INTERVAL {lookback} DAYS
          AND user_identity.email = '{user_email}'
        GROUP BY event_date
        ORDER BY event_count DESC
        LIMIT 1
    """, user_token=token)

    # Compute streaks in Python from daily_volumes
    streak_info = _compute_streaks(daily_volumes)

    return {
        "engagement": {
            "heatmap": _serialize(heatmap),
            "daily_volumes": _serialize(daily_volumes),
            "weekly_trend": _serialize(weekly_trend),
            "busiest_day": _serialize(busiest_day),
            "current_streak": streak_info["current_streak"],
            "longest_streak": streak_info["longest_streak"],
            "total_active_days": streak_info["total_active_days"],
        }
    }


def _compute_streaks(daily_volumes):
    """Compute current and longest streaks from daily activity data."""
    if not daily_volumes:
        return {"current_streak": 0, "longest_streak": 0, "total_active_days": 0}

    from datetime import date, timedelta

    # Parse dates (may be date objects or strings)
    active_dates = set()
    for row in daily_volumes:
        d = row.get("activity_date")
        if d is None:
            continue
        if isinstance(d, str):
            d = date.fromisoformat(d[:10])
        elif hasattr(d, "date"):
            d = d.date()
        active_dates.add(d)

    if not active_dates:
        return {"current_streak": 0, "longest_streak": 0, "total_active_days": 0}

    sorted_dates = sorted(active_dates)
    total_active_days = len(sorted_dates)

    # Compute longest streak
    longest = 1
    current = 1
    for i in range(1, len(sorted_dates)):
        if (sorted_dates[i] - sorted_dates[i - 1]).days == 1:
            current += 1
            longest = max(longest, current)
        else:
            current = 1

    # Compute current streak (from today backwards)
    today = date.today()
    current_streak = 0
    check_date = today
    while check_date in active_dates:
        current_streak += 1
        check_date -= timedelta(days=1)

    # If today isn't active yet, check from yesterday
    if current_streak == 0:
        check_date = today - timedelta(days=1)
        while check_date in active_dates:
            current_streak += 1
            check_date -= timedelta(days=1)

    return {
        "current_streak": current_streak,
        "longest_streak": longest,
        "total_active_days": total_active_days,
    }


def _build_data_health(persona, errors):
    """Build a summary of which data sections have data vs are empty/errored."""
    sections = {
        "activity": ("Platform Activity", "system.access.audit"),
        "queries": ("SQL Queries", "system.query.history"),
        "jobs": ("Jobs & Pipelines", "system.lakeflow.jobs"),
        "lineage": ("Data Lineage", "system.access.table_lineage"),
        "compute": ("Compute Clusters", "system.compute.clusters"),
        "collaboration": ("Collaboration", "system.access.table_lineage"),
        "cost": ("Cost & Billing", "system.billing.usage"),
        "genai": ("GenAI/LLM", "system.billing.usage"),
        "engagement": ("Engagement Timeline", "system.access.audit"),
        "dashboard_activity": ("Dashboards", "system.access.audit"),
        "git_activity": ("Git/Repos", "system.access.audit"),
        "table_governance": ("Table Governance", "system.access.audit"),
        "job_deep_dive": ("Job Deep Dive", "system.lakeflow.job_run_timeline"),
    }
    health = []
    for key, (label, source_table) in sections.items():
        section_data = persona.get(key, {})
        if isinstance(section_data, dict) and "error" in section_data:
            status = "error"
            detail = section_data["error"]
        elif not section_data:
            status = "empty"
            detail = "No data returned"
        else:
            # Check if all sub-keys are empty lists
            has_data = False
            for v in section_data.values():
                if isinstance(v, list) and len(v) > 0:
                    has_data = True
                    break
                elif isinstance(v, (int, float)) and v > 0:
                    has_data = True
                    break
                elif isinstance(v, dict) and len(v) > 0:
                    has_data = True
                    break
            status = "ok" if has_data else "empty"
            detail = None
        health.append({
            "section": key,
            "label": label,
            "source_table": source_table,
            "status": status,
            "detail": detail,
        })
    return {"sections": health, "errors": errors}


def _build_summary(user_email, lookback, token=None, accumulated=None):
    """Build summary statistics, role classification, and archetype scoring."""
    persona_data = accumulated or {}

    # Extract signals from flat dict for scoring
    signals = _extract_signals(persona_data)

    # Classify into 6 archetypes
    archetypes = _classify_archetype(signals)
    primary = archetypes[0]

    # Infer expertise areas
    expertise = _infer_expertise(persona_data, signals)

    # Compute achievement badges
    badges = _compute_badges(persona_data, signals)

    # Compute data reliability score
    reliability = _compute_reliability_score(persona_data, signals)

    return {
        "summary": {
            "user_email": user_email,
            "lookback_days": lookback,
            "inferred_role": primary["name"],
            "expertise_areas": expertise,
        },
        "archetype": {
            "primary": primary,
            "all": archetypes,
        },
        "badges": badges,
        "reliability": reliability,
    }


def _compute_reliability_score(data, signals):
    """Compute composite reliability score (0-100) from multiple signals."""
    components = {}

    # 1. Query success rate (30% weight)
    queries = data.get("queries", {})
    error_stats = queries.get("error_stats", [{}])
    total_q = error_stats[0].get("total_queries", 0) if error_stats else 0
    successful_q = error_stats[0].get("successful", 0) if error_stats else 0
    if total_q > 0:
        query_rate = (successful_q / total_q) * 100
    else:
        query_rate = 100  # No queries = no failures
    components["query_success_rate"] = round(query_rate, 1)

    # 2. Job success rate (30% weight)
    job_success = signals.get("job_success_rate", 0) * 100
    if signals.get("total_job_runs", 0) == 0:
        job_success = 100  # No jobs = no failures
    components["job_success_rate"] = round(job_success, 1)

    # 3. Schema stability (20% weight) — low ALTER TABLE relative to CREATE
    tg = data.get("table_governance", {})
    lifecycle = tg.get("table_lifecycle", [])
    creates = sum(r.get("cnt", 0) for r in lifecycle if r.get("action_name") == "createTable")
    alters = sum(r.get("cnt", 0) for r in lifecycle if r.get("action_name") == "alterTable")
    if creates + alters > 0:
        schema_stability = ((creates) / (creates + alters)) * 100
    else:
        schema_stability = 100
    components["schema_stability"] = round(schema_stability, 1)

    # 4. Pipeline consistency (20% weight) — based on job run regularity
    engagement = data.get("engagement", {})
    total_active = engagement.get("total_active_days", 0)
    longest_streak = engagement.get("longest_streak", 0)
    if total_active > 0:
        pipeline_consistency = min((longest_streak / total_active) * 100 * 1.5, 100)
    else:
        pipeline_consistency = 50
    components["pipeline_consistency"] = round(pipeline_consistency, 1)

    # Weighted composite
    overall = (
        query_rate * 0.30
        + job_success * 0.30
        + schema_stability * 0.20
        + pipeline_consistency * 0.20
    )
    components["overall"] = round(overall, 1)

    return components


# ─── 6 Persona Archetypes ─────────────────────────────────────────────────

PERSONA_ARCHETYPES = [
    {
        "name": "The Pipeline Architect",
        "emoji": "\U0001f3d7\ufe0f",
        "color": "#2196F3",
        "gradient": "linear-gradient(135deg, #1976D2, #42A5F5)",
        "icon_svg": '<svg viewBox="0 0 100 100"><rect x="10" y="60" width="20" height="30" rx="3" fill="white" opacity="0.9"/><rect x="40" y="40" width="20" height="50" rx="3" fill="white" opacity="0.9"/><rect x="70" y="20" width="20" height="70" rx="3" fill="white" opacity="0.9"/><line x1="20" y1="55" x2="50" y2="35" stroke="white" stroke-width="2" opacity="0.6"/><line x1="50" y1="35" x2="80" y2="15" stroke="white" stroke-width="2" opacity="0.6"/></svg>',
        "description": "You are the backbone of the data platform. You build and maintain the pipelines that keep data flowing — from notebooks to production jobs. Your code runs like clockwork, and downstream users depend on your reliability.",
        "traits": ["Heavy job automation", "Pipeline ownership", "Scheduled workloads", "Notebook-to-production", "Code-first development"],
    },
    {
        "name": "The Data Explorer",
        "emoji": "\U0001f50d",
        "color": "#4CAF50",
        "gradient": "linear-gradient(135deg, #388E3C, #66BB6A)",
        "icon_svg": '<svg viewBox="0 0 100 100"><circle cx="42" cy="42" r="25" fill="none" stroke="white" stroke-width="4" opacity="0.9"/><line x1="60" y1="60" x2="85" y2="85" stroke="white" stroke-width="5" stroke-linecap="round" opacity="0.9"/><circle cx="35" cy="35" r="5" fill="white" opacity="0.4"/></svg>',
        "description": "You are the curious analyst who dives deep into data. Your queries are mostly reads, exploring across many tables and schemas. You turn raw data into insights.",
        "traits": ["High SELECT ratio", "Wide table coverage", "Exploratory queries", "Dashboard consumer"],
    },
    {
        "name": "The Platform Guardian",
        "emoji": "\U0001f6e1\ufe0f",
        "color": "#F44336",
        "gradient": "linear-gradient(135deg, #D32F2F, #EF5350)",
        "icon_svg": '<svg viewBox="0 0 100 100"><path d="M50 10 L85 30 L85 55 Q85 80 50 95 Q15 80 15 55 L15 30 Z" fill="none" stroke="white" stroke-width="3" opacity="0.9"/><path d="M40 50 L48 58 L65 40" fill="none" stroke="white" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" opacity="0.9"/></svg>',
        "description": "You are the guardian of the platform. You manage access, configure clusters, and ensure the workspace runs smoothly. Your expertise spans admin services and governance.",
        "traits": ["Admin service usage", "Cluster management", "Unity Catalog governance", "Account management"],
    },
    {
        "name": "The Dashboard Crafter",
        "emoji": "\U0001f4ca",
        "color": "#9C27B0",
        "gradient": "linear-gradient(135deg, #7B1FA2, #AB47BC)",
        "icon_svg": '<svg viewBox="0 0 100 100"><rect x="10" y="10" width="80" height="80" rx="8" fill="none" stroke="white" stroke-width="3" opacity="0.9"/><rect x="20" y="55" width="12" height="25" rx="2" fill="white" opacity="0.7"/><rect x="38" y="35" width="12" height="45" rx="2" fill="white" opacity="0.8"/><rect x="56" y="45" width="12" height="35" rx="2" fill="white" opacity="0.7"/><rect x="74" y="25" width="12" height="55" rx="2" fill="white" opacity="0.9"/></svg>',
        "description": "You bring data to life through visualizations. Dashboards are your primary canvas, and your queries power the insights that drive decisions across the organization.",
        "traits": ["Dashboard-heavy queries", "Visualization focus", "BI workloads", "Stakeholder-facing"],
    },
    {
        "name": "The ML Alchemist",
        "emoji": "\U0001f9ec",
        "color": "#FF6F00",
        "gradient": "linear-gradient(135deg, #E65100, #FF9800)",
        "icon_svg": '<svg viewBox="0 0 100 100"><circle cx="50" cy="30" r="12" fill="none" stroke="white" stroke-width="2.5" opacity="0.9"/><circle cx="25" cy="65" r="10" fill="none" stroke="white" stroke-width="2" opacity="0.7"/><circle cx="75" cy="65" r="10" fill="none" stroke="white" stroke-width="2" opacity="0.7"/><line x1="43" y1="40" x2="30" y2="57" stroke="white" stroke-width="2" opacity="0.5"/><line x1="57" y1="40" x2="70" y2="57" stroke="white" stroke-width="2" opacity="0.5"/><line x1="35" y1="65" x2="65" y2="65" stroke="white" stroke-width="1.5" opacity="0.3"/><circle cx="50" cy="30" r="4" fill="white" opacity="0.4"/><circle cx="25" cy="65" r="3" fill="white" opacity="0.3"/><circle cx="75" cy="65" r="3" fill="white" opacity="0.3"/></svg>',
        "description": "You're the one training models, running experiments, and pushing GenAI into production. Notebooks are your lab, feature stores are your pantry, and inference endpoints are your launchpad.",
        "traits": ["ML/AI workloads", "Model training & serving", "Feature engineering", "GenAI experimentation"],
    },
    {
        "name": "The Cost-Conscious Optimizer",
        "emoji": "\U0001f4b0",
        "color": "#009688",
        "gradient": "linear-gradient(135deg, #00796B, #4DB6AC)",
        "icon_svg": '<svg viewBox="0 0 100 100"><circle cx="50" cy="50" r="35" fill="none" stroke="white" stroke-width="3" opacity="0.9"/><text x="50" y="62" text-anchor="middle" fill="white" font-size="36" font-weight="bold" opacity="0.9">$</text></svg>',
        "description": "You are resource-aware and efficient. You optimize compute usage, use serverless where possible, and keep costs under control while maintaining performance.",
        "traits": ["Low DBU usage", "Serverless adoption", "Efficient queries", "Auto-termination aware"],
    },
]


def _extract_signals(data):
    """Extract scoring signals from flat persona dict."""
    queries = data.get("queries", {})
    stmt_types = queries.get("statement_types", [])
    jobs = data.get("jobs", {})
    owned_jobs = jobs.get("owned_jobs", [])
    job_runs = jobs.get("job_runs", [])
    lineage = data.get("lineage", {})
    activity = data.get("activity", {})
    services = activity.get("services", [])
    query_sources = queries.get("query_sources", [])
    complexity = queries.get("complexity", [])

    # Statement type counts
    type_counts = {}
    for st in stmt_types:
        type_counts[st.get("statement_type", "")] = st.get("query_count", 0)
    total_queries = sum(type_counts.values()) or 1

    select_count = type_counts.get("SELECT", 0)
    write_count = (type_counts.get("INSERT", 0) + type_counts.get("MERGE", 0)
                   + type_counts.get("CREATE_TABLE_AS_SELECT", 0)
                   + type_counts.get("CREATE", 0))
    select_ratio = select_count / total_queries
    write_ratio = write_count / total_queries

    # Jobs
    jobs_count = len(owned_jobs)
    total_runs = sum(jr.get("total_runs", 0) for jr in job_runs)
    total_successes = sum(jr.get("successes", 0) for jr in job_runs)
    job_success_rate = total_successes / total_runs if total_runs > 0 else 0

    # Check for CRON/scheduled triggers
    has_cron = any(
        "cron" in str(j.get("schedule", "")).lower()
        or "periodic" in str(j.get("schedule", "")).lower()
        for j in owned_jobs if j.get("schedule")
    )

    # Read/write ratio from lineage
    tables_read = lineage.get("tables_read", [])
    tables_written = lineage.get("tables_written", [])
    total_reads = sum(t.get("read_count", 0) for t in tables_read)
    total_writes = sum(t.get("write_count", 0) for t in tables_written)
    rw_ratio = total_reads / total_writes if total_writes > 0 else float("inf")

    # Services as dict
    svc_dict = {s.get("service_name", ""): s.get("event_count", 0) for s in services}

    # Query sources as dict
    qs_dict = {s.get("source_type", ""): s.get("count", 0) for s in query_sources}

    # Heavy query ratio
    heavy_count = sum(
        c.get("count", 0) for c in complexity
        if any(kw in c.get("complexity", "").lower() for kw in ["heavy", "extreme"])
    )
    heavy_ratio = heavy_count / total_queries

    # Compute (we don't collect detailed cluster info in the app, but check services)
    clusters_in_services = svc_dict.get("clusters", 0)

    # Phase 8 signals
    # GenAI endpoint count
    genai = data.get("genai", {})
    genai_endpoints = genai.get("user_endpoints", [])
    unique_genai_endpoints = len({e.get("endpoint_name", "") for e in genai_endpoints})

    # Dashboard count
    dash = data.get("dashboard_activity", {})
    unique_dashboards = len(dash.get("unique_dashboards", []))

    # Governance actions
    tg = data.get("table_governance", {})
    gov_actions = tg.get("governance", [])
    permission_updates = sum(
        r.get("cnt", 0) for r in gov_actions
        if r.get("action_name") == "updatePermissions"
    )

    # Tables created
    tables_created_count = len(tg.get("tables_created", []))

    # Git activity
    git = data.get("git_activity", {})
    git_actions_list = git.get("actions", [])
    git_commits = sum(r.get("cnt", 0) for r in git_actions_list if r.get("action_name") == "commitAndPush")

    # Job deep dive: CRON trigger count
    jdd = data.get("job_deep_dive", {})
    run_breakdown = jdd.get("run_breakdown", [])
    cron_runs = sum(r.get("run_count", 0) for r in run_breakdown if r.get("trigger_type") == "CRON")

    # Engagement
    engagement = data.get("engagement", {})
    longest_streak = engagement.get("longest_streak", 0)

    # Cost
    cost = data.get("cost", {})
    cost_share = cost.get("cost_share", [{}])
    user_dbu = float(cost_share[0].get("user_dbu", 0)) if cost_share else 0
    workspace_dbu = float(cost_share[0].get("workspace_total_dbu", 1)) if cost_share else 1

    return {
        "select_ratio": select_ratio,
        "write_ratio": write_ratio,
        "total_queries": total_queries,
        "jobs_count": jobs_count,
        "total_job_runs": total_runs,
        "job_success_rate": job_success_rate,
        "has_cron": has_cron,
        "rw_ratio": rw_ratio,
        "tables_read_count": len(tables_read),
        "svc_dict": svc_dict,
        "qs_dict": qs_dict,
        "heavy_ratio": heavy_ratio,
        "clusters_in_services": clusters_in_services,
        # Phase 8
        "genai_endpoint_count": unique_genai_endpoints,
        "unique_dashboards": unique_dashboards,
        "permission_updates": permission_updates,
        "tables_created_count": tables_created_count,
        "git_commits": git_commits,
        "cron_runs": cron_runs,
        "longest_streak": longest_streak,
        "user_dbu": user_dbu,
        "workspace_dbu": workspace_dbu,
    }


def _classify_archetype(signals):
    """Score all 6 archetypes and return ranked list with match percentages.

    Uses a minimum floor (3%) per archetype so the distribution always looks
    realistic — no archetype ever shows 0%.  The remaining 82% (100 - 6*3) is
    distributed proportionally to raw scores.
    """
    scoring_fns = [
        _score_pipeline_architect,
        _score_data_explorer,
        _score_platform_guardian,
        _score_dashboard_crafter,
        _score_ml_alchemist,
        _score_cost_conscious,
    ]

    results = []
    for archetype, score_fn in zip(PERSONA_ARCHETYPES, scoring_fns):
        score = score_fn(signals)
        results.append({**archetype, "score": score})

    results.sort(key=lambda x: -x["score"])

    # ── Normalize with minimum floor ──
    # Every archetype gets at least FLOOR_PCT%, rest is proportional to score.
    FLOOR_PCT = 3
    NUM = len(results)
    POOL = 100 - FLOOR_PCT * NUM  # 82 distributable points

    total_score = sum(r["score"] for r in results)
    if total_score > 0.5:
        # Enough signal to differentiate — distribute proportionally
        raw_pcts = [FLOOR_PCT + (r["score"] / total_score) * POOL for r in results]
    else:
        # Too little data — distribute roughly equally
        raw_pcts = [100 / NUM] * NUM

    # Round to integers that sum to exactly 100
    int_pcts = [int(p) for p in raw_pcts]
    remainder = 100 - sum(int_pcts)
    fracs = [(raw_pcts[i] - int_pcts[i], i) for i in range(len(int_pcts))]
    fracs.sort(key=lambda x: -x[0])
    for j in range(remainder):
        int_pcts[fracs[j][1]] += 1

    for i, r in enumerate(results):
        r["match_pct"] = int_pcts[i]

    return results


def _smooth(value, midpoint, steepness=1.0):
    """Sigmoid-like smooth scoring: returns 0→1 as value goes from 0→2*midpoint.

    - At value=0 → ~0.0
    - At value=midpoint → 0.5
    - At value=2*midpoint → ~0.88
    - Keeps growing slowly beyond that, capped at 1.0
    """
    import math
    if value <= 0:
        return 0.0
    x = value / max(midpoint, 0.01)
    return min(1.0 - math.exp(-steepness * x), 1.0)


def _score_pipeline_architect(s):
    score = 0.0
    # Jobs count: gradual from 0→15+ (midpoint=6)
    score += _smooth(s["jobs_count"], 6) * 3.0
    # Job reliability: success_rate * volume factor
    if s["total_job_runs"] > 0:
        volume_factor = _smooth(s["total_job_runs"], 20)
        score += s["job_success_rate"] * volume_factor * 2.5
    # Data producer signal: low rw_ratio means writes ≈ reads
    if s["rw_ratio"] != float("inf") and s["rw_ratio"] < 10:
        score += max(0, (1.0 - s["rw_ratio"] / 10.0)) * 2.0
    # Scheduled jobs
    if s["has_cron"]:
        score += 1.0
    # Notebook-to-production
    notebook_queries = s["qs_dict"].get("notebook", 0)
    if notebook_queries > 0 and s["jobs_count"] > 0:
        score += 1.5
    elif notebook_queries > 0:
        score += 0.5
    # Notebook service usage
    score += _smooth(s["svc_dict"].get("notebook", 0), 80) * 0.5
    # CRON trigger volume
    score += _smooth(s.get("cron_runs", 0), 30) * 1.5
    return score


def _score_data_explorer(s):
    score = 0.0
    # SELECT ratio: gradual from 0.5→1.0 (higher = more explorer)
    if s["select_ratio"] > 0.5:
        score += ((s["select_ratio"] - 0.5) / 0.5) * 3.0
    # Tables read count: gradual (midpoint=8)
    score += _smooth(s["tables_read_count"], 8) * 2.5
    # High read/write ratio
    if s["rw_ratio"] != float("inf"):
        score += min(s["rw_ratio"] / 10.0, 1.0) * 1.5
    elif s["tables_read_count"] > 0:
        score += 1.5  # Reads but no writes = pure explorer
    # Dashboard queries (explorers often use dashboards)
    score += _smooth(s["qs_dict"].get("dashboard", 0), 10) * 1.0
    # Query volume (explorers run many queries)
    score += _smooth(s["total_queries"], 50) * 0.5
    return score


def _score_platform_guardian(s):
    score = 0.0
    # Admin service overlap: gradual based on count
    admin_services = {"unityCatalog", "accounts", "permissions", "groups",
                      "iamRole", "tokenManagement", "secrets"}
    overlap = len(admin_services & set(s["svc_dict"].keys()))
    score += _smooth(overlap, 3) * 3.0
    # Cluster management
    score += _smooth(s["clusters_in_services"], 50) * 2.0
    # Unity Catalog usage volume
    score += _smooth(s["svc_dict"].get("unityCatalog", 0), 80) * 1.5
    # Cluster service volume
    score += _smooth(s["svc_dict"].get("clusters", 0), 150) * 1.5
    # Governance: permission updates
    score += _smooth(s.get("permission_updates", 0), 3) * 1.5
    # Tables created (governance aspect)
    score += _smooth(s.get("tables_created_count", 0), 5) * 0.5
    return score


def _score_dashboard_crafter(s):
    score = 0.0
    dashboard_queries = s["qs_dict"].get("dashboard", 0)
    direct_sql = s["qs_dict"].get("direct_sql", 0)
    total_qs = dashboard_queries + direct_sql + 1  # avoid div-by-zero
    # Dashboard query dominance (ratio of dashboard queries)
    if dashboard_queries > 0:
        dash_ratio = dashboard_queries / total_qs
        score += dash_ratio * 4.0
    # Dashboard service events
    score += _smooth(s["svc_dict"].get("dashboards", 0), 50) * 2.0
    # High SELECT ratio (dashboards are mostly reads)
    if s["select_ratio"] > 0.6:
        score += ((s["select_ratio"] - 0.6) / 0.4) * 1.0
    # Unique dashboards
    score += _smooth(s.get("unique_dashboards", 0), 5) * 2.0
    return score


def _score_ml_alchemist(s):
    score = 0.0
    # ML/AI service signals
    ml_services = {"mlflow", "modelServing", "serving-endpoints",
                   "feature-store", "vectorSearch", "model-registry"}
    ml_overlap = len(ml_services & set(s["svc_dict"].keys()))
    score += _smooth(ml_overlap, 2) * 3.0
    # Heavy compute ratio (model training = heavy queries)
    score += min(s["heavy_ratio"] / 0.3, 1.0) * 2.5
    # Notebook-heavy workflow
    notebook_queries = s["qs_dict"].get("notebook", 0)
    direct_sql = s["qs_dict"].get("direct_sql", 0)
    if notebook_queries > 0:
        nb_ratio = notebook_queries / (notebook_queries + direct_sql + 1)
        score += nb_ratio * 1.5
    # Mixed read/write (feature tables, model artifacts) — only counts
    # when there's actual ML service usage to avoid false positives
    if ml_overlap > 0 and s["write_ratio"] > 0.03 and s["select_ratio"] > 0.4:
        score += min(s["write_ratio"] / 0.15, 1.0) * 1.0
    # GenAI endpoints
    score += _smooth(s.get("genai_endpoint_count", 0), 2) * 2.5
    return score


def _score_cost_conscious(s):
    score = 0.0
    # Serverless usage signals
    for key in s["qs_dict"]:
        if "serverless" in key.lower():
            score += 2.0
            break
    for key in s["svc_dict"]:
        if "serverless" in key.lower():
            score += 1.0
            break
    # Moderate query volume (not excessive) — require at least 10 queries to score
    if 10 < s["total_queries"] < 500:
        score += (1.0 - s["total_queries"] / 500.0) * 1.5
    # Efficient queries (low heavy ratio)
    if s["total_queries"] > 10:
        score += max(0, 1.0 - s["heavy_ratio"] / 0.15) * 1.5
    # Low DBU share of workspace
    if s.get("workspace_dbu", 0) > 0 and s.get("user_dbu", 0) > 0:
        share = s["user_dbu"] / s["workspace_dbu"]
        if share < 0.1:
            score += 1.5
        elif share < 0.3:
            score += 0.5
    return score


def _compute_badges(data, signals):
    """Compute achievement badges based on persona data and scoring signals."""
    badges = []

    # ── Activity-based badges ──
    activity = data.get("activity", {})
    summary = activity.get("summary", [{}])
    total_events = summary[0].get("total_events", 0) if summary else 0
    active_days = summary[0].get("active_days", 0) if summary else 0
    clients = activity.get("clients", [])
    services = activity.get("services", [])
    hourly = activity.get("hourly_pattern", [])

    peak_hour = max(hourly, key=lambda h: h.get("event_count", 0), default={}).get("hour_of_day", 12) if hourly else 12

    if signals["total_queries"] >= 100:
        badges.append({"id": "sql-centurion", "name": "SQL Centurion", "icon": "\u2694\ufe0f",
                        "description": "Executed 100+ SQL queries in the analysis period"})

    if peak_hour >= 20 or peak_hour <= 3:
        badges.append({"id": "night-owl", "name": "Night Owl", "icon": "\U0001f989",
                        "description": "Peak activity after 20:00 UTC"})

    if 4 <= peak_hour <= 7:
        badges.append({"id": "early-bird", "name": "Early Bird", "icon": "\U0001f426",
                        "description": "Peak activity before 08:00 UTC"})

    jobs = data.get("jobs", {})
    job_runs = jobs.get("job_runs", [])
    owned_jobs = jobs.get("owned_jobs", [])
    total_runs = sum(jr.get("total_runs", 0) for jr in job_runs)
    total_successes = sum(jr.get("successes", 0) for jr in job_runs)
    if len(owned_jobs) >= 10 and total_runs > 0 and (total_successes / total_runs) > 0.95:
        badges.append({"id": "pipeline-guru", "name": "Pipeline Guru", "icon": "\U0001f680",
                        "description": "10+ jobs with over 95% success rate"})

    lineage = data.get("lineage", {})
    catalogs = lineage.get("catalog_usage", [])
    if len(catalogs) >= 5:
        badges.append({"id": "data-cartographer", "name": "Data Cartographer", "icon": "\U0001f5fa\ufe0f",
                        "description": "Accessed data across 5+ catalogs"})

    if total_runs >= 10 and total_successes == total_runs:
        badges.append({"id": "zero-downtime", "name": "Zero Downtime", "icon": "\u2705",
                        "description": "100% job success rate"})

    if len(clients) >= 4:
        badges.append({"id": "polyglot", "name": "Polyglot", "icon": "\U0001f310",
                        "description": "Uses 4+ different client tools"})

    collab = data.get("collaboration", {})
    downstream = collab.get("downstream_consumers", [])
    if len(downstream) >= 5:
        badges.append({"id": "team-player", "name": "Team Player", "icon": "\U0001f91d",
                        "description": "5+ users depend on your data"})

    tables_read = lineage.get("tables_read", [])
    tables_written = lineage.get("tables_written", [])
    if tables_read and tables_written and owned_jobs:
        badges.append({"id": "full-stack-data", "name": "Full Stack Data", "icon": "\U0001f4e6",
                        "description": "Reads, writes, and runs jobs"})

    serverless_found = any("serverless" in str(s).lower() for s in services) or \
                       any("serverless" in str(qs.get("source_type", "")).lower()
                           for qs in data.get("queries", {}).get("query_sources", []))
    if serverless_found:
        badges.append({"id": "serverless-pioneer", "name": "Serverless Pioneer", "icon": "\u2601\ufe0f",
                        "description": "Embraces serverless compute"})

    if active_days >= 30:
        badges.append({"id": "marathon-runner", "name": "Marathon Runner", "icon": "\U0001f3c3",
                        "description": "Active on 30+ days"})

    if len(services) >= 10:
        badges.append({"id": "service-connoisseur", "name": "Service Connoisseur", "icon": "\U0001f3af",
                        "description": "Used 10+ different services"})

    if total_events >= 10000:
        badges.append({"id": "power-user", "name": "Power User", "icon": "\u26a1",
                        "description": "10,000+ platform events"})

    queries = data.get("queries", {})
    error_stats = queries.get("error_stats", [{}])
    total_q = error_stats[0].get("total_queries", 0) if error_stats else 0
    successful_q = error_stats[0].get("successful", 0) if error_stats else 0
    if total_q >= 50 and total_q > 0 and (successful_q / total_q) > 0.95:
        badges.append({"id": "query-sniper", "name": "Query Sniper", "icon": "\U0001f3af",
                        "description": "95%+ query success rate"})

    if len(tables_read) >= 20:
        badges.append({"id": "table-hoarder", "name": "Table Hoarder", "icon": "\U0001f4da",
                        "description": "Reads from 20+ unique tables"})

    # ── Phase 8 badges ──
    engagement = data.get("engagement", {})
    longest_streak = engagement.get("longest_streak", 0)
    if longest_streak >= 30:
        badges.append({"id": "iron-streak", "name": "Iron Streak", "icon": "\U0001f525",
                        "description": f"{longest_streak}-day activity streak"})

    heatmap = engagement.get("heatmap", [])
    active_hours = len({h.get("hour_of_day") for h in heatmap if h.get("event_count", 0) > 0})
    if active_hours >= 20:
        badges.append({"id": "24-7-operator", "name": "24/7 Operator", "icon": "\U0001f570\ufe0f",
                        "description": "Active across 20+ hours of the day"})

    genai = data.get("genai", {})
    genai_endpoints = {e.get("endpoint_name", "") for e in genai.get("user_endpoints", [])}
    if len(genai_endpoints) >= 3:
        badges.append({"id": "ai-pioneer", "name": "AI Pioneer", "icon": "\U0001f916",
                        "description": "Uses 3+ GenAI/LLM endpoints"})

    git = data.get("git_activity", {})
    git_commits = sum(r.get("cnt", 0) for r in git.get("actions", []) if r.get("action_name") == "commitAndPush")
    if git_commits >= 10:
        badges.append({"id": "version-controller", "name": "Version Controller", "icon": "\U0001f4cb",
                        "description": f"{git_commits} Git commits"})

    dash = data.get("dashboard_activity", {})
    unique_dash = len(dash.get("unique_dashboards", []))
    if unique_dash >= 5:
        badges.append({"id": "dashboard-maestro", "name": "Dashboard Maestro", "icon": "\U0001f3a8",
                        "description": f"Active on {unique_dash} dashboards"})

    tg = data.get("table_governance", {})
    gov_actions = tg.get("governance", [])
    perm_updates = sum(r.get("cnt", 0) for r in gov_actions if r.get("action_name") == "updatePermissions")
    if perm_updates >= 5:
        badges.append({"id": "governance-champion", "name": "Governance Champion", "icon": "\U0001f3db\ufe0f",
                        "description": f"{perm_updates} permission updates"})

    tables_created = tg.get("tables_created", [])
    if len(tables_created) >= 10:
        badges.append({"id": "table-builder", "name": "Table Builder", "icon": "\U0001f3d7\ufe0f",
                        "description": f"Created {len(tables_created)}+ unique tables"})

    cost = data.get("cost", {})
    cost_share = cost.get("cost_share", [{}])
    user_dbu = float(cost_share[0].get("user_dbu", 0)) if cost_share else 0
    workspace_dbu = float(cost_share[0].get("workspace_total_dbu", 1)) if cost_share else 1
    if workspace_dbu > 0 and user_dbu > 0 and (user_dbu / workspace_dbu) < 0.3:
        badges.append({"id": "budget-hawk", "name": "Budget Hawk", "icon": "\U0001f4b0",
                        "description": "Below-average DBU consumption"})

    # ── Rank badges (workspace percentile) ──
    comparison = data.get("comparison", {})
    event_pct = comparison.get("event_percentile", 0)
    query_pct = comparison.get("query_percentile", 0)
    ws_users = comparison.get("workspace_users", 0)

    if event_pct >= 90 and ws_users >= 10:
        badges.append({"id": "top-10-active", "name": "Top 10% Active",
                        "icon": "\U0001f3c6",
                        "description": f"More platform activity than {event_pct}% of workspace users",
                        "rank": True})
    elif event_pct >= 75 and ws_users >= 10:
        badges.append({"id": "top-25-active", "name": "Top 25% Active",
                        "icon": "\U0001f3c5",
                        "description": f"More platform activity than {event_pct}% of workspace users",
                        "rank": True})

    if query_pct >= 90 and ws_users >= 10:
        badges.append({"id": "top-10-sql", "name": "Top 10% SQL User",
                        "icon": "\U0001f947",
                        "description": f"More SQL queries than {query_pct}% of workspace users",
                        "rank": True})
    elif query_pct >= 75 and ws_users >= 10:
        badges.append({"id": "top-25-sql", "name": "Top 25% SQL User",
                        "icon": "\U0001f948",
                        "description": f"Runs more SQL queries than {query_pct}% of workspace users",
                        "rank": True})

    return badges


def _infer_expertise(data, signals):
    """Infer expertise areas from data patterns."""
    expertise = []

    lineage = data.get("lineage", {})
    catalogs = lineage.get("catalog_usage", [])
    for c in catalogs[:3]:
        name = c.get("catalog_name", "")
        if name:
            expertise.append(f"{name} catalog")

    if signals["jobs_count"] > 0:
        expertise.append("Job orchestration")
    if signals["write_ratio"] > 0.1:
        expertise.append("Data pipeline development")
    if signals["qs_dict"].get("dashboard", 0) > 0:
        expertise.append("Dashboard / BI development")
    if signals["qs_dict"].get("notebook", 0) > 0:
        expertise.append("Notebook-based development")

    activity = data.get("activity", {})
    services = activity.get("services", [])
    for s in services[:3]:
        name = s.get("service_name", "")
        if name and name not in expertise:
            expertise.append(name)

    return expertise[:8]


def _serialize(rows):
    """Convert query results to JSON-safe format."""
    if not rows:
        return []
    result = []
    for row in rows:
        clean = {}
        for k, v in row.items():
            if hasattr(v, "isoformat"):
                clean[k] = v.isoformat()
            elif isinstance(v, (dict, list)):
                clean[k] = str(v)
            else:
                clean[k] = v
        result.append(clean)
    return result
