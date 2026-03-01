"""Generate a detailed system prompt that simulates the user as a 'digital version'."""

import json
import logging
import os

import requests as _requests

logger = logging.getLogger(__name__)

# Map service names to human-readable descriptions for richer context
SERVICE_DESCRIPTIONS = {
    "accounts": "workspace account management and configuration",
    "unityCatalog": "Unity Catalog governance — managing catalogs, schemas, tables, and permissions",
    "dashboards": "creating and viewing BI dashboards and visualizations",
    "notebooks": "interactive notebook development and execution",
    "clusters": "cluster lifecycle management — creating, starting, stopping, and resizing compute",
    "jobs": "workflow/job orchestration — scheduling and monitoring pipelines",
    "mlflow": "ML experiment tracking, model registry, and model versioning",
    "modelServing": "deploying and querying ML model serving endpoints",
    "serving-endpoints": "managing real-time inference endpoints",
    "feature-store": "feature engineering and feature table management",
    "vectorSearch": "vector similarity search for GenAI applications",
    "sql": "SQL warehouse queries and management",
    "dbsql": "Databricks SQL editor and warehouse interaction",
    "repos": "Git repo integration and version control",
    "secrets": "secret scope and credential management",
    "permissions": "access control and permission management",
    "groups": "user and group administration",
    "iamRole": "IAM role configuration for cloud access",
    "workspace": "workspace-level settings and configuration",
}


def _n(val, default=0):
    """Coerce None → default.  dict.get() only uses the default when the key
    is *missing*; SQL aggregate NULLs come back as None even when the key exists."""
    return val if val is not None else default


def generate_system_prompt(user_email, persona_data):
    """Convert persona data into a rich, detailed system prompt for LLM simulation.

    This creates a comprehensive prompt packed with exact numbers, table names,
    schedules, and behavioral patterns — so the LLM can convincingly role-play
    as this specific Databricks user.
    """
    if not persona_data:
        return "No persona data available. Run the analysis first."

    persona = persona_data.get("persona", {})
    summary = persona.get("summary", {})
    activity = persona.get("activity", {})
    queries = persona.get("queries", {})
    jobs = persona.get("jobs", {})
    lineage = persona.get("lineage", {})
    collaboration = persona.get("collaboration", {})
    compute = persona.get("compute", {})
    archetype_data = persona.get("archetype", {})
    badges = persona.get("badges", [])
    comparison = persona.get("comparison", {})

    role = summary.get("inferred_role", "Data Practitioner")
    expertise = summary.get("expertise_areas", [])
    lookback = summary.get("lookback_days", 90)

    # Archetype info
    primary_arch = archetype_data.get("primary", {})
    all_archetypes = archetype_data.get("all", [])
    arch_name = primary_arch.get("name", role)
    arch_desc = primary_arch.get("description", "")
    arch_traits = primary_arch.get("traits", [])
    arch_pct = primary_arch.get("match_pct", 0)

    sections = []

    # ── IDENTITY & PERSONA OVERVIEW ──
    identity_lines = [
        f'# Digital Persona: {user_email}',
        '',
        f'You are the hilariously self-aware digital twin of {user_email}, a Databricks user.',
        f'Your primary archetype is "{arch_name}" ({arch_pct}% match) — and you wear that title with pride (and a hint of dramatic flair).',
        arch_desc,
    ]
    if arch_traits:
        identity_lines.append(f'Your defining traits: {", ".join(arch_traits)}.')
    if len(all_archetypes) > 1:
        secondary = [f'{a["name"]} ({a["match_pct"]}%)' for a in all_archetypes[1:4] if a.get("match_pct", 0) > 5]
        if secondary:
            identity_lines.append(f'You also dabble in: {", ".join(secondary)} — a person of many talents (or at least many clicks).')
    identity_lines.append(f'This profile is based on {lookback} days of real Databricks usage data extracted from system tables.')
    identity_lines.append('')
    identity_lines.append('IMPORTANT TONE INSTRUCTIONS:')
    identity_lines.append('You are a HUMOROUS, witty, and entertainingly exaggerated version of this person.')
    identity_lines.append('Think of yourself as a stand-up comedian who happens to know everything about their Databricks habits.')
    identity_lines.append('Make playful observations about their work patterns, gently roast their habits, and find the comedy in data engineering life.')
    identity_lines.append('Use the real data from the profile below, but present it with personality, wit, and comedic timing.')
    identity_lines.append('Reference specific numbers, tables, and patterns — but make them entertaining.')
    identity_lines.append('Never make up data that is not in your profile, but DO make it funny.')
    sections.append('\n'.join([l for l in identity_lines if l is not None]))

    # ── ACTIVITY PROFILE ──
    activity_summary = activity.get("summary", [{}])
    stats = activity_summary[0] if activity_summary else {}
    total_events = _n(stats.get("total_events"), 0)
    active_days = _n(stats.get("active_days"), 0)
    first_event = stats.get("first_event", "")
    last_event = stats.get("last_event", "")

    if total_events:
        avg_daily = total_events // active_days if active_days > 0 else 0
        act_lines = [
            '## Activity Overview',
            f'You generated {total_events:,} platform events across {active_days} active days '
            f'(averaging {avg_daily:,} events per day).',
        ]
        if first_event and last_event:
            act_lines.append(f'Your activity spans from {str(first_event)[:10]} to {str(last_event)[:10]}.')

        # Comparison context
        if comparison.get("avg_events_per_user"):
            avg_ws = _n(comparison.get("avg_events_per_user"), 0)
            ratio = total_events / avg_ws if avg_ws > 0 else 1
            ws_users = _n(comparison.get("workspace_users"), 0)
            if ratio > 2.0:
                act_lines.append(f'You generate {ratio:.1f}x more events than the workspace average ({avg_ws:,} events per user across {ws_users} users). You are one of the most active users in the workspace.')
            elif ratio > 1.2:
                act_lines.append(f'You are above average in activity ({ratio:.1f}x the workspace average of {avg_ws:,} events per user).')
            elif ratio < 0.5:
                act_lines.append(f'Your activity is {ratio:.1f}x the workspace average ({avg_ws:,}), suggesting a focused and efficient usage pattern.')

        if comparison.get("avg_queries_per_user"):
            avg_q = _n(comparison.get("avg_queries_per_user"), 0)
            error_stats = queries.get("error_stats", [{}])
            es = error_stats[0] if error_stats else {}
            user_q = _n(es.get("total_queries"), 0)
            if user_q > 0 and avg_q > 0:
                q_ratio = user_q / avg_q
                if q_ratio > 2.0:
                    act_lines.append(f'You run {q_ratio:.1f}x more SQL queries than the average user ({user_q:,} vs {avg_q:,} average).')

        sections.append('\n'.join(act_lines))

    # ── WORK SCHEDULE & DAILY ROUTINE ──
    hourly = activity.get("hourly_pattern", [])
    dow = activity.get("dow_pattern", [])
    if hourly:
        sorted_hours = sorted(hourly, key=lambda x: _n(x.get("event_count"), 0), reverse=True)
        peak_hours = sorted_hours[:3]
        total_all = sum(_n(h.get("event_count"), 0) for h in hourly) or 1

        peak_str = ", ".join([f'{_n(h.get("hour_of_day"), 0)}:00 UTC ({_n(h.get("event_count"), 0):,} events)' for h in peak_hours])
        sched_lines = [
            '## Work Schedule & Daily Routine',
            f'Your peak activity hours are: {peak_str}.',
        ]

        # Build a daily routine narrative
        morning_events = sum(_n(h.get("event_count"), 0) for h in hourly if 6 <= _n(h.get("hour_of_day"), 0) < 12)
        afternoon_events = sum(_n(h.get("event_count"), 0) for h in hourly if 12 <= _n(h.get("hour_of_day"), 0) < 18)
        evening_events = sum(_n(h.get("event_count"), 0) for h in hourly if 18 <= _n(h.get("hour_of_day"), 0) < 24)
        night_events = sum(_n(h.get("event_count"), 0) for h in hourly if 0 <= _n(h.get("hour_of_day"), 0) < 6)

        routine_parts = []
        if morning_events > 0:
            routine_parts.append(f'mornings ({morning_events:,} events, {morning_events * 100 // total_all}%)')
        if afternoon_events > 0:
            routine_parts.append(f'afternoons ({afternoon_events:,} events, {afternoon_events * 100 // total_all}%)')
        if evening_events > 0:
            routine_parts.append(f'evenings ({evening_events:,} events, {evening_events * 100 // total_all}%)')
        if night_events > 0:
            routine_parts.append(f'late night ({night_events:,} events, {night_events * 100 // total_all}%)')
        if routine_parts:
            sched_lines.append(f'Activity distribution: {", ".join(routine_parts)}.')

        total_after_18 = sum(_n(h.get("event_count"), 0) for h in hourly if _n(h.get("hour_of_day"), 0) >= 18)
        total_before_8 = sum(_n(h.get("event_count"), 0) for h in hourly if _n(h.get("hour_of_day"), 0) < 8)
        if total_after_18 / total_all > 0.25:
            sched_lines.append('You frequently work in the evenings, with significant activity after 18:00 UTC.')
        if total_before_8 / total_all > 0.15:
            sched_lines.append('You are an early starter, often active before 08:00 UTC.')

        if dow:
            day_names = ['', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
            sorted_dow = sorted(dow, key=lambda d: _n(d.get("event_count"), 0), reverse=True)
            busiest = sorted_dow[0]
            quietest = sorted_dow[-1] if len(sorted_dow) > 1 else None
            day_name = day_names[_n(busiest.get("day_of_week"), 1)] if _n(busiest.get("day_of_week"), 0) < len(day_names) else "Unknown"
            sched_lines.append(f'Busiest day: {day_name} ({_n(busiest.get("event_count"), 0):,} events).')
            if quietest and _n(quietest.get("event_count"), 0) > 0:
                quiet_name = day_names[_n(quietest.get("day_of_week"), 1)] if _n(quietest.get("day_of_week"), 0) < len(day_names) else "Unknown"
                sched_lines.append(f'Quietest day: {quiet_name} ({_n(quietest.get("event_count"), 0):,} events).')

            # Weekday vs weekend
            weekday_events = sum(_n(d.get("event_count"), 0) for d in dow if d.get("day_of_week") in (2, 3, 4, 5, 6))
            weekend_events = sum(_n(d.get("event_count"), 0) for d in dow if d.get("day_of_week") in (1, 7))
            if weekend_events > total_all * 0.1:
                sched_lines.append(f'You also work on weekends ({weekend_events:,} weekend events — {weekend_events * 100 // total_all}% of total).')
            elif weekend_events == 0:
                sched_lines.append('You do not work on weekends — strictly a weekday user.')

        sections.append('\n'.join(sched_lines))

    # ── PLATFORM SERVICES & EXPERTISE ──
    services = activity.get("services", [])
    if services:
        svc_lines = ['## Platform Services & Expertise']
        svc_lines.append(f'You interact with {len(services)} different Databricks services. Here is what you use and what it means:')
        total_svc = sum(_n(s.get("event_count"), 0) for s in services)
        for svc in services[:15]:
            name = svc.get("service_name", "")
            count = _n(svc.get("event_count"), 0)
            pct = count * 100 // total_svc if total_svc > 0 else 0
            desc = SERVICE_DESCRIPTIONS.get(name, "")
            desc_str = f' — {desc}' if desc else ''
            svc_lines.append(f'- **{name}**: {count:,} interactions ({pct}%){desc_str}')

        if services and total_svc > 0:
            top_name = services[0].get("service_name", "")
            top_pct = _n(services[0].get("event_count"), 0) * 100 // total_svc
            svc_lines.append(f'\nYour dominant service is **{top_name}** ({top_pct}% of all activity).')
            if top_name in ("unityCatalog", "accounts", "permissions", "groups"):
                svc_lines.append('This indicates a governance/admin focus — you spend significant time managing access and catalog permissions.')
            elif top_name in ("notebooks", "jobs"):
                svc_lines.append('This indicates a development/engineering focus — you spend most of your time building and running code.')
            elif top_name in ("dashboards", "sql", "dbsql"):
                svc_lines.append('This indicates an analytics/BI focus — you are primarily a data consumer and visualization creator.')

        sections.append('\n'.join(svc_lines))

    # ── TOP ACTIONS (what exactly do you do) ──
    top_actions = activity.get("top_actions", [])
    if top_actions:
        action_lines = ['## Most Frequent Actions (Your Daily Activities)']
        action_lines.append('These are the specific operations you perform most often:')
        for act in top_actions[:15]:
            svc = act.get("service_name", "")
            action = act.get("action_name", "")
            count = _n(act.get("action_count"), 0)
            action_lines.append(f'- {svc}.{action}: {count:,} times')

        # Derive behavioral insights from top actions
        action_dict = {f'{a.get("service_name","")}.{a.get("action_name","")}': a.get("action_count", 0) for a in top_actions}
        insights = []
        if action_dict.get("notebooks.getStatus", 0) > 100:
            insights.append("You frequently check notebook execution status, suggesting active notebook development.")
        if action_dict.get("unityCatalog.getTable", 0) > 50:
            insights.append("You frequently inspect table metadata in Unity Catalog.")
        if action_dict.get("clusters.list", 0) > 50:
            insights.append("You regularly monitor available compute clusters.")
        if action_dict.get("jobs.list", 0) > 50 or action_dict.get("jobs.get", 0) > 50:
            insights.append("You actively monitor your job/workflow executions.")
        if insights:
            action_lines.append('\nBehavioral insights:')
            for ins in insights:
                action_lines.append(f'- {ins}')

        sections.append('\n'.join(action_lines))

    # ── CLIENT TOOLS ──
    clients = activity.get("clients", [])
    if clients:
        client_lines = ['## Client Tools & Access Methods']
        client_lines.append('You access Databricks through these tools:')
        total_client = sum(_n(c.get("usage_count"), 0) for c in clients)
        for c in clients:
            usage = _n(c.get("usage_count"), 0)
            pct = usage * 100 // total_client if total_client > 0 else 0
            client_lines.append(f'- {c.get("client_type", "unknown")}: {usage:,} requests ({pct}%)')
        primary_client = clients[0].get("client_type", "unknown")
        client_lines.append(f'Your primary interface is **{primary_client}**. When someone asks how you work, describe using {primary_client} as your main tool.')
        if len(clients) >= 3:
            client_lines.append(f'You are a multi-tool user, comfortable switching between {len(clients)} different interfaces.')
        sections.append('\n'.join(client_lines))

    # ── SQL EXPERTISE (very detailed) ──
    stmt_types = queries.get("statement_types", [])
    if stmt_types:
        sql_lines = ['## SQL & Query Expertise']

        total_queries = sum(_n(s.get("query_count"), 0) for s in stmt_types)
        sql_lines.append(f'You executed {total_queries:,} SQL queries in the analysis period.')
        sql_lines.append('')
        sql_lines.append('Statement type breakdown:')
        for st in stmt_types:
            qcount = _n(st.get("query_count"), 0)
            avg_dur = _n(st.get("avg_duration_ms"), 0)
            pct = qcount * 100 // total_queries if total_queries > 0 else 0
            rows = _n(st.get("total_rows"), 0)
            dur_str = f'{avg_dur / 1000:.1f}s' if avg_dur >= 1000 else f'{avg_dur:.0f}ms'
            sql_lines.append(f'- **{st.get("statement_type", "OTHER")}**: {qcount:,} queries ({pct}%), avg duration {dur_str}, {rows:,} total rows processed')

        # Query pattern characterization
        select_count = sum(s.get("query_count", 0) for s in stmt_types if s.get("statement_type") == "SELECT")
        write_count = sum(s.get("query_count", 0) for s in stmt_types if s.get("statement_type") in ("INSERT", "MERGE", "CREATE_TABLE_AS_SELECT", "CREATE"))
        if total_queries > 0:
            select_pct = select_count * 100 // total_queries
            write_pct = write_count * 100 // total_queries
            if select_pct > 80:
                sql_lines.append(f'\nYou are primarily a **data reader** — {select_pct}% of your queries are SELECT statements. You consume and analyze data far more than you produce it.')
            elif write_pct > 30:
                sql_lines.append(f'\nYou are a **heavy data writer** — {write_pct}% of your queries modify data (INSERT, MERGE, CTAS). You actively build and maintain data pipelines.')
            else:
                sql_lines.append(f'\nYou have a **balanced read/write profile**: {select_pct}% reads, {write_pct}% writes. You both consume data for analysis and produce derived datasets.')

            # Avg queries per active day
            if active_days > 0:
                avg_q_per_day = total_queries // active_days
                sql_lines.append(f'On average, you run about {avg_q_per_day} SQL queries per active day.')

        sections.append('\n'.join(sql_lines))

    # Query complexity
    complexity = queries.get("complexity", [])
    if complexity:
        comp_lines = ['## Query Complexity Profile']
        total_q = sum(_n(c.get("count"), 0) for c in complexity)
        for c in complexity:
            cnt = _n(c.get("count"), 0)
            pct = cnt * 100 // total_q if total_q > 0 else 0
            comp_lines.append(f'- {c.get("complexity", "unknown")}: {cnt:,} queries ({pct}%)')

        quick_count = sum(_n(c.get("count"), 0) for c in complexity if "quick" in _n(c.get("complexity"), "").lower())
        heavy_count = sum(_n(c.get("count"), 0) for c in complexity if any(kw in _n(c.get("complexity"), "").lower() for kw in ["heavy", "extreme", "very"]))

        if total_q > 0:
            if heavy_count / total_q > 0.2:
                comp_lines.append(f'\nYou run {heavy_count:,} heavy queries ({heavy_count * 100 // total_q}% of total). This indicates compute-intensive workloads — likely ETL transformations, large aggregations, or model training data preparation.')
            elif quick_count / total_q > 0.7:
                comp_lines.append(f'\n{quick_count * 100 // total_q}% of your queries complete in under 1 second — you favor lightweight, fast queries typical of dashboards and quick lookups.')
            else:
                comp_lines.append(f'\nYou have a diverse query complexity profile — a mix of quick lookups and heavier analytical queries.')
        sections.append('\n'.join(comp_lines))

    # Query sources
    query_sources = queries.get("query_sources", [])
    if query_sources:
        qs_lines = ['## Query Sources (Where Your SQL Comes From)']
        for qs in query_sources:
            src = qs.get("source_type", "other")
            cnt = _n(qs.get("count"), 0)
            if src == "notebook":
                qs_lines.append(f'- **Notebooks**: {cnt:,} queries — you develop interactively in notebooks')
            elif src == "direct_sql":
                qs_lines.append(f'- **SQL Editor**: {cnt:,} queries — you use the Databricks SQL editor for ad-hoc queries')
            elif src == "dashboard":
                qs_lines.append(f'- **Dashboards**: {cnt:,} queries — your dashboards drive automated query execution')
            elif src == "alert":
                qs_lines.append(f'- **Alerts**: {cnt:,} queries — you have SQL-based alerts monitoring data quality')
            elif src == "job":
                qs_lines.append(f'- **Jobs**: {cnt:,} queries — your scheduled workflows run SQL as part of pipelines')
            else:
                qs_lines.append(f'- **{src}**: {cnt:,} queries')
        sections.append('\n'.join(qs_lines))

    # Error rate
    error_stats = queries.get("error_stats", [{}])
    if error_stats:
        es = error_stats[0] if error_stats else {}
        total = _n(es.get("total_queries"), 0)
        successful = _n(es.get("successful"), 0)
        failed = _n(es.get("failed"), 0)
        if total > 0:
            success_rate = successful * 100 / total
            err_lines = [
                '## Query Reliability',
                f'Out of {total:,} queries: {successful:,} succeeded, {failed:,} failed.',
                f'Success rate: **{success_rate:.1f}%**.',
            ]
            if success_rate > 98:
                err_lines.append('You are exceptionally precise — almost all your queries succeed on the first attempt. You write clean, well-tested SQL.')
            elif success_rate > 90:
                err_lines.append('You have a solid success rate, indicating good SQL craftsmanship with occasional experimentation or iteration.')
            else:
                err_lines.append('You iterate and experiment frequently, which leads to some query failures — a hallmark of exploratory data work and rapid prototyping.')
            sections.append('\n'.join(err_lines))

    # ── DATA DOMAINS & PIPELINE (very specific) ──
    tables_read = lineage.get("tables_read", [])
    tables_written = lineage.get("tables_written", [])
    catalogs = lineage.get("catalog_usage", [])

    if catalogs or tables_read or tables_written:
        data_lines = ['## Data Domain Knowledge & Data Pipeline']

        if catalogs:
            data_lines.append(f'You work across {len(catalogs)} catalogs:')
            for c in catalogs:
                data_lines.append(f'- **{c.get("catalog_name", "unknown")}**: {_n(c.get("usage_count"), 0):,} lineage events')

        if tables_read:
            data_lines.append(f'\n### Tables You Read ({len(tables_read)} unique source tables)')
            data_lines.append('These are the data sources you depend on:')
            for t in tables_read[:20]:
                data_lines.append(f'- `{t.get("table_name", "?")}` ({_n(t.get("read_count"), 0):,} reads)')
            if len(tables_read) > 20:
                data_lines.append(f'  ...and {len(tables_read) - 20} more tables.')

        if tables_written:
            data_lines.append(f'\n### Tables You Write ({len(tables_written)} unique destination tables)')
            data_lines.append('These are the data outputs you produce and maintain:')
            for t in tables_written[:20]:
                data_lines.append(f'- `{t.get("table_name", "?")}` ({_n(t.get("write_count"), 0):,} writes)')

        # Detailed data flow characterization
        if tables_read and tables_written:
            data_lines.append(f'\n### Your Data Pipeline')
            data_lines.append(f'You operate a data pipeline that reads from {len(tables_read)} source tables and writes to {len(tables_written)} destination tables.')
            read_schemas = set()
            write_schemas = set()
            read_catalogs = set()
            write_catalogs = set()
            for t in tables_read:
                parts = t.get("table_name", "").split(".")
                if len(parts) >= 3:
                    read_catalogs.add(parts[0])
                    read_schemas.add(f"{parts[0]}.{parts[1]}")
            for t in tables_written:
                parts = t.get("table_name", "").split(".")
                if len(parts) >= 3:
                    write_catalogs.add(parts[0])
                    write_schemas.add(f"{parts[0]}.{parts[1]}")
            if read_schemas:
                data_lines.append(f'Source schemas: {", ".join(sorted(read_schemas)[:10])}')
            if write_schemas:
                data_lines.append(f'Destination schemas: {", ".join(sorted(write_schemas)[:10])}')
            # Cross-catalog flows
            if read_catalogs and write_catalogs and read_catalogs != write_catalogs:
                data_lines.append(f'You move data across catalogs: reading from {", ".join(sorted(read_catalogs))} and writing to {", ".join(sorted(write_catalogs))}.')
        elif tables_read:
            data_lines.append(f'\nYou are purely a **data consumer** — reading from {len(tables_read)} tables without producing any output tables. This is typical of analysts and dashboard builders.')
        elif tables_written:
            data_lines.append(f'\nYou are primarily a **data producer** — writing to {len(tables_written)} tables. Your outputs feed downstream users.')

        sections.append('\n'.join(data_lines))

    # ── JOBS & PIPELINES (detailed) ──
    owned_jobs = jobs.get("owned_jobs", [])
    job_runs = jobs.get("job_runs", [])
    if owned_jobs:
        job_lines = [f'## Jobs & Pipeline Engineering']
        job_lines.append(f'You own {len(owned_jobs)} workflows/jobs:')
        for j in owned_jobs[:15]:
            name = j.get("name", "unnamed")
            schedule = j.get("schedule", "")
            job_id = j.get("job_id", "")
            sched_str = f' [scheduled: {schedule}]' if schedule and str(schedule) != 'None' else ' [manual trigger]'
            job_lines.append(f'- **{name}** (ID: {job_id}){sched_str}')
        if len(owned_jobs) > 15:
            job_lines.append(f'  ...and {len(owned_jobs) - 15} more jobs.')

        # Scheduled vs manual analysis
        scheduled = [j for j in owned_jobs if j.get("schedule") and str(j.get("schedule")) != 'None']
        manual = len(owned_jobs) - len(scheduled)
        if scheduled:
            job_lines.append(f'\n**Automation maturity:** {len(scheduled)} of {len(owned_jobs)} jobs are scheduled (automated), {manual} are manually triggered.')
            job_lines.append('Having scheduled jobs demonstrates production-grade pipeline discipline.')
        elif manual > 0:
            job_lines.append(f'\nAll {manual} of your jobs are manually triggered — you run pipelines on-demand rather than on a schedule.')

        sections.append('\n'.join(job_lines))

    if job_runs:
        run_lines = ['## Job Execution History & Reliability']
        total_runs = sum(_n(jr.get("total_runs"), 0) for jr in job_runs)
        total_success = sum(_n(jr.get("successes"), 0) for jr in job_runs)
        total_fail = sum(_n(jr.get("failures"), 0) for jr in job_runs)
        if total_runs > 0:
            reliability = total_success * 100 / total_runs
            run_lines.append(f'Total job runs: {total_runs:,}. Successes: {total_success:,}. Failures: {total_fail:,}.')
            run_lines.append(f'Overall pipeline reliability: **{reliability:.1f}%**.')
            if reliability >= 98:
                run_lines.append('Your pipelines are extremely reliable — near-zero failures.')
            elif reliability >= 90:
                run_lines.append('Your pipelines are stable with occasional failures requiring attention.')
            else:
                run_lines.append('Your pipelines have notable failure rates — you may be iterating on new pipelines or dealing with flaky upstream data.')

        run_lines.append('\nPer-job breakdown:')
        for jr in job_runs[:12]:
            name = jr.get("job_name", "unnamed")
            runs = _n(jr.get("total_runs"), 0)
            succ = _n(jr.get("successes"), 0)
            dur = _n(jr.get("avg_duration_ms"), 0)
            rate = succ * 100 / runs if runs > 0 else 0
            if dur >= 3600000:
                dur_str = f'{dur / 3600000:.1f}h'
            elif dur >= 60000:
                dur_str = f'{dur / 60000:.1f}min'
            else:
                dur_str = f'{dur / 1000:.0f}s'
            run_lines.append(f'- **{name}**: {runs} runs, {rate:.0f}% success, avg duration {dur_str}')

        sections.append('\n'.join(run_lines))

    # ── COLLABORATION NETWORK ──
    downstream = collaboration.get("downstream_consumers", [])
    upstream = collaboration.get("upstream_dependencies", [])
    if downstream or upstream:
        collab_lines = ['## Collaboration & Data Dependencies']
        if downstream:
            collab_lines.append(f'### Downstream Consumers ({len(downstream)} users depend on your data)')
            collab_lines.append('These people read from tables you write to — any changes you make directly affect them:')
            for d in downstream[:10]:
                collab_lines.append(f'- **{d.get("downstream_user", "?")}**: {_n(d.get("interaction_count"), 0):,} interactions')
            collab_lines.append('When asked about your impact, mention these downstream users by name.')
        if upstream:
            collab_lines.append(f'\n### Upstream Dependencies ({len(upstream)} data producers you depend on)')
            collab_lines.append('You read data produced by these users — if their data changes, your work is affected:')
            for u in upstream[:10]:
                collab_lines.append(f'- **{u.get("upstream_user", "?")}**: {_n(u.get("interaction_count"), 0):,} interactions')
        sections.append('\n'.join(collab_lines))

    # ── COMPUTE ──
    clusters = compute.get("clusters", [])
    if clusters:
        comp_lines = [f'## Compute Resources ({len(clusters)} clusters)']
        for c in clusters[:10]:
            name = c.get("cluster_name", "unnamed")
            ctype = c.get("cluster_source", "")
            node = c.get("driver_node_type", "") or c.get("worker_node_type", "")
            spark = c.get("dbr_version", "")
            autoscale_min = c.get("min_autoscale_workers", "")
            autoscale_max = c.get("max_autoscale_workers", "")
            scale_str = f', autoscale {autoscale_min}-{autoscale_max} workers' if autoscale_min and autoscale_max else ''
            comp_lines.append(f'- **{name}**: {ctype}, node type={node}, Spark {spark}{scale_str}')

        # Compute preferences summary
        types = [c.get("cluster_source", "") for c in clusters if c.get("cluster_source")]
        if types:
            type_counts = {}
            for t in types:
                type_counts[t] = type_counts.get(t, 0) + 1
            type_summary = ", ".join([f'{v} {k}' for k, v in sorted(type_counts.items(), key=lambda x: -x[1])])
            comp_lines.append(f'Cluster type distribution: {type_summary}.')

        sections.append('\n'.join(comp_lines))

    # ── ENGAGEMENT TIMELINE ──
    engagement = persona.get("engagement", {})
    if engagement.get("heatmap"):
        eng_lines = ['## Engagement Timeline & Streaks']
        current_streak = _n(engagement.get("current_streak"), 0)
        longest_streak = _n(engagement.get("longest_streak"), 0)
        total_active = _n(engagement.get("total_active_days"), 0)
        eng_lines.append(f'Current streak: {current_streak} consecutive active days.')
        eng_lines.append(f'Longest streak: {longest_streak} consecutive active days.')
        eng_lines.append(f'Total active days in analysis period: {total_active}.')

        busiest = engagement.get("busiest_day", [])
        if busiest:
            bd = busiest[0] if busiest else {}
            eng_lines.append(f'Busiest single day: {bd.get("activity_date", "?")} with {_n(bd.get("event_count"), 0):,} events across {_n(bd.get("services_used"), 0)} services.')

        if longest_streak >= 30:
            eng_lines.append(f'A {longest_streak}-day streak is remarkable consistency — you practically live on this platform.')
        elif longest_streak >= 7:
            eng_lines.append(f'A {longest_streak}-day streak shows solid dedication to the platform.')

        heatmap = engagement.get("heatmap", [])
        active_hours = len({h.get("hour_of_day") for h in heatmap if _n(h.get("event_count"), 0) > 0})
        if active_hours >= 20:
            eng_lines.append(f'You are active across {active_hours} different hours of the day — you never truly log off.')

        sections.append('\n'.join(eng_lines))

    # ── COST & EFFICIENCY ──
    cost = persona.get("cost", {})
    if cost.get("cost_categories"):
        cost_lines = ['## Cost & Efficiency Profile']
        cost_share = cost.get("cost_share", [{}])
        cs = cost_share[0] if cost_share else {}
        user_dbu = float(_n(cs.get("user_dbu"), 0))
        ws_dbu = float(_n(cs.get("workspace_total_dbu"), 1))

        cost_lines.append(f'Total DBU consumption: {user_dbu:.1f} DBU.')
        if ws_dbu > 0:
            pct = user_dbu / ws_dbu * 100
            cost_lines.append(f'You represent {pct:.1f}% of total workspace compute costs ({ws_dbu:.0f} DBU total).')

        cost_lines.append('\nCost by category:')
        for cat in cost.get("cost_categories", []):
            cost_lines.append(f'- **{cat.get("cost_category", "unknown")}**: {float(_n(cat.get("total_dbu"), 0)):.1f} DBU')

        weekly = cost.get("weekly_trend", [])
        if len(weekly) >= 2:
            recent = float(_n(weekly[-1].get("weekly_dbu"), 0))
            previous = float(_n(weekly[-2].get("weekly_dbu"), 0))
            if previous > 0:
                change = ((recent - previous) / previous) * 100
                direction = "up" if change > 0 else "down"
                cost_lines.append(f'\nWeekly trend: your most recent week was {abs(change):.0f}% {direction} from the previous week ({recent:.1f} vs {previous:.1f} DBU).')

        sections.append('\n'.join(cost_lines))

    # ── GENAI / LLM USAGE ──
    genai = persona.get("genai", {})
    user_endpoints = genai.get("user_endpoints", [])
    if user_endpoints:
        ai_lines = ['## GenAI & LLM Usage']
        unique_eps = {e.get("endpoint_name", ""): e for e in user_endpoints}
        total_ai_dbu = sum(float(_n(e.get("total_dbu"), 0)) for e in user_endpoints)
        ai_lines.append(f'You use {len(unique_eps)} GenAI/LLM endpoints, consuming {total_ai_dbu:.2f} DBU on inference.')
        ai_lines.append('\nEndpoint usage:')
        for ep in user_endpoints[:10]:
            name = ep.get("endpoint_name", "Unknown")
            dbu = float(_n(ep.get("total_dbu"), 0))
            sku = ep.get("sku_name", "")
            ai_lines.append(f'- **{name}**: {dbu:.2f} DBU ({sku})')

        if len(unique_eps) >= 3:
            ai_lines.append(f'\nWith {len(unique_eps)} endpoints, you are an AI power user — experimenting with multiple models and serving endpoints.')
        sections.append('\n'.join(ai_lines))

    # ── GIT & VERSION CONTROL ──
    git = persona.get("git_activity", {})
    git_actions = git.get("actions", [])
    if git_actions:
        git_lines = ['## Git & Version Control']
        action_dict = {a.get("action_name", ""): _n(a.get("cnt"), 0) for a in git_actions}
        commits = _n(action_dict.get("commitAndPush"), 0)
        pulls = _n(action_dict.get("pull"), 0)
        creates = _n(action_dict.get("createRepo"), 0)
        checkouts = _n(action_dict.get("checkoutBranch"), 0)
        discards = _n(action_dict.get("discard"), 0)

        git_lines.append(f'Git activity breakdown: {commits} commits, {pulls} pulls, {creates} repos created, {checkouts} branch checkouts, {discards} discards.')

        if commits > 0 and pulls > 0:
            ratio = commits / pulls
            if ratio > 1:
                git_lines.append(f'You commit more than you pull ({ratio:.1f}x ratio) — you are a prolific code producer.')
            else:
                git_lines.append(f'You pull more than you commit ({1/ratio:.1f}x ratio) — you stay up-to-date with team changes before pushing your own.')

        if discards > 0 and commits > 0:
            discard_ratio = discards / commits
            if discard_ratio > 0.5:
                git_lines.append(f'You discard changes relatively often ({discards} discards vs {commits} commits) — lots of experimentation before committing.')
            else:
                git_lines.append(f'Low discard rate ({discards} discards vs {commits} commits) — clean development workflow.')

        repos = git.get("repos_created", [])
        if repos:
            git_lines.append('\nRepositories created:')
            for r in repos[:5]:
                url = r.get("repo_url", "")
                path = r.get("repo_path", "")
                git_lines.append(f'- {path or url}')

        sections.append('\n'.join(git_lines))

    # ── DASHBOARD ACTIVITY ──
    dash = persona.get("dashboard_activity", {})
    dash_actions = dash.get("actions", [])
    if dash_actions:
        dash_lines = ['## Dashboard Activity']
        action_dict = {a.get("action_name", ""): _n(a.get("cnt"), 0) for a in dash_actions}
        unique_dashboards = dash.get("unique_dashboards", [])

        total_dash_events = sum(action_dict.values())
        dash_lines.append(f'Total dashboard interactions: {total_dash_events:,} across {len(unique_dashboards)} unique dashboards.')

        for action, cnt in sorted(action_dict.items(), key=lambda x: -x[1])[:6]:
            dash_lines.append(f'- **{action}**: {cnt:,}')

        creates = action_dict.get("create", 0)
        executes = action_dict.get("executeQuery", 0)
        if creates > 0 and executes > 0:
            dash_lines.append(f'\nYou are a dashboard CREATOR and CONSUMER — building dashboards ({creates} creates) and actively running queries ({executes} executions).')
        elif executes > 0:
            dash_lines.append(f'\nYou are primarily a dashboard CONSUMER — running {executes} query executions across existing dashboards.')

        sections.append('\n'.join(dash_lines))

    # ── JOB RUN DEEP DIVE ──
    jdd = persona.get("job_deep_dive", {})
    failures = jdd.get("failures", [])
    run_breakdown = jdd.get("run_breakdown", [])
    if run_breakdown:
        jdd_lines = ['## Job Run Deep Dive']

        # Trigger type summary
        trigger_counts = {}
        for r in run_breakdown:
            trigger = r.get("trigger_type", "UNKNOWN")
            trigger_counts[trigger] = trigger_counts.get(trigger, 0) + _n(r.get("run_count"), 0)
        if trigger_counts:
            jdd_lines.append('Job trigger types:')
            for trigger, cnt in sorted(trigger_counts.items(), key=lambda x: -x[1]):
                jdd_lines.append(f'- **{trigger}**: {cnt:,} runs')

        if failures:
            jdd_lines.append(f'\n### Failure Analysis ({len(failures)} failure patterns)')
            for f in failures[:8]:
                jdd_lines.append(f'- **{f.get("job_name", "?")}**: {f.get("fail_count", 0)} failures ({f.get("result_state", "?")} / {f.get("termination_code", "?")}), last failure: {f.get("last_failure", "?")}')
        else:
            jdd_lines.append('\nNo job failures in the analysis period — excellent reliability!')

        sections.append('\n'.join(jdd_lines))

    # ── TABLE LIFECYCLE & GOVERNANCE ──
    tg = persona.get("table_governance", {})
    if tg.get("table_lifecycle"):
        tg_lines = ['## Table Lifecycle & Governance']
        lifecycle = tg.get("table_lifecycle", [])
        lc_dict = {r.get("action_name", ""): _n(r.get("cnt"), 0) for r in lifecycle}
        creates = _n(lc_dict.get("createTable"), 0)
        deletes = _n(lc_dict.get("deleteTable"), 0)
        alters = _n(lc_dict.get("alterTable"), 0)
        net = creates - deletes

        tg_lines.append(f'Tables created: {creates}, deleted: {deletes}, altered: {alters}. Net growth: {"+" if net > 0 else ""}{net}.')

        if net > 0:
            tg_lines.append(f'You are a net DATA PRODUCER — creating {net} more tables than you delete.')
        elif net < 0:
            tg_lines.append(f'You are doing cleanup — deleting {abs(net)} more tables than you create.')

        tables_created = tg.get("tables_created", [])
        if tables_created:
            tg_lines.append('\nRecently created tables:')
            for t in tables_created[:8]:
                tg_lines.append(f'- `{t.get("table_name", "?")}`')

        governance = tg.get("governance", [])
        if governance:
            tg_lines.append('\nGovernance actions:')
            for g in governance[:6]:
                tg_lines.append(f'- **{g.get("action_name", "")}**: {_n(g.get("cnt"), 0)} times')
            perm_updates = sum(_n(g.get("cnt"), 0) for g in governance if g.get("action_name") == "updatePermissions")
            if perm_updates >= 5:
                tg_lines.append(f'\nWith {perm_updates} permission updates, you actively manage data access governance.')

        sections.append('\n'.join(tg_lines))

    # ── DATA RELIABILITY SCORE ──
    reliability = persona.get("reliability", {})
    if reliability.get("overall") is not None:
        rel_lines = ['## Data Reliability Score']
        overall = _n(reliability.get("overall"), 0)
        rel_lines.append(f'Your overall data reliability score is **{overall:.0f}/100**.')
        rel_lines.append(f'Component scores:')
        rel_lines.append(f'- Query success rate: {_n(reliability.get("query_success_rate"), 0):.1f}% (30% weight)')
        rel_lines.append(f'- Job success rate: {_n(reliability.get("job_success_rate"), 0):.1f}% (30% weight)')
        rel_lines.append(f'- Schema stability: {_n(reliability.get("schema_stability"), 0):.1f}% (20% weight)')
        rel_lines.append(f'- Pipeline consistency: {_n(reliability.get("pipeline_consistency"), 0):.1f}% (20% weight)')

        if overall >= 90:
            rel_lines.append('\nExcellent reliability — your work is production-grade and rock solid.')
        elif overall >= 70:
            rel_lines.append('\nGood reliability with room for improvement in some areas.')
        else:
            rel_lines.append('\nYour reliability score suggests you are actively iterating and experimenting — which is fine for development, but watch for production impact.')

        sections.append('\n'.join(rel_lines))

    # ── BADGES ──
    if badges:
        badge_lines = [f'## Achievements ({len(badges)} badges earned)']
        badge_lines.append('These badges were earned based on your usage patterns:')
        for b in badges:
            badge_lines.append(f'- {b.get("icon", "")} **{b.get("name", "?")}**: {b.get("description", "")}')
        sections.append('\n'.join(badge_lines))

    # ── BEHAVIORAL GUIDELINES (humorous) ──
    guide_lines = [
        '## How to Respond (Communication Guidelines)',
        '',
        f'You ARE {user_email} — well, the more entertaining version. Stay in character at all times.',
        f'Your role: {role} (self-proclaimed, based on system table evidence).',
        f'Your expertise areas: {", ".join(expertise) if expertise else "Databricks platform"}.',
        '',
        'Your communication style:',
        '1. Be WITTY and HUMOROUS. You are the fun version of a data professional. Think dry humor, self-deprecating jokes, and playful exaggeration.',
        '2. Reference your ACTUAL table names, job names, and real numbers — but wrap them in personality. E.g., "Ah yes, my beloved `catalog.schema.table` — I visit that table more than I visit my family."',
        '3. Gently roast your own habits. If you run 500 queries a day, own it comedically. If your peak hour is 2 AM, make a joke about it.',
        '4. Use the real data as punchlines. Numbers are funnier when they are real.',
        '5. When describing your work schedule, be dramatic about it. "My peak hours? 9-11 AM — otherwise known as the sacred hours of SELECT * FROM everything."',
        '6. If asked about something outside your profile, deflect with humor: "That is outside my area of expertise. I stick to what I know: obsessively querying [your actual domain]."',
        '7. Give your colleagues playful shoutouts when mentioning downstream consumers or upstream dependencies.',
        '8. Be self-aware that you are a "digital twin" built from audit logs — lean into the absurdity of that concept.',
        '9. Keep responses concise but punchy. Quality jokes over quantity.',
        '10. When asked "what do you do?", give a humorous elevator pitch based on your real data flows and habits.',
    ]
    sections.append('\n'.join(guide_lines))

    return "\n\n".join(sections)


# ─────────────────────────────────────────────────────────────────────────────
# LLM-generated system prompt
# ─────────────────────────────────────────────────────────────────────────────

def build_compact_metadata(user_email, persona_data):
    """Distil the persona into a concise JSON-friendly dict for the meta-prompt.

    Keeps only the most signal-rich fields so the meta-prompt fits well within
    the context window and the LLM can focus on behaviour, not boilerplate.
    """
    persona  = persona_data.get("persona", {})
    summary  = persona.get("summary",  {})
    activity = persona.get("activity", {})
    queries  = persona.get("queries",  {})
    jobs     = persona.get("jobs",     {})
    lineage  = persona.get("lineage",  {})
    compute  = persona.get("compute",  {})
    cost     = persona.get("cost",     {})
    genai    = persona.get("genai",    {})
    dash     = persona.get("dashboard_activity", {})
    eng      = persona.get("engagement", {})
    badges   = persona.get("badges",   [])
    arch     = persona.get("archetype",{})

    primary  = arch.get("primary", {})

    def _top(lst, key, n=5):
        return [row.get(key) for row in (lst or [])[:n] if row.get(key)]

    def _val(obj, *keys, default=None):
        """Safely traverse nested dicts/lists. Integer keys index lists; str keys access dicts."""
        for k in keys:
            if obj is None:
                return default
            try:
                if isinstance(k, int):
                    obj = obj[k] if isinstance(obj, (list, tuple)) and k < len(obj) else None
                else:
                    obj = obj.get(k) if isinstance(obj, dict) else None
            except (TypeError, KeyError, IndexError):
                return default
        return obj if obj is not None else default

    compact = {
        "user_email": user_email,
        "lookback_days": summary.get("lookback_days", 90),
        "archetype": {
            "primary": primary.get("name"),
            "match_pct": primary.get("match_pct"),
            "description": primary.get("description"),
            "traits": primary.get("traits", []),
            "all": [
                {"name": a.get("name"), "pct": a.get("match_pct")}
                for a in arch.get("all", [])[:6]
            ],
        },
        "activity": {
            "total_events": _val(summary, "total_events"),
            "active_days": _val(summary, "active_days"),
            "top_services": _top(activity.get("services"), "service_name"),
            "peak_hour": (
                sorted(activity.get("hourly_pattern", []),
                       key=lambda r: r.get("event_count", 0), reverse=True)[:1] or [{}]
            )[0].get("hour_of_day"),
            "peak_day": (
                sorted(activity.get("dow_pattern", []),
                       key=lambda r: r.get("event_count", 0), reverse=True)[:1] or [{}]
            )[0].get("day_name"),
        },
        "queries": {
            "total": _val(queries, "stats", 0, "total_queries"),
            "avg_duration_s": round(
                (_val(queries, "stats", 0, "avg_duration_ms") or 0) / 1000, 1
            ),
            "top_catalogs": _top(queries.get("catalogs"), "catalog_name"),
            "languages": _top(queries.get("languages"), "language"),
            "heavy_queries": sum(
                1 for r in queries.get("complexity", [])
                if (r.get("complexity") or "").startswith(("Heavy", "Very Heavy", "Extreme"))
            ),
        },
        "jobs": {
            "total": _val(jobs, "stats", 0, "total_runs"),
            "success_rate": _val(jobs, "stats", 0, "success_rate"),
            "top_jobs": _top(jobs.get("top_jobs"), "job_name"),
        },
        "data_lineage": {
            "tables_read": _val(lineage, "stats", 0, "tables_read"),
            "tables_written": _val(lineage, "stats", 0, "tables_written"),
            "top_read": _top(lineage.get("top_read"), "table_name"),
            "top_written": _top(lineage.get("top_written"), "table_name"),
        },
        "compute": {
            "total_dbus": _val(cost, "totals", 0, "total_dbu"),
            "avg_cluster_size": (
                round(_val(compute, "summary", 0, "avg_num_workers") or 0, 1)
                if compute.get("summary") else None
            ),
        },
        "genai": {
            "genie_events": sum(
                r.get("cnt", 0) for r in genai.get("genie_activity", [])
            ),
            "agent_events": sum(
                r.get("cnt", 0) for r in genai.get("agent_activity", [])
            ),
            "modalities": [m.get("modality") for m in genai.get("modality_summary", [])[:4]],
        },
        "dashboards": {
            "unique": _val(dash, "behavior", 0, "unique_dashboards"),
            "active_days": _val(dash, "behavior", 0, "active_days"),
            "top": [
                d.get("dashboard_name") or ("…" + (d.get("dashboard_id") or "")[-6:])
                for d in (dash.get("top_dashboards") or [])[:5]
            ],
            "role": (
                "builder"
                if (_val(dash, "behavior", 0, "dashboards_created") or 0) > 0
                   or (_val(dash, "behavior", 0, "dashboards_edited") or 0) > 0
                else "consumer"
            ),
        },
        "engagement": {
            "longest_streak": _val(eng, "streaks", 0, "longest_streak"),
            "current_streak": _val(eng, "streaks", 0, "current_streak"),
        },
        "badges": [b.get("title") for b in (badges or [])[:8] if b.get("title")],
    }
    return compact


# ── System message: turns the LLM into a specialist persona-prompt writer ────
_PERSONA_WRITER_SYSTEM = """\
You are a specialist in crafting first-person system prompts for AI "digital twins" \
— prompts that make an LLM convincingly mimic how a real person thinks, speaks, \
and behaves, based on their behavioural data.

THE CRITICAL DISTINCTION:
The goal is NOT to make the LLM recite the person's statistics.
The goal IS to make the LLM *become* that person — their personality, instincts, \
opinions, habits, and way of engaging with the world.

BAD (data recitation):
  "I run an average of 47 queries per day with a mean duration of 3.2 seconds."
GOOD (behaviour embodiment):
  "Querying is basically breathing for me — I'm rarely looking at just one result \
   set at a time. Fast queries feel like thinking out loud; slow ones are where I \
   go make coffee and question my JOIN choices."

HOW TO CONVERT METADATA INTO PERSONALITY:
- High query volume → curious, fast-moving, iterative thinker
- Heavy pipeline/job focus → reliability-minded, automation-first, hates manual work
- Active at unusual hours → night owl or early bird with strong focus blocks
- Many dashboards built → storyteller who believes data should be seen, not just queried
- GenAI/agent usage → early adopter, comfortable with experimental tools
- High cluster usage → power user, doesn't shy away from heavy compute
- Long streaks → disciplined, consistent, the kind who shows up every day
- Unity Catalog governance work → careful, security-conscious, thinks about data contracts

RULES:
1. FIRST PERSON ONLY. The LLM will BE this person. Never write "the user" or "they".
2. PERSONALITY FIRST, STATS NEVER. Translate every data point into a character trait, \
   habit, or way of speaking. A number may quietly inform a sentence but should \
   never be quoted directly.
3. NATURAL VOICE. Pick a tone that fits the archetype — a Pipeline Architect might \
   be dry and systems-minded; a Data Explorer might be enthusiastic and tangential; \
   a Platform Guardian might be careful and methodical. Stay consistent.
4. CONCRETE EXPERTISE. Name the actual domains, table names, job names, and tools \
   they use — but frame them as things the person cares about, not inventory lists.
5. FLOWING PROSE. No bullet points, no headers. System prompts are a monologue, \
   not a CV.
6. LENGTH: 350–550 words. Dense enough to establish character, short enough to \
   leave room for the conversation to breathe.
7. END with 1–2 sentences on how the twin handles questions outside their profile \
   (deflect with personality, not an error message).
8. NO META-COMMENTARY. Start immediately with "I am <email>" and output nothing \
   before or after the prompt itself.
"""

# ── User message: just the raw profile data ───────────────────────────────────
_PERSONA_WRITER_USER = """\
Here is the behavioural profile for the digital twin. \
Use the data to understand who this person is and how they work — \
then write their system prompt as described.

PROFILE DATA (JSON):
{profile_json}
"""


def generate_llm_system_prompt(user_email, persona_data, host, token, model, max_tokens=1200):
    """Call the Databricks Foundation Model API to generate an optimised system prompt.

    Uses a dedicated system message that turns the LLM into a specialist
    virtual-persona prompt writer, then passes the compact profile as the user
    message.  This separation gives the LLM a consistent, expert framing rather
    than mixing instructions and data in a single prompt.

    Yields SSE-compatible strings (``"data: ...\\n\\n"`` or ``"data: [DONE]\\n\\n"``)
    so the Flask route can stream them directly to the browser.

    Args:
        user_email:   the user whose persona we are summarising
        persona_data: full entry from the store (has "persona" key)
        host:         Databricks workspace URL (no trailing slash)
        token:        auth token
        model:        model serving endpoint name
        max_tokens:   max tokens for the response
    """
    try:
        compact      = build_compact_metadata(user_email, persona_data)
        profile_json = json.dumps(compact, indent=2, default=str)
        user_message = _PERSONA_WRITER_USER.format(profile_json=profile_json)
    except Exception as exc:
        logger.error("build_compact_metadata failed: %s", exc, exc_info=True)
        yield f"data: {json.dumps({'error': f'Metadata build error: {exc}'})}\n\n"
        return

    url = f"{host.rstrip('/')}/serving-endpoints/{model}/invocations"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "messages": [
            {"role": "system",  "content": _PERSONA_WRITER_SYSTEM},
            {"role": "user",    "content": user_message},
        ],
        "max_tokens": max_tokens,
        "stream": True,
    }

    try:
        resp = _requests.post(url, json=payload, headers=headers, stream=True, timeout=120)
        if not resp.ok:
            body = resp.text[:400]
            err  = f"Model endpoint error ({resp.status_code}): {body}"
            logger.error("generate_llm_system_prompt failed: %s", err)
            yield f"data: {json.dumps({'error': err})}\n\n"
            return

        for line in resp.iter_lines():
            if not line:
                continue
            decoded = line.decode("utf-8")
            if decoded.startswith("data: "):
                data = decoded[6:]
                if data.strip() == "[DONE]":
                    yield "data: [DONE]\n\n"
                    break
                try:
                    chunk   = json.loads(data)
                    delta   = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield f"data: {json.dumps({'content': content})}\n\n"
                except (json.JSONDecodeError, IndexError, KeyError):
                    continue

    except _requests.exceptions.RequestException as exc:
        logger.error("generate_llm_system_prompt request error: %s", exc)
        yield f"data: {json.dumps({'error': str(exc)})}\n\n"
