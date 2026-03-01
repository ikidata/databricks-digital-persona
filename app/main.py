"""Main Flask application for Digital Persona Creator."""

import json
import logging
import os
import secrets
import requests
from flask import Flask, Response, render_template, request, jsonify
from app.store import store
from app.config import (
    get_system_tables,
    get_databricks_host,
    get_warehouse_http_path,
    get_chat_model_endpoint,
    get_chat_max_tokens,
)
from app.db import check_table_access, check_all_tables_access
from app.analyzer import run_analysis
from app.prompt_generator import generate_system_prompt, generate_llm_system_prompt

# Enable logging so analysis debug messages are visible
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

app = Flask(__name__)

# Flask secret key — used for session signing and CSRF tokens.
# In production (Databricks Apps) set FLASK_SECRET_KEY env var.
# Falls back to a random key per-process (sessions won't survive restarts).
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32)

# ── Startup diagnostics ─────────────────────────────────────────────────────
_startup_logger = logging.getLogger("app.startup")
_startup_logger.info("DATABRICKS_HOST env = %r", os.environ.get("DATABRICKS_HOST"))
_startup_logger.info("get_databricks_host() = %r", get_databricks_host())
_startup_logger.info("WAREHOUSE_ID env = %r", os.environ.get("WAREHOUSE_ID"))
_startup_logger.info("resolved http_path = %r", get_warehouse_http_path())
_startup_logger.info("MODEL_SERVING_ENDPOINT env = %r", os.environ.get("MODEL_SERVING_ENDPOINT"))


@app.after_request
def _add_security_headers(response):
    """Inject standard security headers on every response."""
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    # Permissive CSP — tightened here to allow inline scripts/styles (required by the
    # persona page) while blocking external script loading.
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
        "connect-src 'self'; frame-ancestors 'self';"
    )
    return response


def _get_current_user():
    """Get the current user's email.

    When LOCAL=true (set by run_locally.py), returns the email passed via
    the --email CLI argument (stored in LOCAL_USER_EMAIL env var).
    Otherwise, reads X-Forwarded-Email header injected by Databricks Apps reverse proxy.
    """
    if os.environ.get("LOCAL", "false").lower() == "true":
        return os.environ.get("LOCAL_USER_EMAIL", "dev-user@example.com")
    # In Databricks Apps, the reverse proxy injects X-Forwarded-Email
    return request.headers.get("X-Forwarded-Email", "unknown@databricks.user")


def _get_current_username():
    """Get the display name from headers."""
    return request.headers.get("X-Forwarded-Preferred-Username", _get_current_user())


# ─── Debug ───────────────────────────────────────────────────────────────────

@app.route("/api/debug-host")
def debug_host():
    """Temporary endpoint to diagnose serving-endpoint 404s."""
    host = get_databricks_host()
    model = get_chat_model_endpoint()
    return jsonify({
        "DATABRICKS_HOST_env": os.environ.get("DATABRICKS_HOST"),
        "get_databricks_host": host,
        "MODEL_SERVING_ENDPOINT_env": os.environ.get("MODEL_SERVING_ENDPOINT"),
        "get_chat_model_endpoint": model,
        "constructed_url": f"{host}/serving-endpoints/{model}/invocations",
        "X-Forwarded-Host": request.headers.get("X-Forwarded-Host"),
        "X-Forwarded-Access-Token_present": bool(request.headers.get("X-Forwarded-Access-Token")),
    })


# ─── Pages ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Landing page with navigation."""
    user_email = _get_current_user()
    username = _get_current_username()
    has_data = store.has_persona_data(user_email)
    access_results = store.get_access_results(user_email)
    return render_template("index.html",
                           user_email=user_email,
                           username=username,
                           has_data=has_data,
                           has_access_check=access_results is not None)


@app.route("/check-access")
def check_access_page():
    """Check Access page - validates system table permissions."""
    user_email = _get_current_user()
    username = _get_current_username()
    access_results = store.get_access_results(user_email)
    return render_template("check_access.html",
                           user_email=user_email,
                           username=username,
                           access_results=access_results)


@app.route("/analyze")
def analyze_page():
    """Run Digital Persona Analysis page."""
    user_email = _get_current_user()
    username = _get_current_username()
    status = store.get_analysis_status(user_email)
    has_data = store.has_persona_data(user_email)
    return render_template("analyze.html",
                           user_email=user_email,
                           username=username,
                           status=status,
                           has_data=has_data)


@app.route("/persona")
def persona_page():
    """Visualization page showing persona data."""
    user_email = _get_current_user()
    username = _get_current_username()
    persona_entry = store.get_persona_data(user_email)
    persona = persona_entry["persona"] if persona_entry else None
    return render_template("persona.html",
                           user_email=user_email,
                           username=username,
                           persona=persona,
                           persona_json=json.dumps(persona, default=str) if persona else "null")


@app.route("/digital-version")
def digital_version_page():
    """Digital Version page - system prompt generator."""
    user_email = _get_current_user()
    username = _get_current_username()
    persona_entry = store.get_persona_data(user_email)
    prompt = ""
    persona = None
    if persona_entry:
        try:
            prompt = generate_system_prompt(user_email, persona_entry)
        except Exception as e:
            app.logger.error("generate_system_prompt failed: %s", e)
            prompt = "Error generating system prompt. Please re-run the analysis."
        persona = persona_entry.get("persona")
    else:
        prompt = (
            "You are a helpful Databricks assistant. "
            "Run the persona analysis to get a personalised system prompt "
            "based on your activity data."
        )
    model_name = get_chat_model_endpoint()
    return render_template("digital_version.html",
                           user_email=user_email,
                           username=username,
                           system_prompt=prompt,
                           persona=persona,
                           has_data=persona_entry is not None,
                           model_name=model_name)


# ─── API Endpoints ──────────────────────────────────────────────────────────

@app.route("/api/check-access", methods=["POST"])
def api_check_access():
    """API: Run access check against all system tables.

    Uses a single SQL connection for all checks to avoid repeated
    handshake overhead that would cause a proxy 504 timeout.
    """
    user_email = _get_current_user()
    user_token = request.headers.get("X-Forwarded-Access-Token")
    app.logger.info(
        "check-access: user=%s has_obo_token=%s", user_email, bool(user_token)
    )
    tables = get_system_tables()

    try:
        results = check_all_tables_access(tables, user_token=user_token)
    except Exception as exc:
        app.logger.error("check-access failed: %s", exc, exc_info=True)
        return jsonify({"status": "error", "error": str(exc)}), 500

    store.set_access_results(user_email, results)
    return jsonify({"status": "ok", "results": results})


@app.route("/api/run-analysis", methods=["POST"])
def api_run_analysis():
    """API: Trigger persona analysis (runs in background thread)."""
    user_email = _get_current_user()
    status = store.get_analysis_status(user_email)

    if status.get("status") == "running":
        return jsonify({"status": "already_running", "message": "Analysis is already running."})

    # Get user access token if available (on-behalf-of-user)
    user_token = request.headers.get("X-Forwarded-Access-Token")
    run_analysis(user_email, user_token=user_token)

    return jsonify({"status": "started", "message": "Analysis started."})


@app.route("/api/analysis-status")
def api_analysis_status():
    """API: Get current analysis status."""
    user_email = _get_current_user()
    status = store.get_analysis_status(user_email)
    has_data = store.has_persona_data(user_email)
    return jsonify({**status, "has_data": has_data})



@app.route("/api/persona-data")
def api_persona_data():
    """API: Get the full persona data."""
    user_email = _get_current_user()
    entry = store.get_persona_data(user_email)
    if not entry:
        return jsonify({"status": "no_data"})
    return jsonify({"status": "ok", "data": entry["persona"], "timestamp": entry["timestamp"]})


@app.route("/api/system-prompt")
def api_system_prompt():
    """API: Get the generated system prompt."""
    user_email = _get_current_user()
    entry = store.get_persona_data(user_email)
    if not entry:
        return jsonify({"status": "no_data", "prompt": ""})
    prompt = generate_system_prompt(user_email, entry)
    return jsonify({"status": "ok", "prompt": prompt})


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """API: Chat with the digital twin via Databricks Foundation Model API.

    Streams responses token-by-token using server-sent events.
    The persona system prompt is automatically injected.
    """
    user_email = _get_current_user()
    entry = store.get_persona_data(user_email)

    body = request.get_json()
    messages = body.get("messages", [])
    if not messages:
        return jsonify({"error": "No messages provided."}), 400

    if entry:
        # Prefer the LLM-generated behavioural prompt (produced by the persona-writer
        # LLM on the Digital Version page) over the static template prompt.  The LLM
        # version captures personality and habits rather than listing raw KPI data.
        llm_prompt = store.get_llm_prompt(user_email)
        system_prompt = llm_prompt if llm_prompt else generate_system_prompt(user_email, entry)
    else:
        # No persona yet — use a minimal prompt so the user can test chat connectivity
        system_prompt = (
            "You are a helpful Databricks assistant. "
            "Run the persona analysis to get a personalised system prompt based on your activity data."
        )

    # Databricks Foundation Model API endpoint
    host = get_databricks_host()
    model = get_chat_model_endpoint()
    max_tokens = get_chat_max_tokens()
    url = f"{host}/serving-endpoints/{model}/invocations"
    app.logger.info("chat: host=%s, url=%s", host, url)

    # Auth: prefer the current user's OBO token (X-Forwarded-Access-Token injected
    # by Databricks Apps reverse proxy) so the request runs under user permissions.
    # Fall back to app service principal / PAT if OBO token is unavailable.
    obo_token = request.headers.get("X-Forwarded-Access-Token")
    if obo_token:
        token = obo_token
    else:
        from app.db import get_auth_token
        token = get_auth_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            *messages,
        ],
        "max_tokens": max_tokens,
        "stream": True,
    }

    def stream():
        try:
            resp = requests.post(url, json=payload, headers=headers, stream=True, timeout=60)
            if not resp.ok:
                # Read the body to get an error message (may be HTML or JSON)
                body = resp.text[:500]
                if resp.headers.get("content-type", "").startswith("text/html"):
                    err_msg = (
                        f"Model endpoint returned HTTP {resp.status_code}. "
                        f'Check that endpoint "{model}" exists and your credentials have access.'
                    )
                else:
                    err_msg = f"Model endpoint error ({resp.status_code}): {body}"
                yield f"data: {json.dumps({'error': err_msg})}\n\n"
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
                        chunk = json.loads(data)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield f"data: {json.dumps({'content': content})}\n\n"
                    except (json.JSONDecodeError, IndexError, KeyError):
                        continue
        except requests.exceptions.RequestException as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ─── LLM-generated system prompt ────────────────────────────────────────────

@app.route("/api/regenerate-system-prompt")
def api_regenerate_system_prompt():
    """Stream an LLM-generated optimised system prompt as server-sent events.

    The LLM (databricks-claude-sonnet) receives a compact behaviour profile
    and writes a vivid first-person system prompt, which is streamed back
    token-by-token so the UI can show it appearing live in the textarea.

    The accumulated prompt is also stored so it can be used for chat.
    """
    user_email = _get_current_user()
    entry = store.get_persona_data(user_email)
    if not entry:
        return jsonify({"error": "No persona data. Run analysis first."}), 400

    host  = get_databricks_host()
    model = get_chat_model_endpoint()
    max_tokens = max(get_chat_max_tokens() * 2, 1800)  # 400-650 words needs ~1000-1400 tokens
    app.logger.info("regenerate-system-prompt: host=%s", host)

    obo_token = request.headers.get("X-Forwarded-Access-Token")
    if obo_token:
        token = obo_token
    else:
        from app.db import get_auth_token
        token = get_auth_token()

    def stream():
        accumulated = []
        for chunk in generate_llm_system_prompt(
            user_email, entry, host, token, model, max_tokens=max_tokens
        ):
            yield chunk
            # Extract content tokens to accumulate the full prompt
            if chunk.startswith("data: ") and "[DONE]" not in chunk:
                try:
                    parsed = json.loads(chunk[6:].strip())
                    if parsed.get("content"):
                        accumulated.append(parsed["content"])
                except (json.JSONDecodeError, KeyError):
                    pass
        # Store the generated prompt for use in chat
        full_prompt = "".join(accumulated)
        if full_prompt.strip():
            store.set_llm_prompt(user_email, full_prompt)

    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ─── Export ─────────────────────────────────────────────────────────────────

@app.route("/api/export-persona")
def api_export_persona():
    """Export the persona as a fully self-contained standalone HTML file.

    All CSS, JS and persona data are embedded inline. Avatar PNG images are
    base64-encoded so the file works offline without any server.
    """
    import base64
    import re
    from datetime import datetime

    user_email = _get_current_user()
    persona_entry = store.get_persona_data(user_email)
    if not persona_entry:
        return jsonify({"status": "no_data",
                        "message": "No persona data available. Run analysis first."}), 404

    persona = persona_entry["persona"]
    persona_json = json.dumps(persona, default=str)
    username = _get_current_username()

    # Render the full persona template to a string
    html = render_template(
        "persona.html",
        user_email=user_email,
        username=username,
        persona=persona,
        persona_json=persona_json,
    )

    # ── Inline avatar images as base64 data URIs ──────────────────────────────
    static_dir = os.path.join(os.path.dirname(__file__), "static")

    def _inline_static_src(match):
        path = match.group(1)           # e.g. /static/avatars/data-explorer.png
        rel = path.lstrip("/")[len("static/"):]   # avatars/data-explorer.png
        abs_path = os.path.join(static_dir, rel.replace("/", os.sep))
        try:
            with open(abs_path, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode()
            ext = rel.rsplit(".", 1)[-1].lower()
            mime = {"png": "image/png", "jpg": "image/jpeg",
                    "svg": "image/svg+xml", "gif": "image/gif"}.get(ext, "image/png")
            return f'src="data:{mime};base64,{b64}"'
        except (IOError, OSError):
            return match.group(0)

    html = re.sub(r'src="(/static/[^"]+)"', _inline_static_src, html)

    # ── Remove live-app nav links that won't work offline ─────────────────────
    # Replace each <a href="/..."> that points to an internal route with <span>
    html = re.sub(
        r'<a href="/(?:analyze|check-access|digital-version)"[^>]*>([^<]*)</a>',
        r'<span style="color:var(--text-muted);padding:8px 14px;">\1</span>',
        html,
    )

    # ── Inject export header banner ────────────────────────────────────────────
    ts_generated = str(persona_entry.get("timestamp", ""))[:19]
    ts_exported  = datetime.now().strftime("%Y-%m-%d %H:%M")
    banner = (
        f'<div style="background:#1a1f26;border-bottom:2px solid #30363D;'
        f'padding:10px 24px;font-size:12px;color:#8B949E;'
        f'display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">'
        f'<span>📄 <strong style="color:#E6EDF3">Databricks Digital Persona Export</strong>'
        f' &nbsp;·&nbsp; {user_email}</span>'
        f'<span>Analysis: {ts_generated} &nbsp;·&nbsp; Exported: {ts_exported}</span>'
        f'</div>'
    )
    html = html.replace(
        '<main class="main-content">',
        banner + '<main class="main-content" style="animation:none">',
    )

    # ── Disable animations so static snapshot looks clean ─────────────────────
    html = html.replace(
        "animation: fadeInUp 0.4s ease;",
        "/* animation disabled in export */",
    )

    safe_email = re.sub(r"[^\w.-]", "_", user_email)
    filename = f"digital-persona-{safe_email}-{ts_exported[:10]}.html"
    return Response(
        html,
        mimetype="text/html; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ─── Run ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("DATABRICKS_APP_PORT", 8000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
