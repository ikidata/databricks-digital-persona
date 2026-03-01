"""Tests for Flask routes and API endpoints.

Uses Flask's test client — no Databricks connection needed.
Validates page rendering, API behaviour, digital version gate,
and input validation.

Run with: python -m pytest tests/test_routes.py -v
"""

import json
import os
import sys
import types
from unittest.mock import MagicMock

# Ensure we can import the app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set LOCAL mode and provide env-var overrides so the app doesn't need
# config.yaml or a real Databricks connection during tests.
os.environ["LOCAL"] = "true"
os.environ["LOCAL_USER_EMAIL"] = "test-user@example.com"
os.environ.setdefault("DATABRICKS_HOST", "https://test-workspace.azuredatabricks.net")
os.environ.setdefault("MODEL_SERVING_ENDPOINT", "test-model-endpoint")
os.environ.setdefault("DATABRICKS_WAREHOUSE_HTTP_PATH", "/sql/1.0/warehouses/test")

# Mock the databricks SDK modules so tests run without the SDK installed
_db_mock = types.ModuleType("databricks")
_db_sql_mock = types.ModuleType("databricks.sql")
_db_sdk_mock = types.ModuleType("databricks.sdk")
_db_sdk_core_mock = types.ModuleType("databricks.sdk.core")
_db_sdk_core_mock.Config = MagicMock()
_db_sdk_core_mock.oauth_service_principal = MagicMock()
_db_sql_mock.connect = MagicMock()
_db_mock.sql = _db_sql_mock
_db_mock.sdk = _db_sdk_mock
_db_sdk_mock.core = _db_sdk_core_mock
for mod_name, mod in [
    ("databricks", _db_mock),
    ("databricks.sql", _db_sql_mock),
    ("databricks.sdk", _db_sdk_mock),
    ("databricks.sdk.core", _db_sdk_core_mock),
]:
    sys.modules.setdefault(mod_name, mod)

from app.main import app
from app.store import store


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _client():
    app.config["TESTING"] = True
    return app.test_client()


def _inject_persona():
    """Inject fake persona data into the store for the test user."""
    store.set_persona_data("test-user@example.com", {
        "archetype": {
            "primary": {
                "name": "The Pipeline Architect",
                "emoji": "🏗️",
                "color": "#58A6FF",
                "gradient": "linear-gradient(135deg, #1a365d, #2563eb)",
                "traits": ["Automation", "Reliability"],
            },
            "all": [
                {"name": "The Pipeline Architect", "emoji": "🏗️", "color": "#58A6FF",
                 "match_pct": 35, "description": "Builds pipelines", "traits": ["Auto"]},
                {"name": "The Data Explorer", "emoji": "🔍", "color": "#F0883E",
                 "match_pct": 25, "description": "Explores data", "traits": ["SQL"]},
                {"name": "The Platform Guardian", "emoji": "🛡️", "color": "#8B949E",
                 "match_pct": 15, "description": "Manages platform", "traits": ["Admin"]},
                {"name": "The Dashboard Crafter", "emoji": "📊", "color": "#D2A8FF",
                 "match_pct": 12, "description": "Makes dashboards", "traits": ["BI"]},
                {"name": "The ML Alchemist", "emoji": "🧪", "color": "#7EE787",
                 "match_pct": 8, "description": "Trains models", "traits": ["ML"]},
                {"name": "The Cost-Conscious Optimizer", "emoji": "💰", "color": "#F9E2AF",
                 "match_pct": 5, "description": "Saves costs", "traits": ["Lean"]},
            ],
        },
        "badges": [{"title": "SQL Centurion", "icon": "🏆", "desc": "100+ queries"}],
        "summary": {"reliability_score": 85},
        "activity": {},
        "queries": {},
        "jobs": {},
    })


def _cleanup():
    store.clear("test-user@example.com")


# ═══════════════════════════════════════════════════════════════════════════
#  Test: Page routes return 200
# ═══════════════════════════════════════════════════════════════════════════

class TestPageRoutes:
    """Verify all page routes return 200 OK."""

    def test_index(self):
        c = _client()
        resp = c.get("/")
        assert resp.status_code == 200
        assert b"Digital Persona" in resp.data or b"digital" in resp.data.lower()

    def test_check_access_page(self):
        c = _client()
        resp = c.get("/check-access")
        assert resp.status_code == 200

    def test_analyze_page(self):
        c = _client()
        resp = c.get("/analyze")
        assert resp.status_code == 200

    def test_persona_page(self):
        c = _client()
        resp = c.get("/persona")
        assert resp.status_code == 200

    def test_digital_version_page(self):
        c = _client()
        resp = c.get("/digital-version")
        assert resp.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════
#  Test: Digital Version page gate
# ═══════════════════════════════════════════════════════════════════════════

class TestDigitalVersionGate:
    """Verify the Digital Version page blocks access without persona data."""

    def test_no_data_shows_blocker(self):
        _cleanup()
        c = _client()
        resp = c.get("/digital-version")
        assert resp.status_code == 200
        assert b"No Persona Data Yet" in resp.data
        assert b"Run Analysis" in resp.data

    def test_no_data_hides_chat(self):
        _cleanup()
        c = _client()
        resp = c.get("/digital-version")
        # The chat input element (id="chatInput") should not be rendered,
        # though JS references to it may still exist in the script block.
        assert b'id="chatInput"' not in resp.data

    def test_with_data_shows_chat(self):
        _inject_persona()
        try:
            c = _client()
            resp = c.get("/digital-version")
            assert resp.status_code == 200
            assert b"chatInput" in resp.data
            assert b"No Persona Data Yet" not in resp.data
        finally:
            _cleanup()

    def test_with_data_shows_archetype(self):
        _inject_persona()
        try:
            c = _client()
            resp = c.get("/digital-version")
            assert b"Pipeline Architect" in resp.data
        finally:
            _cleanup()


# ═══════════════════════════════════════════════════════════════════════════
#  Test: API endpoints
# ═══════════════════════════════════════════════════════════════════════════

class TestAPIEndpoints:
    """Test API endpoint behaviour."""

    def test_analysis_status_default(self):
        _cleanup()
        c = _client()
        resp = c.get("/api/analysis-status")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["status"] == "idle"
        assert data["has_data"] is False

    def test_persona_data_no_data(self):
        _cleanup()
        c = _client()
        resp = c.get("/api/persona-data")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["status"] == "no_data"

    def test_persona_data_with_data(self):
        _inject_persona()
        try:
            c = _client()
            resp = c.get("/api/persona-data")
            data = json.loads(resp.data)
            assert data["status"] == "ok"
            assert "archetype" in data["data"]
        finally:
            _cleanup()

    def test_system_prompt_no_data(self):
        _cleanup()
        c = _client()
        resp = c.get("/api/system-prompt")
        data = json.loads(resp.data)
        assert data["status"] == "no_data"

    def test_chat_requires_post(self):
        c = _client()
        resp = c.get("/api/chat")
        assert resp.status_code == 405  # Method Not Allowed

    def test_chat_requires_messages(self):
        c = _client()
        resp = c.post("/api/chat",
                       data=json.dumps({"messages": []}),
                       content_type="application/json")
        assert resp.status_code == 400
        data = json.loads(resp.data)
        assert "error" in data

    def test_export_no_data_returns_404(self):
        _cleanup()
        c = _client()
        resp = c.get("/api/export-persona")
        assert resp.status_code == 404

    def test_regenerate_prompt_no_data_returns_400(self):
        _cleanup()
        c = _client()
        resp = c.get("/api/regenerate-system-prompt")
        assert resp.status_code == 400

    def test_debug_host_endpoint(self):
        c = _client()
        resp = c.get("/api/debug-host")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "get_databricks_host" in data


# ═══════════════════════════════════════════════════════════════════════════
#  Test: Security headers on responses
# ═══════════════════════════════════════════════════════════════════════════

class TestResponseHeaders:
    """Verify security headers are present on actual responses."""

    def test_x_content_type_options(self):
        c = _client()
        resp = c.get("/")
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"

    def test_x_frame_options(self):
        c = _client()
        resp = c.get("/")
        assert resp.headers.get("X-Frame-Options") == "SAMEORIGIN"

    def test_content_security_policy(self):
        c = _client()
        resp = c.get("/")
        csp = resp.headers.get("Content-Security-Policy", "")
        assert "default-src" in csp

    def test_referrer_policy(self):
        c = _client()
        resp = c.get("/")
        assert resp.headers.get("Referrer-Policy") is not None


# ═══════════════════════════════════════════════════════════════════════════
#  Test: Input validation
# ═══════════════════════════════════════════════════════════════════════════

class TestInputValidation:
    """Test that API endpoints validate input properly."""

    def test_chat_rejects_empty_body(self):
        c = _client()
        resp = c.post("/api/chat",
                       data=json.dumps({}),
                       content_type="application/json")
        assert resp.status_code == 400

    def test_check_access_requires_post(self):
        c = _client()
        resp = c.get("/api/check-access")
        assert resp.status_code == 405

    def test_run_analysis_requires_post(self):
        c = _client()
        resp = c.get("/api/run-analysis")
        assert resp.status_code == 405


# ═══════════════════════════════════════════════════════════════════════════
#  Run tests directly with: python tests/test_routes.py
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    passed = 0
    failed = 0

    for cls_name in [TestPageRoutes, TestDigitalVersionGate, TestAPIEndpoints,
                     TestResponseHeaders, TestInputValidation]:
        print(f"\n{'=' * 60}")
        print(f"  {cls_name.__name__}")
        print(f"{'=' * 60}")
        obj = cls_name()
        for name in sorted(dir(obj)):
            if not name.startswith("test_"):
                continue
            try:
                getattr(obj, name)()
                print(f"  PASS  {name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL  {name}: {e}")
                failed += 1

    print(f"\n{'=' * 60}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")
    sys.exit(1 if failed else 0)
