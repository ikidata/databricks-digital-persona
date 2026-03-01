"""Security tests for the Digital Persona application.

Validates that the codebase follows security best practices:
- No hardcoded secrets in source files
- Security headers on all responses
- config.yaml excluded from version control
- No XSS via |safe on user-supplied content
- Token handling safety (not leaked in responses)
- Input validation on API endpoints
- SQL injection prevention
- Debug mode not hardcoded

Run with: python -m pytest tests/test_security.py -v
"""

import os
import re
import pathlib

ROOT = pathlib.Path(__file__).parent.parent


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _read(p: pathlib.Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _all_source_files():
    """Yield all .py and .html files (excluding __pycache__, .venv, tests)."""
    for p in ROOT.rglob("*"):
        if p.is_dir():
            continue
        if "__pycache__" in str(p) or ".venv" in str(p):
            continue
        if p.suffix in {".py", ".html"}:
            yield p


# ═══════════════════════════════════════════════════════════════════════════
#  Test: No hardcoded secrets in source code
# ═══════════════════════════════════════════════════════════════════════════

SECRET_PATTERNS = [
    (r"dapi[0-9a-f]{32,}", "Databricks PAT token"),
    (r"(?i)client[_-]?secret\s*=\s*['\"]([^'\"]{8,})['\"]", "Hardcoded client_secret"),
    (r"(?i)password\s*=\s*['\"]([^'\"]{4,})['\"]", "Hardcoded password"),
    (r"(?i)api[_-]?key\s*=\s*['\"]([^'\"]{8,})['\"]", "Hardcoded API key"),
]

ALLOWED_SECRET_FILES = {"config.yaml.example", "security_scan.py", "test_security.py"}


class TestNoHardcodedSecrets:
    """Ensure no real credentials are present in tracked source files."""

    def test_no_secrets_in_python_or_html(self):
        violations = []
        for p in _all_source_files():
            if p.name in ALLOWED_SECRET_FILES:
                continue
            content = _read(p)
            for pattern, label in SECRET_PATTERNS:
                for m in re.finditer(pattern, content):
                    line_no = content[:m.start()].count("\n") + 1
                    lines = content.splitlines()
                    src_line = lines[line_no - 1].strip() if line_no <= len(lines) else ""
                    # Skip comments
                    if src_line.startswith("#") or src_line.startswith("//"):
                        continue
                    violations.append(f"{p.relative_to(ROOT)}:{line_no} — {label}")
        assert not violations, f"Hardcoded secrets found:\n" + "\n".join(violations)

    def test_no_secrets_in_yaml(self):
        """Ensure config.yaml.example doesn't contain real tokens."""
        example = ROOT / "config.yaml.example"
        if not example.exists():
            return
        content = _read(example)
        assert not re.search(r"dapi[0-9a-f]{32,}", content), \
            "config.yaml.example contains a real Databricks PAT"
        assert not re.search(r"dose[0-9a-f]{20,}", content), \
            "config.yaml.example contains a real OAuth secret"


# ═══════════════════════════════════════════════════════════════════════════
#  Test: config.yaml is git-ignored
# ═══════════════════════════════════════════════════════════════════════════

class TestConfigSafety:
    """Ensure secrets config is excluded from version control."""

    def test_config_yaml_in_gitignore(self):
        gitignore = ROOT / ".gitignore"
        assert gitignore.exists(), ".gitignore file is missing"
        content = _read(gitignore)
        assert "config.yaml" in content, "config.yaml is not in .gitignore"

    def test_env_files_in_gitignore(self):
        gitignore = ROOT / ".gitignore"
        content = _read(gitignore)
        assert ".env" in content, ".env is not in .gitignore"

    def test_investigate_script_in_gitignore(self):
        gitignore = ROOT / ".gitignore"
        content = _read(gitignore)
        assert "investigate_databricks.py" in content, \
            "investigate_databricks.py should be git-ignored"


# ═══════════════════════════════════════════════════════════════════════════
#  Test: Security headers in Flask app
# ═══════════════════════════════════════════════════════════════════════════

class TestSecurityHeaders:
    """Verify that security headers are set on responses."""

    def _get_main_content(self):
        return _read(ROOT / "app" / "main.py")

    def test_x_content_type_options(self):
        content = self._get_main_content()
        assert "X-Content-Type-Options" in content, \
            "X-Content-Type-Options header not set"

    def test_x_frame_options(self):
        content = self._get_main_content()
        assert "X-Frame-Options" in content, \
            "X-Frame-Options header not set"

    def test_content_security_policy(self):
        content = self._get_main_content()
        assert "Content-Security-Policy" in content, \
            "Content-Security-Policy header not set"

    def test_referrer_policy(self):
        content = self._get_main_content()
        assert "Referrer-Policy" in content, \
            "Referrer-Policy header not set"

    def test_no_wildcard_cors(self):
        """Ensure no Access-Control-Allow-Origin: * is set."""
        content = self._get_main_content()
        assert "Access-Control-Allow-Origin" not in content or \
               '"*"' not in content, \
            "Wildcard CORS (*) detected — restrict to specific origins"


# ═══════════════════════════════════════════════════════════════════════════
#  Test: No XSS via |safe on user content
# ═══════════════════════════════════════════════════════════════════════════

class TestXSSPrevention:
    """Verify Jinja2 templates don't use |safe on user-supplied data."""

    def test_no_safe_on_user_input(self):
        """Flag |safe usage on variables that could contain user-typed text."""
        violations = []
        user_vars = {"query_text", "error_message", "user_input", "statement_text",
                     "user_email", "username"}
        for p in ROOT.rglob("*.html"):
            for no, line in enumerate(_read(p).splitlines(), 1):
                if "|safe" not in line:
                    continue
                for var in user_vars:
                    if var in line and "|safe" in line:
                        violations.append(f"{p.relative_to(ROOT)}:{no}")
        assert not violations, \
            f"|safe used on user-sourced variables:\n" + "\n".join(violations)

    def test_persona_json_safe_is_intentional(self):
        """persona_json and system_prompt are server-generated, so |safe is OK.
        Just verify they actually exist (the templates depend on them)."""
        digital = _read(ROOT / "app" / "templates" / "digital_version.html")
        assert "system_prompt" in digital or "tojson" in digital


# ═══════════════════════════════════════════════════════════════════════════
#  Test: Debug mode safety
# ═══════════════════════════════════════════════════════════════════════════

class TestDebugMode:
    """Ensure debug mode isn't hardcoded to True."""

    def test_no_hardcoded_debug_true(self):
        content = _read(ROOT / "app" / "main.py")
        # Match: debug=True that isn't controlled by env var
        lines = content.splitlines()
        for no, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            # Flag: app.run(debug=True) without env var check
            if re.search(r'\.run\([^)]*debug\s*=\s*True', stripped):
                assert False, f"main.py:{no} — debug=True hardcoded in app.run()"

    def test_debug_from_env_var(self):
        content = _read(ROOT / "app" / "main.py")
        assert "FLASK_DEBUG" in content, \
            "Debug should be controlled via FLASK_DEBUG env var"


# ═══════════════════════════════════════════════════════════════════════════
#  Test: Token handling
# ═══════════════════════════════════════════════════════════════════════════

class TestTokenHandling:
    """Verify tokens are handled securely."""

    def test_obo_token_from_header(self):
        """OBO token should come from X-Forwarded-Access-Token header."""
        content = _read(ROOT / "app" / "main.py")
        assert "X-Forwarded-Access-Token" in content

    def test_token_not_in_json_response(self):
        """Tokens should never be returned in API JSON responses."""
        content = _read(ROOT / "app" / "main.py")
        # Look for jsonify calls that include token variables
        violations = []
        for no, line in enumerate(content.splitlines(), 1):
            if "jsonify" in line and "token" in line.lower():
                # Allow: checking token_present (bool), not the actual value
                if "present" in line.lower() or "bool(" in line:
                    continue
                violations.append(f"main.py:{no}: {line.strip()[:80]}")
        assert not violations, \
            f"Token values may be returned in JSON:\n" + "\n".join(violations)

    def test_secret_key_not_hardcoded(self):
        """Flask secret_key should use env var or random generation."""
        content = _read(ROOT / "app" / "main.py")
        assert "secrets.token_hex" in content or "FLASK_SECRET_KEY" in content


# ═══════════════════════════════════════════════════════════════════════════
#  Test: SQL injection prevention
# ═══════════════════════════════════════════════════════════════════════════

class TestSQLInjection:
    """Check for SQL injection patterns in the analyzer."""

    def test_no_request_data_in_sql(self):
        """SQL queries should never directly interpolate request.form/args/json."""
        content = _read(ROOT / "app" / "analyzer.py")
        dangerous = re.findall(
            r'execute_query\(f""".*?request\.(form|args|json|data)',
            content, re.DOTALL
        )
        assert not dangerous, \
            "SQL queries interpolate request data directly — use parameterised queries"

    def test_user_email_is_identity_header(self):
        """user_email comes from identity header, not user input."""
        content = _read(ROOT / "app" / "main.py")
        # Verify _get_current_user reads from trusted header or env var
        assert "X-Forwarded-Email" in content
        assert "LOCAL_USER_EMAIL" in content


# ═══════════════════════════════════════════════════════════════════════════
#  Test: Sensitive data not logged
# ═══════════════════════════════════════════════════════════════════════════

class TestLogSafety:
    """Ensure tokens and secrets aren't logged."""

    def test_no_token_values_logged(self):
        violations = []
        for p in (ROOT / "app").rglob("*.py"):
            for no, line in enumerate(_read(p).splitlines(), 1):
                low = line.lower().strip()
                if low.startswith("#"):
                    continue
                # Flag: logging actual token values (not just their presence)
                if re.search(r'(log|print)\s*\(.*?(user_token|obo_token|bearer)', low):
                    # Allow: logging that token is present (bool check)
                    if "present" in low or "bool(" in low or "none" in low:
                        continue
                    violations.append(f"{p.relative_to(ROOT)}:{no}")
        assert not violations, \
            f"Token values may be logged:\n" + "\n".join(violations)


# ═══════════════════════════════════════════════════════════════════════════
#  Run tests directly with: python tests/test_security.py
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    passed = 0
    failed = 0

    for cls_name in [TestNoHardcodedSecrets, TestConfigSafety, TestSecurityHeaders,
                     TestXSSPrevention, TestDebugMode, TestTokenHandling,
                     TestSQLInjection, TestLogSafety]:
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
