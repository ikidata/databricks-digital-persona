#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cybersecurity scan for the Digital Persona Flask application.

Checks for:
  1.  Hardcoded secrets / tokens in source files
  2.  SQL injection risks (f-string SQL with raw user input)
  3.  XSS risks (|safe Jinja2 filter on user-supplied data)
  4.  Debug mode enabled in production
  5.  Missing CSRF protection on state-changing POST endpoints
  6.  Missing rate-limiting
  7.  Sensitive data in log statements
  8.  Insecure authentication / token handling
  9.  Exposed sensitive information in error responses
  10. config.yaml checked into version control (secrets leak)
  11. Python dependency vulnerabilities (pip-audit if installed)
  12. Overly broad CORS / header exposure
"""
import io
import os
import re
import sys
import subprocess
import pathlib

# Force UTF-8 output on Windows so box-drawing chars and emoji render correctly
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).parent

SEVERITY = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}
COLOR = {
    "CRITICAL": "\033[91m",   # bright red
    "HIGH":     "\033[31m",   # red
    "MEDIUM":   "\033[33m",   # yellow
    "LOW":      "\033[34m",   # blue
    "INFO":     "\033[36m",   # cyan
    "PASS":     "\033[32m",   # green
    "RESET":    "\033[0m",
}

findings: list[dict] = []


def find(severity, title, detail, file_hint=""):
    findings.append({"severity": severity, "title": title,
                     "detail": detail, "file": file_hint})


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _py_files():
    return list(ROOT.rglob("*.py")) + list(ROOT.rglob("*.html"))


def _read(p: pathlib.Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _lines(p: pathlib.Path):
    return _read(p).splitlines()


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 1 — Hardcoded secrets / tokens
# ─────────────────────────────────────────────────────────────────────────────
SECRET_PATTERNS = [
    (r"dapi[0-9a-f]{32,}",               "Databricks PAT token (dapi…)"),
    (r"(?i)client[_-]?secret\s*=\s*['\"]([^'\"]{8,})['\"]",
                                          "Hardcoded client_secret"),
    (r"(?i)password\s*=\s*['\"]([^'\"]{4,})['\"]",
                                          "Hardcoded password"),
    (r"(?i)api[_-]?key\s*=\s*['\"]([^'\"]{8,})['\"]",
                                          "Hardcoded API key"),
    (r"(?i)secret\s*=\s*['\"]dose[0-9a-f]{20,}['\"]",
                                          "OAuth client secret (dose…)"),
    (r"(?i)token\s*=\s*['\"]dapi[0-9a-f]{32,}['\"]",
                                          "Token variable assigned a PAT"),
]

ALLOWED_SECRET_FILES = {"config.yaml.example", "security_scan.py",
                        "investigate_databricks.py"}

def check_hardcoded_secrets():
    for p in ROOT.rglob("*"):
        if p.is_dir() or p.suffix not in {".py", ".html", ".yaml", ".yml", ".env", ".txt"}:
            continue
        if p.name in ALLOWED_SECRET_FILES:
            continue
        content = _read(p)
        for pattern, label in SECRET_PATTERNS:
            for m in re.finditer(pattern, content):
                # Skip comments
                line_no = content[:m.start()].count("\n") + 1
                lines   = content.splitlines()
                src_line = lines[line_no - 1].strip() if line_no <= len(lines) else ""
                if src_line.startswith("#") or src_line.startswith("//"):
                    continue
                find("CRITICAL", f"Hardcoded secret: {label}",
                     f"Match: '{m.group(0)[:40]}…' at line {line_no}",
                     str(p.relative_to(ROOT)))


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 2 — SQL injection risk (f-string SQL + raw user input)
# ─────────────────────────────────────────────────────────────────────────────
def check_sql_injection():
    analyzer = ROOT / "app" / "analyzer.py"
    if not analyzer.exists():
        return
    content = _read(analyzer)
    # Look for f-string SQL blocks — count parameters that come from user_email
    # (user_email is already validated as an email via the identity header,
    # but format-string SQL is still a code-smell / injection surface)
    fstring_sql = re.findall(
        r'execute_query\(f""".*?WHERE.*?{user_email}.*?"""',
        content, re.DOTALL
    )
    if fstring_sql:
        find("MEDIUM",
             "SQL built with f-string interpolation (not parameterized)",
             f"{len(fstring_sql)} execute_query calls interpolate user_email directly into SQL. "
             "This relies on Databricks' connector sanitisation. "
             "Prefer parameterised queries when the connector supports them.",
             "app/analyzer.py")
    else:
        find("INFO", "SQL interpolation scan complete",
             "No obvious injection patterns beyond user_email in analyzer.py", "")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 3 — XSS via |safe filter
# ─────────────────────────────────────────────────────────────────────────────
def check_xss():
    for p in ROOT.rglob("*.html"):
        content = _read(p)
        # |safe on persona_json is intentional; flag any |safe on user text
        for no, line in enumerate(content.splitlines(), 1):
            if "|safe" in line and "persona_json" not in line and "system_prompt" not in line:
                # Check for variables that could contain user-typed content
                if re.search(r'\b(query_text|error_message|user_input|statement_text)\b', line):
                    find("HIGH", "Potential XSS: |safe on user-sourced content",
                         f"Line {no}: {line.strip()[:100]}", str(p.relative_to(ROOT)))


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 4 — Debug mode / Flask secret key
# ─────────────────────────────────────────────────────────────────────────────
def check_debug_mode():
    main = ROOT / "app" / "main.py"
    content = _read(main)
    # Debug is set from env var — only flag if hardcoded True
    if re.search(r'debug\s*=\s*True', content, re.IGNORECASE):
        find("HIGH", "Flask debug=True hardcoded",
             "Debug mode exposes interactive debugger — must be False in production.",
             "app/main.py")
    else:
        find("PASS", "Debug mode", "Controlled via FLASK_DEBUG env var. OK.", "")

    if "app.secret_key" not in content and "SECRET_KEY" not in content:
        find("MEDIUM", "No Flask secret_key set",
             "Flask session cookies are unsigned without a secret_key. "
             "Set app.secret_key = os.environ['FLASK_SECRET_KEY'] if sessions are used.",
             "app/main.py")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 5 — CSRF protection on POST endpoints
# ─────────────────────────────────────────────────────────────────────────────
def check_csrf():
    main = ROOT / "app" / "main.py"
    content = _read(main)
    post_routes = re.findall(r'@app\.route\("[^"]+",\s*methods=\["POST"\]', content)
    csrf_present = "CSRFProtect" in content or "csrf_token" in content or "flask_wtf" in content
    if post_routes and not csrf_present:
        find("MEDIUM",
             "No CSRF protection on POST endpoints",
             f"{len(post_routes)} POST endpoint(s) found but no CSRF middleware detected. "
             "Because this is a Databricks App behind reverse-proxy auth the risk is reduced, "
             "but consider flask-wtf or a custom CSRF token for defense-in-depth.",
             "app/main.py")
    else:
        find("PASS", "CSRF", "CSRF protection present or no POST endpoints.", "")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 6 — Rate limiting
# ─────────────────────────────────────────────────────────────────────────────
def check_rate_limiting():
    main = ROOT / "app" / "main.py"
    content = _read(main)
    if "flask_limiter" not in content and "Limiter" not in content and "rate_limit" not in content:
        find("LOW",
             "No rate limiting detected",
             "Heavy endpoints (run-analysis, chat) have no per-user rate limit. "
             "Consider flask-limiter to prevent abuse / cost overruns.",
             "app/main.py")
    else:
        find("PASS", "Rate limiting", "Rate limiting detected.", "")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 7 — Sensitive data in log statements
# ─────────────────────────────────────────────────────────────────────────────
def check_log_leakage():
    issues = []
    for p in ROOT.rglob("*.py"):
        for no, line in enumerate(_lines(p), 1):
            low = line.lower()
            # Flag logging of tokens / secrets (not in tests)
            if re.search(r'(log|print)\s*\(.*?(token|secret|password|dapi|bearer)', low):
                if "test_" not in p.name:
                    issues.append(f"{p.relative_to(ROOT)}:{no}: {line.strip()[:80]}")
    if issues:
        find("HIGH", f"Sensitive data may be logged ({len(issues)} instance(s))",
             "\n  ".join(issues[:5]) + (" …" if len(issues) > 5 else ""), "")
    else:
        find("PASS", "Log leakage", "No obvious token/secret logging found.", "")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 8 — Token handling safety
# ─────────────────────────────────────────────────────────────────────────────
def check_token_handling():
    main = ROOT / "app" / "main.py"
    content = _read(main)
    # Good: OBO token from header, not from body
    if "X-Forwarded-Access-Token" in content:
        find("PASS", "OBO token source",
             "Chat endpoint reads OBO token from X-Forwarded-Access-Token header. OK.", "")
    else:
        find("HIGH", "OBO token not read from trusted header",
             "Token should come from X-Forwarded-Access-Token injected by Databricks Apps proxy.",
             "app/main.py")

    # Check: token not echoed back in responses
    if re.search(r'jsonify.*?token', content, re.IGNORECASE):
        find("MEDIUM", "Token may be returned in API response",
             "Verify no token values are included in JSON responses.", "app/main.py")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 9 — Error message information leakage
# ─────────────────────────────────────────────────────────────────────────────
def check_error_leakage():
    for p in (ROOT / "app").rglob("*.py"):
        for no, line in enumerate(_lines(p), 1):
            # Exposing stack traces or internal paths in responses
            if re.search(r'traceback\.print|exc_info=True.*jsonify|str\(e\).*jsonify', line):
                find("MEDIUM", "Full exception may leak to API consumer",
                     f"{p.relative_to(ROOT)}:{no}: {line.strip()[:100]}", "")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 10 — config.yaml in version control
# ─────────────────────────────────────────────────────────────────────────────
def check_config_yaml_committed():
    gitignore = ROOT / ".gitignore"
    config    = ROOT / "config.yaml"

    if config.exists():
        # Check it's in .gitignore
        if gitignore.exists():
            gi_content = _read(gitignore)
            if "config.yaml" in gi_content:
                find("PASS", "config.yaml in .gitignore",
                     "config.yaml exists but is listed in .gitignore. OK.", "")
            else:
                find("CRITICAL", "config.yaml not in .gitignore",
                     "config.yaml exists and is NOT in .gitignore — may be committed to git.",
                     ".gitignore")
        else:
            find("HIGH", "No .gitignore — config.yaml may leak",
                 "config.yaml contains credentials. Create .gitignore immediately.", "")

        # Scan config.yaml for actual secrets
        cfg = _read(config)
        if re.search(r"dapi[0-9a-f]{32,}", cfg):
            find("CRITICAL", "Real Databricks PAT found in config.yaml",
                 "Rotate this token immediately if it has been committed to git.", "config.yaml")
        if re.search(r"dose[0-9a-f]{20,}", cfg):
            find("HIGH", "OAuth client secret (dose…) found in config.yaml",
                 "Ensure config.yaml is in .gitignore and not committed.", "config.yaml")
    else:
        find("PASS", "config.yaml", "config.yaml does not exist (using env vars). OK.", "")

    # Check git history for accidental commits
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "--all", "--", "config.yaml"],
            capture_output=True, text=True, cwd=ROOT, timeout=10
        )
        if result.stdout.strip():
            find("HIGH", "config.yaml was committed to git history",
                 f"Commits found:\n  {result.stdout.strip()[:200]}\n"
                 "Use 'git filter-repo' or BFG to purge secrets from history.",
                 "git history")
    except (subprocess.SubprocessError, FileNotFoundError):
        pass


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 11 — Python dependency vulnerabilities (pip-audit)
# ─────────────────────────────────────────────────────────────────────────────
def check_dependencies():
    try:
        # Use plain text output (compatible with all pip-audit versions)
        result = subprocess.run(
            [sys.executable, "-m", "pip_audit"],
            capture_output=True, text=True, cwd=ROOT, timeout=120
        )
        output = (result.stdout + result.stderr).strip()
        if result.returncode == 0:
            find("PASS", "pip-audit", "No known vulnerabilities in dependencies.", "")
        else:
            # Count lines that look like vulnerability entries
            vuln_lines = [l for l in output.splitlines()
                          if l.strip() and not l.startswith("Name") and ":" in l]
            detail = "\n  ".join(vuln_lines[:6]) if vuln_lines else output[:300]
            sev = "HIGH" if any("critical" in l.lower() for l in vuln_lines) else "MEDIUM"
            find(sev, f"pip-audit: vulnerable package(s) detected",
                 detail or "See pip-audit output above.", "requirements.txt")
    except FileNotFoundError:
        find("INFO", "pip-audit not installed",
             "Run: pip install pip-audit  then re-run this scan.", "")
    except subprocess.TimeoutExpired:
        find("INFO", "pip-audit timed out", "Could not complete dependency scan.", "")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 12 — Security headers / CORS
# ─────────────────────────────────────────────────────────────────────────────
def check_security_headers():
    main = ROOT / "app" / "main.py"
    content = _read(main)
    missing = []
    if "X-Content-Type-Options" not in content:
        missing.append("X-Content-Type-Options: nosniff")
    if "X-Frame-Options" not in content:
        missing.append("X-Frame-Options: DENY")
    if "Content-Security-Policy" not in content:
        missing.append("Content-Security-Policy")
    if missing:
        find("LOW",
             "Security response headers not set",
             "Missing: " + ", ".join(missing) + ". "
             "Databricks Apps proxy may add some of these, but explicit headers are better.",
             "app/main.py")
    else:
        find("PASS", "Security headers", "All key security headers present.", "")


# ─────────────────────────────────────────────────────────────────────────────
# RUN ALL CHECKS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "═" * 70)
print("  🔒  Digital Persona — Security Scan")
print("═" * 70 + "\n")

check_hardcoded_secrets()
check_sql_injection()
check_xss()
check_debug_mode()
check_csrf()
check_rate_limiting()
check_log_leakage()
check_token_handling()
check_error_leakage()
check_config_yaml_committed()
check_dependencies()
check_security_headers()

# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────

counts = {s: 0 for s in SEVERITY}
for f in findings:
    if f["severity"] in counts:
        counts[f["severity"]] += 1

# Sort by severity
findings.sort(key=lambda f: SEVERITY.get(f["severity"], 99))

for f in findings:
    sev  = f["severity"]
    col  = COLOR.get(sev, "")
    rst  = COLOR["RESET"]
    icon = {"CRITICAL": "🚨", "HIGH": "❗", "MEDIUM": "⚠️ ",
            "LOW": "ℹ️ ", "INFO": "💬", "PASS": "✅"}.get(sev, "  ")
    label = f"[{sev:<8}]"
    file_hint = f"  ({f['file']})" if f["file"] else ""
    print(f"{col}{icon} {label}{rst} {f['title']}{file_hint}")
    if sev not in ("PASS",) or f["detail"].startswith("No ") or True:
        # Show detail for non-PASS, or short PASS messages
        if f["detail"] and sev != "PASS":
            for line in f["detail"].splitlines():
                print(f"           {line}")
    print()

print("═" * 70)
print(f"  Summary: ", end="")
parts = []
for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"):
    c = counts.get(sev, 0)
    if c:
        parts.append(f"{COLOR[sev]}{c} {sev}{COLOR['RESET']}")
passes = sum(1 for f in findings if f["severity"] == "PASS")
if passes:
    parts.append(f"{COLOR['PASS']}{passes} PASS{COLOR['RESET']}")
print("  ".join(parts) if parts else "All clear!")
print("═" * 70 + "\n")

sys.exit(1 if counts.get("CRITICAL", 0) > 0 else 0)
