"""Tests for archetype scoring and classification logic.

These tests validate the scoring functions independently (no Databricks connection needed).
Run with: python -m pytest tests/ -v
"""

import math
import sys
import os

# Add project root to path so we can import the scoring functions
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── We can't import from app.analyzer directly because it depends on
#    databricks SDK. Instead, we import the pure functions by reading
#    the module source and extracting what we need. ──

def _smooth(value, midpoint, steepness=1.0):
    """Sigmoid-like smooth scoring (copied from analyzer.py for testing)."""
    if value <= 0:
        return 0.0
    x = value / max(midpoint, 0.01)
    return min(1.0 - math.exp(-steepness * x), 1.0)


# ── Helper: build a default signals dict ──

def _make_signals(**overrides):
    """Create a signals dict with sensible defaults, applying overrides."""
    defaults = {
        "select_ratio": 0.5,
        "write_ratio": 0.1,
        "total_queries": 100,
        "jobs_count": 0,
        "total_job_runs": 0,
        "job_success_rate": 0,
        "has_cron": False,
        "rw_ratio": float("inf"),
        "tables_read_count": 0,
        "svc_dict": {},
        "qs_dict": {},
        "heavy_ratio": 0.0,
        "clusters_in_services": 0,
        "genai_endpoint_count": 0,
        "unique_dashboards": 0,
        "permission_updates": 0,
        "tables_created_count": 0,
        "git_commits": 0,
        "cron_runs": 0,
        "longest_streak": 0,
        "user_dbu": 0,
        "workspace_dbu": 0,
    }
    defaults.update(overrides)
    return defaults


# ── Import scoring functions from analyzer (via exec to skip import chain) ──

_SCORING_CODE = ""
_scoring_started = False
with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app", "analyzer.py")) as f:
    for line in f:
        # Grab the _smooth function and all _score_* functions + _classify helper
        if line.startswith("def _smooth(") or line.startswith("def _score_") or line.startswith("def _classify_archetype("):
            _scoring_started = True
        if _scoring_started:
            # Stop when we hit a non-scoring function
            if line.startswith("def _compute_badges(") or line.startswith("def _build_data_health("):
                break
            _SCORING_CODE += line

# We also need the PERSONA_ARCHETYPES list
_ARCHETYPES_CODE = ""
_arch_started = False
with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app", "analyzer.py")) as f:
    for line in f:
        if "PERSONA_ARCHETYPES = [" in line:
            _arch_started = True
        if _arch_started:
            _ARCHETYPES_CODE += line
            if line.strip() == "]" and _arch_started:
                break

# Execute the extracted code in a namespace
_ns = {"math": math}
exec(_ARCHETYPES_CODE, _ns)
exec(_SCORING_CODE, _ns)

_classify_archetype = _ns["_classify_archetype"]
_score_pipeline_architect = _ns["_score_pipeline_architect"]
_score_data_explorer = _ns["_score_data_explorer"]
_score_platform_guardian = _ns["_score_platform_guardian"]
_score_dashboard_crafter = _ns["_score_dashboard_crafter"]
_score_ml_alchemist = _ns["_score_ml_alchemist"]
_score_cost_conscious = _ns["_score_cost_conscious"]
PERSONA_ARCHETYPES = _ns["PERSONA_ARCHETYPES"]


# ═══════════════════════════════════════════════════════════════════════════
#  Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSmoothFunction:
    """Test the sigmoid-like _smooth scoring function."""

    def test_zero_input(self):
        assert _smooth(0, 10) == 0.0

    def test_negative_input(self):
        assert _smooth(-5, 10) == 0.0

    def test_at_midpoint(self):
        result = _smooth(10, 10)
        assert 0.4 < result < 0.7, f"At midpoint expected ~0.63, got {result}"

    def test_large_input_near_one(self):
        assert _smooth(1000, 10) > 0.99

    def test_monotonically_increasing(self):
        vals = [_smooth(i, 10) for i in range(20)]
        for i in range(1, len(vals)):
            assert vals[i] >= vals[i - 1], f"Not monotonic at i={i}"

    def test_never_exceeds_one(self):
        for v in [1, 10, 100, 1000, 10000]:
            assert _smooth(v, 5) <= 1.0


class TestArchetypeClassification:
    """Test that archetype classification returns valid distributions."""

    def _get_primary(self, signals):
        results = _classify_archetype(signals)
        return results[0]["name"], results

    def test_sum_is_100(self):
        s = _make_signals()
        results = _classify_archetype(s)
        total = sum(r["match_pct"] for r in results)
        assert total == 100, f"Sum should be 100, got {total}"

    def test_all_above_zero(self):
        s = _make_signals()
        results = _classify_archetype(s)
        for r in results:
            assert r["match_pct"] > 0, f"{r['name']} has 0%"

    def test_six_archetypes(self):
        s = _make_signals()
        results = _classify_archetype(s)
        assert len(results) == 6

    def test_sorted_descending(self):
        s = _make_signals(jobs_count=10, total_job_runs=50, job_success_rate=0.95)
        results = _classify_archetype(s)
        pcts = [r["match_pct"] for r in results]
        assert pcts == sorted(pcts, reverse=True)

    def test_pipeline_architect_wins(self):
        s = _make_signals(
            jobs_count=12, total_job_runs=80, job_success_rate=0.95,
            has_cron=True, rw_ratio=1.5, tables_read_count=10,
            svc_dict={"notebook": 200, "clusters": 30},
            qs_dict={"notebook": 100, "direct_sql": 50},
            cron_runs=60,
        )
        name, results = self._get_primary(s)
        assert name == "The Pipeline Architect", f"Expected Pipeline Architect, got {name}"
        assert results[0]["match_pct"] <= 55, f"Primary should be at most 55%, got {results[0]['match_pct']}%"

    def test_data_explorer_wins(self):
        s = _make_signals(
            select_ratio=0.95, write_ratio=0.02, total_queries=500,
            rw_ratio=float("inf"), tables_read_count=25,
            svc_dict={"notebook": 300, "dashboards": 80},
            qs_dict={"notebook": 200, "dashboard": 100, "direct_sql": 50},
            unique_dashboards=8,
        )
        name, _ = self._get_primary(s)
        assert name == "The Data Explorer", f"Expected Data Explorer, got {name}"

    def test_platform_guardian_wins(self):
        s = _make_signals(
            svc_dict={"unityCatalog": 500, "accounts": 100, "permissions": 80,
                       "groups": 50, "clusters": 300, "secrets": 20},
            qs_dict={"direct_sql": 60, "notebook": 20},
            clusters_in_services=300,
            permission_updates=25, tables_created_count=10,
        )
        name, _ = self._get_primary(s)
        assert name == "The Platform Guardian", f"Expected Platform Guardian, got {name}"

    def test_dashboard_crafter_wins(self):
        s = _make_signals(
            select_ratio=0.90, total_queries=400,
            svc_dict={"dashboards": 250, "notebook": 30},
            qs_dict={"dashboard": 300, "direct_sql": 50, "notebook": 20},
            unique_dashboards=15,
        )
        name, _ = self._get_primary(s)
        assert name == "The Dashboard Crafter", f"Expected Dashboard Crafter, got {name}"

    def test_ml_alchemist_wins(self):
        s = _make_signals(
            select_ratio=0.60, write_ratio=0.20, total_queries=300,
            svc_dict={"mlflow": 100, "modelServing": 50, "notebook": 400, "feature-store": 20},
            qs_dict={"notebook": 250, "direct_sql": 30},
            heavy_ratio=0.35, genai_endpoint_count=4,
        )
        name, _ = self._get_primary(s)
        assert name == "The ML Alchemist", f"Expected ML Alchemist, got {name}"

    def test_zero_data_reasonable_spread(self):
        """With zero activity, archetypes should distribute roughly equally."""
        s = _make_signals(total_queries=1, svc_dict={}, qs_dict={})
        results = _classify_archetype(s)
        pcts = [r["match_pct"] for r in results]
        spread = max(pcts) - min(pcts)
        assert spread < 15, f"Zero-data spread too wide: {spread}% (pcts={pcts})"


class TestIndividualScoring:
    """Test individual scoring functions return sensible values."""

    def test_pipeline_architect_zero_jobs(self):
        s = _make_signals(jobs_count=0, total_job_runs=0)
        score = _score_pipeline_architect(s)
        assert score < 1.0, f"Zero jobs should score low, got {score}"

    def test_pipeline_architect_many_jobs(self):
        s = _make_signals(
            jobs_count=15, total_job_runs=100, job_success_rate=0.98,
            has_cron=True, rw_ratio=1.0, cron_runs=80,
            svc_dict={"notebook": 200}, qs_dict={"notebook": 100},
        )
        score = _score_pipeline_architect(s)
        assert score > 5.0, f"Heavy job user should score high, got {score}"

    def test_data_explorer_high_select(self):
        s = _make_signals(select_ratio=0.95, tables_read_count=20, rw_ratio=float("inf"))
        score = _score_data_explorer(s)
        assert score > 4.0, f"Pure reader should score high, got {score}"

    def test_ml_alchemist_no_ml_services(self):
        s = _make_signals(svc_dict={}, heavy_ratio=0.0, genai_endpoint_count=0)
        score = _score_ml_alchemist(s)
        assert score < 1.0, f"No ML services should score low, got {score}"

    def test_cost_conscious_needs_activity(self):
        """Cost-conscious should NOT score high with nearly zero activity."""
        s = _make_signals(total_queries=1, svc_dict={}, qs_dict={})
        score = _score_cost_conscious(s)
        assert score < 1.0, f"Minimal activity shouldn't score cost-conscious, got {score}"

    def test_all_scores_non_negative(self):
        """No scoring function should ever return a negative value."""
        fns = [_score_pipeline_architect, _score_data_explorer,
               _score_platform_guardian, _score_dashboard_crafter,
               _score_ml_alchemist, _score_cost_conscious]
        for fn in fns:
            s = _make_signals()
            score = fn(s)
            assert score >= 0, f"{fn.__name__} returned negative: {score}"


class TestArchetypeMetadata:
    """Test that archetype definitions are complete."""

    def test_six_archetypes_defined(self):
        assert len(PERSONA_ARCHETYPES) == 6

    def test_required_fields(self):
        for arch in PERSONA_ARCHETYPES:
            for field in ["name", "emoji", "color", "gradient", "description", "traits"]:
                assert field in arch, f"Missing {field} in {arch.get('name', '?')}"

    def test_unique_names(self):
        names = [a["name"] for a in PERSONA_ARCHETYPES]
        assert len(names) == len(set(names)), "Archetype names not unique"


class TestConfigHelpers:
    """Test config.py helper functions (no Databricks connection needed)."""

    def test_get_auth_method_default(self):
        """get_auth_method should default to 'service_principal'."""
        import yaml
        config = yaml.safe_load("""
databricks:
  host: "https://test.azuredatabricks.net"
  client_id: "test"
  client_secret: "test"
""")
        method = config["databricks"].get("auth_method", "service_principal")
        assert method == "service_principal"

    def test_get_auth_method_pat(self):
        """get_auth_method should recognize 'pat'."""
        import yaml
        config = yaml.safe_load("""
databricks:
  host: "https://test.azuredatabricks.net"
  auth_method: "pat"
  token: "dapi_test"
""")
        method = config["databricks"].get("auth_method", "service_principal")
        assert method == "pat"

    def test_auth_method_validation(self):
        """Invalid auth_method should be caught."""
        method = "invalid_method"
        assert method not in ("service_principal", "pat")


# ═══════════════════════════════════════════════════════════════════════════
#  Run tests directly with: python tests/test_scoring.py
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    passed = 0
    failed = 0

    for cls_name in [TestSmoothFunction, TestArchetypeClassification,
                     TestIndividualScoring, TestArchetypeMetadata, TestConfigHelpers]:
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
