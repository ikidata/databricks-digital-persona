"""Tests for the in-memory PersonaStore.

Validates thread-safe storage, data isolation between users, and all
CRUD operations on persona data, analysis status, access results, and
LLM-generated prompts.

Run with: python -m pytest tests/test_store.py -v
"""

import threading
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.store import PersonaStore


# ═══════════════════════════════════════════════════════════════════════════
#  Test: Persona data CRUD
# ═══════════════════════════════════════════════════════════════════════════

class TestPersonaData:
    """Test basic persona data set/get/has/clear operations."""

    def _fresh_store(self):
        return PersonaStore()

    def test_no_data_initially(self):
        s = self._fresh_store()
        assert not s.has_persona_data("user@test.com")
        assert s.get_persona_data("user@test.com") is None

    def test_set_and_get(self):
        s = self._fresh_store()
        data = {"archetype": {"primary": "Explorer"}, "badges": []}
        s.set_persona_data("user@test.com", data)
        entry = s.get_persona_data("user@test.com")
        assert entry is not None
        assert entry["persona"] == data
        assert "timestamp" in entry

    def test_has_persona_data(self):
        s = self._fresh_store()
        s.set_persona_data("user@test.com", {"k": "v"})
        assert s.has_persona_data("user@test.com")
        assert not s.has_persona_data("other@test.com")

    def test_overwrite_persona_data(self):
        s = self._fresh_store()
        s.set_persona_data("user@test.com", {"version": 1})
        s.set_persona_data("user@test.com", {"version": 2})
        entry = s.get_persona_data("user@test.com")
        assert entry["persona"]["version"] == 2

    def test_clear_removes_all(self):
        s = self._fresh_store()
        s.set_persona_data("user@test.com", {"x": 1})
        s.set_access_results("user@test.com", [{"table": "t", "ok": True}])
        s.set_analysis_status("user@test.com", "complete", 100, "Done")
        s.set_llm_prompt("user@test.com", "You are...")

        s.clear("user@test.com")

        assert not s.has_persona_data("user@test.com")
        assert s.get_access_results("user@test.com") is None
        assert s.get_analysis_status("user@test.com")["status"] == "idle"
        assert s.get_llm_prompt("user@test.com") is None


# ═══════════════════════════════════════════════════════════════════════════
#  Test: User data isolation
# ═══════════════════════════════════════════════════════════════════════════

class TestDataIsolation:
    """Ensure one user's data never leaks to another."""

    def test_different_users_isolated(self):
        s = PersonaStore()
        s.set_persona_data("alice@test.com", {"user": "alice"})
        s.set_persona_data("bob@test.com", {"user": "bob"})

        alice = s.get_persona_data("alice@test.com")
        bob = s.get_persona_data("bob@test.com")

        assert alice["persona"]["user"] == "alice"
        assert bob["persona"]["user"] == "bob"

    def test_clear_one_user_keeps_other(self):
        s = PersonaStore()
        s.set_persona_data("alice@test.com", {"user": "alice"})
        s.set_persona_data("bob@test.com", {"user": "bob"})

        s.clear("alice@test.com")

        assert not s.has_persona_data("alice@test.com")
        assert s.has_persona_data("bob@test.com")

    def test_access_results_isolated(self):
        s = PersonaStore()
        s.set_access_results("alice@test.com", [{"table": "a"}])
        s.set_access_results("bob@test.com", [{"table": "b"}])

        assert s.get_access_results("alice@test.com")["results"][0]["table"] == "a"
        assert s.get_access_results("bob@test.com")["results"][0]["table"] == "b"


# ═══════════════════════════════════════════════════════════════════════════
#  Test: Analysis status
# ═══════════════════════════════════════════════════════════════════════════

class TestAnalysisStatus:
    """Test analysis status tracking."""

    def test_default_status_is_idle(self):
        s = PersonaStore()
        status = s.get_analysis_status("new@test.com")
        assert status["status"] == "idle"
        assert status["progress"] == 0

    def test_set_running(self):
        s = PersonaStore()
        s.set_analysis_status("user@test.com", "running", 50, "Step 7/15")
        status = s.get_analysis_status("user@test.com")
        assert status["status"] == "running"
        assert status["progress"] == 50
        assert status["message"] == "Step 7/15"

    def test_set_complete(self):
        s = PersonaStore()
        s.set_analysis_status("user@test.com", "complete", 100, "Done")
        status = s.get_analysis_status("user@test.com")
        assert status["status"] == "complete"
        assert status["progress"] == 100

    def test_set_error(self):
        s = PersonaStore()
        s.set_analysis_status("user@test.com", "error", 33, "SQL timeout")
        status = s.get_analysis_status("user@test.com")
        assert status["status"] == "error"
        assert "timeout" in status["message"].lower()


# ═══════════════════════════════════════════════════════════════════════════
#  Test: LLM prompt storage
# ═══════════════════════════════════════════════════════════════════════════

class TestLLMPrompts:
    """Test LLM-generated prompt storage."""

    def test_no_prompt_initially(self):
        s = PersonaStore()
        assert s.get_llm_prompt("user@test.com") is None

    def test_set_and_get_prompt(self):
        s = PersonaStore()
        s.set_llm_prompt("user@test.com", "You are a Pipeline Architect...")
        prompt = s.get_llm_prompt("user@test.com")
        assert prompt == "You are a Pipeline Architect..."

    def test_overwrite_prompt(self):
        s = PersonaStore()
        s.set_llm_prompt("user@test.com", "Version 1")
        s.set_llm_prompt("user@test.com", "Version 2")
        assert s.get_llm_prompt("user@test.com") == "Version 2"


# ═══════════════════════════════════════════════════════════════════════════
#  Test: Thread safety
# ═══════════════════════════════════════════════════════════════════════════

class TestThreadSafety:
    """Verify the store is safe under concurrent access."""

    def test_concurrent_writes(self):
        """Multiple threads writing persona data should not corrupt the store."""
        s = PersonaStore()
        errors = []

        def writer(email, value):
            try:
                for i in range(50):
                    s.set_persona_data(email, {"iteration": i, "value": value})
            except Exception as e:
                errors.append(str(e))

        threads = [
            threading.Thread(target=writer, args=(f"user{i}@test.com", i))
            for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent write errors: {errors}"
        # All 10 users should have data
        for i in range(10):
            assert s.has_persona_data(f"user{i}@test.com")

    def test_concurrent_reads_and_writes(self):
        """Reads and writes should not deadlock or raise."""
        s = PersonaStore()
        s.set_persona_data("user@test.com", {"initial": True})
        errors = []

        def reader():
            try:
                for _ in range(100):
                    s.get_persona_data("user@test.com")
                    s.has_persona_data("user@test.com")
            except Exception as e:
                errors.append(str(e))

        def writer():
            try:
                for i in range(100):
                    s.set_persona_data("user@test.com", {"iter": i})
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=reader) for _ in range(5)]
        threads += [threading.Thread(target=writer) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent read/write errors: {errors}"


# ═══════════════════════════════════════════════════════════════════════════
#  Run tests directly with: python tests/test_store.py
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    passed = 0
    failed = 0

    for cls_name in [TestPersonaData, TestDataIsolation, TestAnalysisStatus,
                     TestLLMPrompts, TestThreadSafety]:
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
