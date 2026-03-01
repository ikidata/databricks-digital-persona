"""In-memory data store to persist data across page navigation.

Since persona analysis is a heavy operation, we store all results in memory
so that navigating between pages doesn't trigger re-queries. Data is keyed
by user email.
"""

import threading
from datetime import datetime


class PersonaStore:
    """Thread-safe in-memory store for persona data."""

    def __init__(self):
        self._lock = threading.Lock()
        self._data = {}  # keyed by user email
        self._access_results = {}  # keyed by user email
        self._analysis_status = {}  # keyed by user email
        self._llm_prompts = {}  # keyed by user email — LLM-generated system prompts

    def set_access_results(self, user_email, results):
        with self._lock:
            self._access_results[user_email] = {
                "results": results,
                "timestamp": datetime.utcnow().isoformat(),
            }

    def get_access_results(self, user_email):
        with self._lock:
            return self._access_results.get(user_email)

    def set_analysis_status(self, user_email, status, progress=0, message=""):
        with self._lock:
            self._analysis_status[user_email] = {
                "status": status,  # "idle", "running", "complete", "error"
                "progress": progress,
                "message": message,
                "timestamp": datetime.utcnow().isoformat(),
            }

    def get_analysis_status(self, user_email):
        with self._lock:
            return self._analysis_status.get(user_email, {
                "status": "idle",
                "progress": 0,
                "message": "",
            })

    def set_persona_data(self, user_email, data):
        with self._lock:
            self._data[user_email] = {
                "persona": data,
                "timestamp": datetime.utcnow().isoformat(),
            }

    def get_persona_data(self, user_email):
        with self._lock:
            return self._data.get(user_email)

    def has_persona_data(self, user_email):
        with self._lock:
            return user_email in self._data

    def set_llm_prompt(self, user_email, prompt_text):
        """Store an LLM-generated system prompt for the user."""
        with self._lock:
            self._llm_prompts[user_email] = {
                "prompt": prompt_text,
                "timestamp": datetime.utcnow().isoformat(),
            }

    def get_llm_prompt(self, user_email):
        """Return the stored LLM-generated prompt text, or None."""
        with self._lock:
            entry = self._llm_prompts.get(user_email)
            return entry["prompt"] if entry else None

    def clear(self, user_email):
        with self._lock:
            self._data.pop(user_email, None)
            self._access_results.pop(user_email, None)
            self._analysis_status.pop(user_email, None)
            self._llm_prompts.pop(user_email, None)


# Global singleton store
store = PersonaStore()
