"""Tests for the brain.py provider-class refactor.

Verifies model selection is centralized and that the Ollama provider can never
silently fall back to the weak groq LLM_MODEL — the root cause of the earlier
hallucination/derailment bug. No network calls; only env wiring is exercised.
"""
import os
import sys
import unittest
from pathlib import Path

# Make backend/src importable when run from repo root or backend/.
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


class ProviderWiringTests(unittest.TestCase):
    def setUp(self):
        # Snapshot env so each test is isolated.
        self._env = dict(os.environ)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)

    def test_ollama_provider_uses_ollama_llm_model(self):
        os.environ["OLLAMA_LLM_MODEL"] = "qwen2.5:14b"
        os.environ["OLLAMA_MODEL"] = "legacy:7b"
        os.environ["LLM_MODEL"] = "llama-3.1-8b-instant"
        import brain

        provider = brain.build_ollama_provider()
        self.assertEqual(provider.model, "qwen2.5:14b")
        self.assertEqual(provider.name, "ollama")
        # The weak groq model must never leak into the ollama provider.
        self.assertNotEqual(provider.model, os.environ["LLM_MODEL"])

    def test_ollama_provider_legacy_fallback_not_llm_model(self):
        # OLLAMA_LLM_MODEL unset -> fall back to OLLAMA_MODEL, never LLM_MODEL.
        os.environ.pop("OLLAMA_LLM_MODEL", None)
        os.environ["OLLAMA_MODEL"] = "legacy:7b"
        os.environ["LLM_MODEL"] = "llama-3.1-8b-instant"
        import brain

        provider = brain.build_ollama_provider()
        self.assertEqual(provider.model, "legacy:7b")

    def test_groq_provider_uses_llm_model(self):
        os.environ["LLM_MODEL"] = "llama-3.1-8b-instant"
        os.environ.pop("GROQ_API_KEY", None)
        import brain

        provider = brain.build_groq_provider()
        self.assertEqual(provider.model, "llama-3.1-8b-instant")
        self.assertEqual(provider.name, "groq")

    def test_router_ollama(self):
        os.environ["LLM_PROVIDER"] = "ollama"
        os.environ["OLLAMA_LLM_MODEL"] = "qwen2.5:14b"
        import brain

        provider = brain.get_main_provider()
        self.assertIsInstance(provider, brain.OllamaProvider)
        self.assertEqual(provider.model, "qwen2.5:14b")

    def test_router_groq_default(self):
        os.environ["LLM_PROVIDER"] = "groq"
        os.environ["LLM_MODEL"] = "llama-3.1-8b-instant"
        import brain

        provider = brain.get_main_provider()
        self.assertIsInstance(provider, brain.GroqProvider)

    def test_router_unknown_falls_back_to_groq(self):
        os.environ["LLM_PROVIDER"] = "openai"  # no longer a class
        os.environ["LLM_MODEL"] = "llama-3.1-8b-instant"
        import brain

        provider = brain.get_main_provider()
        self.assertIsInstance(provider, brain.GroqProvider)

    def test_explicit_model_override_in_groq_chat_complete(self):
        # groq_chat_complete with explicit model builds a one-off provider
        # carrying that model, regardless of env defaults.
        os.environ["LLM_PROVIDER"] = "groq"
        os.environ["LLM_MODEL"] = "llama-3.1-8b-instant"
        os.environ.pop("GROQ_API_KEY", None)
        import brain

        captured = {}

        class FakeProvider:
            def __init__(self, model):
                self.name = "groq"
                self.model = model

            def chat(self, messages, raw=False, **kwargs):
                captured["model"] = self.model
                captured["raw"] = raw
                return "ok"

        original_get = brain.get_main_provider
        original_build_groq = brain.build_groq_provider
        brain.get_main_provider = lambda: FakeProvider("env-default")
        brain.build_groq_provider = lambda model=None: FakeProvider(model)
        try:
            brain.groq_chat_complete(["m"], model="override-model", raw=True)
        finally:
            brain.get_main_provider = original_get
            brain.build_groq_provider = original_build_groq
        self.assertEqual(captured["model"], "override-model")
        self.assertTrue(captured["raw"])


if __name__ == "__main__":
    unittest.main()