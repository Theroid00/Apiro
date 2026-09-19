"""Runtime resources share heavy clients while isolating mutable run state."""

from apiro.application.runtime import RuntimeResources


class _Embedder:
    def query(self, *_args, **_kwargs):
        return []


class _LLM:
    def generate(self, _prompt):
        return ""

    chat = generate


def test_create_service_returns_fresh_bounded_services():
    resources = RuntimeResources(
        embedder=_Embedder(),
        llm_client=_LLM(),
        axiom_extractor=object(),
        doc_count=1,
        model="stub",
        ollama_url="http://invalid.test",
    )

    first = resources.create_service(default_mode="simple")
    second = resources.create_service(default_mode="investigator")

    assert first is not second
    assert first.default_mode == "simple"
    assert second.default_mode == "investigator"
