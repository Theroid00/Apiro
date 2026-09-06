"""Runtime resources share heavy clients while isolating mutable run state."""

from apiro.application.runtime import RuntimeResources


class _Embedder:
    def query(self, *_args, **_kwargs):
        return []


class _LLM:
    def generate(self, _prompt):
        return ""

    chat = generate


def test_create_traversal_returns_fresh_mutable_components():
    resources = RuntimeResources(
        embedder=_Embedder(),
        llm_client=_LLM(),
        axiom_extractor=object(),
        doc_count=1,
        model="stub",
        ollama_url="http://invalid.test",
    )

    first = resources.create_traversal()
    second = resources.create_traversal()

    assert first is not second
    assert first.expander is not second.expander
    assert first.saturation is not second.saturation
    assert first.rabbit_hole is not second.rabbit_hole
    assert first.contradiction is not second.contradiction
    assert first.expander.llm_client is second.expander.llm_client
    assert first.expander.chroma_client._embedder is resources.embedder
    assert second.expander.chroma_client._embedder is resources.embedder
