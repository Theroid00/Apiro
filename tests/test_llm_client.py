from apiro.llm_client import OllamaLLMClient


def test_json_generation_pins_seed_and_requests_json(monkeypatch):
    captured = {}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": '{"hypotheses":[]}'}

    def post(url, *, json, timeout):
        captured.update(url=url, payload=json, timeout=timeout)
        return Response()

    monkeypatch.setattr("apiro.llm_client.requests.post", post)
    client = OllamaLLMClient(
        "http://ollama", "model", seed=19, json_num_predict=768
    )

    assert client.generate_json("prompt") == '{"hypotheses":[]}'
    assert captured["payload"]["format"] == "json"
    assert captured["payload"]["options"]["seed"] == 19
    assert captured["payload"]["options"]["num_predict"] == 768


def test_json_generation_accepts_an_ollama_schema(monkeypatch):
    captured = {}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": "{}"}

    def post(_url, *, json, timeout):
        captured.update(payload=json, timeout=timeout)
        return Response()

    monkeypatch.setattr("apiro.llm_client.requests.post", post)
    schema = {"type": "object", "required": ["hypotheses"]}

    OllamaLLMClient("http://ollama", "model").generate_json("prompt", schema=schema)

    assert captured["payload"]["format"] == schema
