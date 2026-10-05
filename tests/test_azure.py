"""Exercise Azure SDK routing without credentials, network, or paid calls."""
import json
import httpx
import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
import config
from src import generator, embeddings


def configure(monkeypatch):
    monkeypatch.setattr(config, 'AZURE_OPENAI_API_KEY', 'offline-test-key')
    monkeypatch.setattr(config, 'AZURE_OPENAI_ENDPOINT', 'https://offline-test.openai.azure.com')
    monkeypatch.setattr(config, 'AZURE_OPENAI_CHAT_DEPLOYMENT', 'my-chat')
    monkeypatch.setattr(config, 'AZURE_OPENAI_EMBEDDING_DEPLOYMENT', 'my-embedding')
    monkeypatch.setattr(generator, 'AZURE_OPENAI_CHAT_DEPLOYMENT', 'my-chat')
    monkeypatch.setattr(embeddings, 'AZURE_OPENAI_EMBEDDING_DEPLOYMENT', 'my-embedding')


def test_azure_requests_and_grounded_answer(monkeypatch):
    configure(monkeypatch)
    # All HTTP requests are intercepted below; isolate from machine proxy settings.
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    calls = []

    def handle(transport, request):
        body = json.loads(request.content)
        calls.append((request, body))
        assert request.headers['api-key'] == 'offline-test-key'
        assert request.url.params['api-version'] == config.AZURE_OPENAI_API_VERSION
        if request.url.path.endswith('/embeddings'):
            assert request.url.path == '/openai/deployments/my-embedding/embeddings'
            assert body['model'] == embeddings.EMBEDDING_MODEL
            result = {'object': 'list', 'model': 'text-embedding-3-small',
                      'data': [{'object': 'embedding', 'index': 0, 'embedding': [0.2, 0.8]}],
                      'usage': {'prompt_tokens': 2, 'total_tokens': 2}}
        else:
            assert request.url.path == '/openai/deployments/my-chat/chat/completions'
            assert 'temperature' not in body
            assert 'Use only the supplied passages' in body['messages'][0]['content']
            assert '[1] notes.txt' in body['messages'][1]['content']
            result = {'id': 'offline', 'object': 'chat.completion', 'created': 1,
                      'model': 'gpt-4o', 'choices': [{'index': 0, 'finish_reason': 'stop',
                      'message': {'role': 'assistant', 'content': 'The robot is blue [1].'}}],
                      'usage': {'prompt_tokens': 20, 'completion_tokens': 8, 'total_tokens': 28}}
        return httpx.Response(200, json=result, request=request)

    monkeypatch.setattr(httpx.HTTPTransport, 'handle_request', handle)
    embed = embeddings.get_embeddings()
    # Avoid fetching a tokenizer file in this offline transport test.
    embed.check_embedding_ctx_length = False
    assert embed.embed_query('robot') == [0.2, 0.8]
    retriever = RunnableLambda(lambda _: [Document(page_content='The robot is blue.', metadata={'source': 'notes.txt'})])
    answer = generator.create_rag_chain(retriever).invoke('What color is the robot?')
    assert answer['status'] == 'answered'
    assert answer['sources'][0]['file_name'] == 'notes.txt'
    assert len(calls) == 2


def test_missing_deployment_and_invalid_endpoint(monkeypatch):
    configure(monkeypatch)
    assert config.validate_api_keys() == (True, [])
    with pytest.raises(ValueError, match='configuration is missing'):
        config.azure_client_settings('')
    monkeypatch.setattr(config, 'AZURE_OPENAI_ENDPOINT', 'https://offline-test.openai.azure.com/openai/v1/')
    with pytest.raises(ValueError, match='resource root'):
        config.azure_client_settings('my-chat')
    monkeypatch.setattr(config, 'AZURE_OPENAI_CHAT_DEPLOYMENT', '')
    assert 'AZURE_OPENAI_CHAT_DEPLOYMENT' in config.validate_api_keys()[1]
