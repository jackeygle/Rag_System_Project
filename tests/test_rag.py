import math
import sys
from pathlib import Path
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableLambda
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import vector_store as indexing
from src.generator import create_rag_chain, validate_answer, format_docs, NO_EVIDENCE
from src.retriever import create_retriever


class LocalEmbeddings(Embeddings):
    def __init__(self): self.calls = 0
    def vector(self, text):
        vector = [float(text.lower().count(word)) for word in ['robot', 'finance', 'blue', 'green']]
        vector.append(0.1)
        norm = math.sqrt(sum(x*x for x in vector))
        return [x/norm for x in vector]
    def embed_documents(self, texts):
        self.calls += len(texts)
        return [self.vector(text) for text in texts]
    def embed_query(self, text): return self.vector(text)


@pytest.fixture
def library(tmp_path, monkeypatch):
    monkeypatch.setattr(indexing, 'DB_DIR', tmp_path / 'vectors')
    indexing.DB_DIR.mkdir()
    embeddings = LocalEmbeddings()
    store = indexing.open_vector_store(embedding=embeddings)
    return tmp_path, store, embeddings


def doc(text, source='a.pdf', page=0):
    return Document(page_content=text, metadata={'source': source, 'file_name': Path(source).name, 'page': page})


def test_empty_retrieval_never_calls_llm():
    def fail(_): raise AssertionError('LLM must not be called')
    chain = create_rag_chain(RunnableLambda(lambda _: []), llm=RunnableLambda(fail))
    assert chain.invoke('Who is the CEO?')['answer'] == NO_EVIDENCE


def test_sources_match_exact_retrieved_passages():
    docs = [doc('The robot is blue.', page=2)]
    chain = create_rag_chain(RunnableLambda(lambda _: docs), llm=RunnableLambda(lambda _: 'The robot is blue [1].'))
    result = chain.invoke('What colour is the robot?')
    assert result['sources'][0]['page'] == 3
    assert result['sources'][0]['excerpt'] == 'The robot is blue.'
    assert 'page 3' in format_docs(docs)


@pytest.mark.parametrize('answer', ['Unsupported statement.', 'Wrong citation [0].', 'Wrong citation [2].'])
def test_invalid_or_absent_citations_are_rejected(answer):
    assert validate_answer(answer, [doc('Evidence')])['status'] == 'insufficient_evidence'


def test_duplicate_citations_have_one_source():
    assert len(validate_answer('Robot [1]. Blue [1].', [doc('Blue robot')])['sources']) == 1


def test_incremental_update_and_delete(library):
    _, store, emb = library
    indexing.create_vector_store([doc('blue robot')], vector_store=store)
    assert emb.calls == 1
    indexing.create_vector_store([doc('blue robot')], vector_store=store)
    assert emb.calls == 1
    indexing.create_vector_store([doc('green robot')], vector_store=store)
    assert emb.calls == 2
    assert store._collection.count() == 1
    assert store.get()['documents'] == ['green robot']
    indexing.delete_indexed_document('a.pdf', vector_store=store)
    assert store._collection.count() == 0
    assert indexing.list_indexed_documents() == []


def test_scoped_retrieval(library):
    _, store, _ = library
    indexing.create_vector_store([doc('blue robot', 'robot.pdf'), doc('finance', 'finance.pdf')], vector_store=store)
    retrieve = create_retriever(store, sources=['finance.pdf'], min_relevance=0)
    assert [d.metadata['source'] for d in retrieve.invoke('robot')] == ['finance.pdf']
    assert create_retriever(store, sources=[]).invoke('robot') == []
    assert create_retriever(store, min_relevance=.9).invoke('blue robot')[0].metadata['source'] == 'robot.pdf'


def test_sync_skips_unchanged_files_and_handles_deleted_files(library, monkeypatch):
    tmp, store, emb = library
    directory = tmp / 'docs'; directory.mkdir()
    path = directory / 'robot.txt'; path.write_text('blue robot')
    indexing.sync_directory(directory, vector_store=store)
    before = emb.calls
    from src import document_loader
    loader = document_loader.load_single_document
    monkeypatch.setattr(document_loader, 'load_single_document', lambda _: pytest.fail('Unchanged file was parsed again'))
    indexing.sync_directory(directory, vector_store=store)
    assert emb.calls == before
    monkeypatch.setattr(document_loader, 'load_single_document', loader)
    path.write_text('green robot')
    indexing.sync_directory(directory, vector_store=store)
    assert emb.calls > before
    path.unlink()
    indexing.sync_directory(directory, vector_store=store)
    assert store._collection.count() == 0


def test_failed_replacement_keeps_previous_vectors(library, monkeypatch):
    _, store, _ = library
    indexing.create_vector_store([doc('blue robot')], vector_store=store)
    monkeypatch.setattr(store, 'add_documents', lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('Provider unavailable')))
    with pytest.raises(RuntimeError): indexing.create_vector_store([doc('green robot')], vector_store=store)
    assert store.get()['documents'] == ['blue robot']


def test_persistent_store_reopens_without_reembedding(library):
    _, store, emb = library
    indexing.create_vector_store([doc('blue robot')], vector_store=store)
    reopened = indexing.open_vector_store(embedding=emb)
    assert reopened._collection.count() == 1
    assert emb.calls == 1


def test_blank_query_rejected():
    with pytest.raises(ValueError): create_rag_chain(RunnableLambda(lambda _: [])).invoke(' ')


def test_streamlit_starts_without_credentials(monkeypatch):
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'streamlit_app.py'))
    monkeypatch.setattr('config.validate_api_keys', lambda: (False, ['AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_ENDPOINT']))
    app.run(timeout=20)
    assert not app.exception
    assert app.chat_input[0].disabled
    assert len(app.main.chat_input) == 1  # Composer belongs to the page, not the pinned bottom area.
    assert len(app.warning) == 1


def test_upload_rejects_empty_and_preserves_existing_file(tmp_path, monkeypatch):
    import streamlit_app
    monkeypatch.setattr(streamlit_app, 'DATA_DIR', tmp_path)
    original = tmp_path / 'report.txt'; original.write_text('Old evidence')
    class Upload:
        name = 'report.txt'
        def getvalue(self): return b''
    with pytest.raises(ValueError): streamlit_app.save_upload(Upload())
    assert original.read_text() == 'Old evidence'


def test_upload_sanitizes_path_and_rejects_unreadable_text(tmp_path, monkeypatch):
    import streamlit_app
    monkeypatch.setattr(streamlit_app, 'DATA_DIR', tmp_path)
    class Upload:
        name = '../../report.txt'
        def getvalue(self): return b'Blue robot.'
    assert streamlit_app.save_upload(Upload()) == 'report.txt'
    assert (tmp_path / 'report.txt').read_text() == 'Blue robot.'
    class Blank(Upload):
        def getvalue(self): return b'   '
    with pytest.raises(ValueError): streamlit_app.save_upload(Blank())
    assert (tmp_path / 'report.txt').read_text() == 'Blue robot.'


def test_streamlit_question_displays_cited_source(library, monkeypatch):
    from streamlit.testing.v1 import AppTest
    import config
    from src import generator
    tmp, store, _ = library
    directory = tmp / 'ui-docs'; directory.mkdir()
    (directory / 'robot.txt').write_text('The robot is blue.')
    original_sync = indexing.sync_directory
    monkeypatch.setattr(config, 'DATA_DIR', directory)
    monkeypatch.setattr(config, 'validate_api_keys', lambda: (True, []))
    monkeypatch.setattr(indexing, 'sync_directory', lambda path: original_sync(path, vector_store=store))
    monkeypatch.setattr(generator, 'get_llm', lambda: RunnableLambda(lambda _: 'The robot is blue [1].'))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'streamlit_app.py')).run(timeout=20)
    assert not app.exception
    app.chat_input[0].set_value('What colour is the robot?').run(timeout=20)
    assert not app.exception
    assert 'source_reader_open' not in app.session_state or not app.session_state['source_reader_open']
    app.button(key='cite_1_1').click().run()
    assert any(exp.label == '[1] robot.txt' for exp in app.expander)
    assert any(item.value == 'The robot is blue.' for item in app.text)
    app.button(key='close_reader').click().run()
    assert not app.session_state['source_reader_open']
    app.radio[0].set_value('Selected documents').run()
    assert app.chat_input[0].disabled
    app.radio[0].set_value('Entire library').run()
    app.chat_input[0].set_value('Tell me about the robot again.').run()
    assert not app.exception
    app.button(key='cite_1_1').click().run()
    assert app.session_state['source_answer'] == 1
    assert any(item.value == 'What colour is the robot?' for item in app.markdown)
    next(button for button in app.button if button.label == 'New conversation').click().run()
    assert not app.exception
    assert app.session_state['messages'] == []
    assert not app.chat_message


def test_indexing_progress_reports_real_file_stages(library):
    tmp, store, _ = library
    folder = tmp / 'progress-docs'; folder.mkdir()
    (folder / 'robot.txt').write_text('The robot is blue.')
    events = []
    indexing.sync_directory(folder, vector_store=store, progress=lambda *event: events.append(event))
    assert [event[0] for event in events] == ['checking', 'parsing', 'indexing', 'ready']
    assert events[-1][2:] == (1, 1)
    events.clear()
    indexing.sync_directory(folder, vector_store=store, progress=lambda *event: events.append(event))
    assert [event[0] for event in events] == ['checking', 'ready']
