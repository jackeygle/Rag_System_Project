"""Score-filtered retrieval with optional document scope."""
from langchain_core.runnables import RunnableLambda
from config import TOP_K, MIN_RELEVANCE


def create_retriever(vector_store, k=None, sources=None, min_relevance=None):
    limit = TOP_K if k is None else k
    threshold = MIN_RELEVANCE if min_relevance is None else min_relevance
    if limit < 1 or not 0 <= threshold <= 1:
        raise ValueError("Invalid retrieval parameters")
    def retrieve(question):
        # None means the whole library; [] explicitly means no documents.
        if sources == []:
            return []
        kwargs = {}
        if sources is not None:
            kwargs["filter"] = {"source": sources[0]} if len(sources) == 1 else {"source": {"$in": sources}}
        hits = vector_store.similarity_search_with_relevance_scores(question, k=limit, **kwargs)
        return [doc for doc, score in hits if score >= threshold and doc.page_content.strip()]
    return RunnableLambda(retrieve)
