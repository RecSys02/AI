import importlib
import pathlib
import sys
import types


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_retriever_module():
    dummy_st = types.ModuleType("sentence_transformers")

    class DummySentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass

    dummy_st.SentenceTransformer = DummySentenceTransformer
    sys.modules.setdefault("sentence_transformers", dummy_st)

    if "services.retriever" in sys.modules:
        del sys.modules["services.retriever"]

    return importlib.import_module("services.retriever")


def test_western_query_enables_keyword_filter():
    retriever = _load_retriever_module()

    use_filter, terms = retriever._needs_keyword_filter("강남 양식집 추천해줘", "restaurant")

    assert use_filter is True
    assert "양식" in terms
    assert "파스타" in terms


def test_western_terms_match_related_meta():
    retriever = _load_retriever_module()

    _, terms = retriever._needs_keyword_filter("강남 양식집 추천해줘", "restaurant")

    assert retriever._has_any_keyword(
        {"description": "수제 파스타와 스테이크가 유명한 분위기 좋은 레스토랑"},
        terms,
    )
    assert not retriever._has_any_keyword({"description": "곱창전골 전문점"}, terms)
