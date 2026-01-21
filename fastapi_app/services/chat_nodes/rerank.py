import json
from typing import Dict

from services.chat_nodes.config import RERANK_K
from services.chat_nodes.intent import is_date_query
from services.chat_nodes.llm_clients import detect_llm
from services.chat_nodes.state import GraphState, slim_retrievals
from utils.geo import append_node_trace_result


async def rerank_node(state: GraphState) -> Dict:
    """Re-rank the candidate list using an LLM selector."""
    retrievals = state.get("retrievals") or []
    if not retrievals:
        result = {"retrievals": []}
        append_node_trace_result(state.get("query", ""), "rerank", result)
        return result

    # desired_k is bounded by available candidates and request top_k.
    desired_k = max(1, min(state.get("top_k") or RERANK_K, len(retrievals)))

    def _build_ctx(r: dict) -> str:
        meta = r.get("meta") or {}
        name = meta.get("name") or meta.get("title") or "장소"
        summary = meta.get("summary_one_sentence") or meta.get("description") or meta.get("content") or ""
        addr = meta.get("address") or meta.get("location", {}).get("addr1") or ""
        kw = meta.get("keywords") or []
        kw_str = ", ".join([str(k) for k in kw]) if kw else ""
        popularity = []
        if meta.get("views") is not None:
            popularity.append(f"조회수 {meta['views']}")
        if meta.get("likes") is not None:
            popularity.append(f"좋아요 {meta['likes']}")
        if meta.get("bookmarks") is not None:
            popularity.append(f"북마크 {meta['bookmarks']}")
        rating_parts = []
        if meta.get("starts"):
            rating_parts.append(f"평점 {meta['starts']}")
        if meta.get("counts"):
            rating_parts.append(f"리뷰 {meta['counts']}")
        rating_str = ", ".join(rating_parts)
        pop_str = ", ".join(popularity)
        parts = [name, summary]
        if addr:
            parts.append(f"주소: {addr}")
        if kw_str:
            parts.append(f"키워드: {kw_str}")
        if pop_str:
            parts.append(f"인기: {pop_str}")
        if rating_str:
            parts.append(rating_str)
        return " ".join([p for p in parts if p])

    candidates_txt = "\n".join([f"[id={i}] {_build_ctx(r)}" for i, r in enumerate(retrievals)])
    user_query = state.get("query", "")
    messages = [
        (
            "system",
            f"사용자 질문과 아래 후보를 보고 가장 관련 높은 상위 {desired_k}개를 고르고, "
            "id만 JSON 배열로 반환해. 예: [0,2]. 다른 텍스트는 넣지 말 것.",
        ),
    ]
    if is_date_query(user_query):
        messages.append(
            (
                "system",
                "사용자 의도는 데이트/커플이다. 분위기, 로맨틱함, 기념일/특별한 경험, 조용함을 우선 고려하라.",
            )
        )
    messages.extend(
        [
            ("system", f"후보:\n{candidates_txt}"),
            ("user", user_query),
        ]
    )

    raw = None
    try:
        # LLM returns a JSON array of indices (e.g., [0,2]).
        resp = await detect_llm.ainvoke(messages, max_tokens=50)
        raw = resp.content or ""
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            idxs = []
            for v in parsed:
                try:
                    iv = int(v)
                except Exception:
                    continue
                if 0 <= iv < len(retrievals) and iv not in idxs:
                    idxs.append(iv)
            if idxs:
                selected = [retrievals[i] for i in idxs[:desired_k]]
                if len(selected) < desired_k:
                    # Fill remaining slots with non-selected items in original order.
                    remaining = [r for j, r in enumerate(retrievals) if j not in idxs]
                    selected.extend(remaining[: max(0, desired_k - len(selected))])
                result = {"retrievals": selected}
                append_node_trace_result(
                    state.get("query", ""),
                    "rerank",
                    {
                        "raw": raw,
                        "idxs": idxs,
                        "retrievals": slim_retrievals(result["retrievals"]),
                    },
                )
                return result
    except Exception:
        pass

    # 파싱 실패 시 fallback: 상위 desired_k 사용
    selected = retrievals[:desired_k]
    result = {"retrievals": selected}
    append_node_trace_result(
        state.get("query", ""),
        "rerank",
        {
            "raw": raw,
            "idxs": None,
            "retrievals": slim_retrievals(result["retrievals"]),
        },
    )
    return result
