"""
LangGraph Research Agent — мини-агент, отвечающий на вопросы о проектах портфолио
(Elina, TradeBot Backtest, TaskFlow Scheduler Bot), используя TF-IDF retrieval
поверх их README/ARCHITECTURE.md как search tool.

Демонстрирует stateful workflow с conditional routing и циклом:

    analyze_query → search → evaluate_results
                                  │
                        достаточно? / нет?
                          │              │
                   generate_answer   rewrite_query
                          │              │
                       verify ←──── search (снова)
                          │
                         END

Запуск:
    export ANTHROPIC_API_KEY=...
    python agent.py "Как работает Near-SL Guard в TradeBot?"

Ключ читается только из переменной окружения / .env — никогда не хардкодится
и не появляется в записанном трейсе выполнения.
"""
from dotenv import load_dotenv
load_dotenv()  # Загружает переменные из .env в окружение ДО всего остального

import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import TypedDict

from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic

from retriever import load_index

MODEL_NAME = "claude-haiku-4-5-20251001"
MAX_ATTEMPTS = 3


# ---------- State ----------

class ResearchState(TypedDict):
    question: str
    search_queries: list[str]
    search_results: list[dict]          # накопленные найденные чанки за все попытки
    answer: str
    needs_more_search: bool
    verification_passed: bool
    attempts: int
    trace: list[dict]                   # запись каждого шага для последующего рендера


# ---------- Инфраструктура ----------

def _log_step(state: ResearchState, node: str, summary: str, detail: dict | None = None) -> None:
    state["trace"].append({
        "node": node,
        "summary": summary,
        "detail": detail or {},
        "ts": datetime.now(timezone.utc).isoformat(),
    })


def get_llm() -> ChatAnthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY не найден в окружении. "
            "Задайте его через `export ANTHROPIC_API_KEY=...` или файл .env (не коммитить!)."
        )
    return ChatAnthropic(model=MODEL_NAME, temperature=0, max_tokens=1024)


_INDEX = None


def get_index():
    global _INDEX
    if _INDEX is None:
        _INDEX = load_index("chunks.json")
    return _INDEX


# ---------- Nodes ----------

def analyze_query(state: ResearchState) -> ResearchState:
    llm = get_llm()
    prompt = f"""Ты — исследовательский ассистент. Пользователь задал вопрос про технические проекты
(Elina — AI-ассистент в Telegram, TradeBot Backtest — торговый бот для Bybit,
TaskFlow Scheduler Bot — бот для управления задачами через Airtable).

Вопрос пользователя: {state['question']}

Сформулируй от 1 до 3 коротких поисковых запросов (по несколько слов каждый) для поиска
по документации этих проектов, которые помогут ответить на вопрос.
Ответь СТРОГО в формате JSON-массива строк, без пояснений. Например: ["near-sl guard", "риск-менеджмент"]"""

    response = llm.invoke(prompt)
    raw = response.content.strip()
    # снимаем возможные markdown-фенсы
    raw = raw.strip("`").removeprefix("json").strip()
    try:
        queries = json.loads(raw)
        if not isinstance(queries, list):
            raise ValueError
    except (json.JSONDecodeError, ValueError):
        queries = [state["question"]]

    state["search_queries"] = queries
    _log_step(state, "analyze_query", f"Сформировано {len(queries)} поисковых запросов", {"queries": queries})
    return state


def search(state: ResearchState) -> ResearchState:
    idx = get_index()
    new_results = []
    for q in state["search_queries"]:
        hits = idx.search(q, top_k=3)
        for h in hits:
            new_results.append({
                "query": q,
                "project": h["chunk"]["project"],
                "doc_type": h["chunk"]["doc_type"],
                "heading": h["chunk"]["heading"],
                "text": h["chunk"]["text"],
                "score": round(h["score"], 3),
            })

    # дедуп по (project, heading), сохраняя лучший score
    seen: dict[tuple, dict] = {}
    for r in state["search_results"] + new_results:
        key = (r["project"], r["heading"])
        if key not in seen or r["score"] > seen[key]["score"]:
            seen[key] = r
    state["search_results"] = sorted(seen.values(), key=lambda r: r["score"], reverse=True)

    _log_step(
        state, "search",
        f"Найдено {len(new_results)} новых совпадений по {len(state['search_queries'])} запрос(ам), всего в контексте {len(state['search_results'])}",
        {"queries": state["search_queries"], "new_hits": len(new_results)},
    )
    return state


def evaluate_results(state: ResearchState) -> ResearchState:
    llm = get_llm()
    context = "\n\n".join(
        f"[{r['project']} / {r['heading']}] {r['text'][:300]}"
        for r in state["search_results"][:6]
    )
    prompt = f"""Вопрос: {state['question']}

Найденный контекст:
{context}

Достаточно ли этого контекста, чтобы полно и точно ответить на вопрос?
Ответь СТРОГО одним словом: YES или NO."""

    response = llm.invoke(prompt)
    verdict = response.content.strip().upper()
    needs_more = "NO" in verdict and "YES" not in verdict
    state["needs_more_search"] = needs_more and state["attempts"] < MAX_ATTEMPTS

    _log_step(
        state, "evaluate_results",
        f"Вердикт: {'нужно больше данных' if state['needs_more_search'] else 'достаточно контекста'} (попытка {state['attempts']}/{MAX_ATTEMPTS})",
        {"raw_verdict": verdict, "attempts": state["attempts"]},
    )
    return state


def rewrite_query(state: ResearchState) -> ResearchState:
    llm = get_llm()
    prev_queries = ", ".join(state["search_queries"])
    prompt = f"""Вопрос пользователя: {state['question']}
Предыдущие поисковые запросы (не дали достаточно результатов): {prev_queries}

Предложи 1-2 ДРУГИХ поисковых запроса — используй другие ключевые слова или синонимы,
чтобы найти недостающую информацию. Ответь СТРОГО в формате JSON-массива строк."""

    response = llm.invoke(prompt)
    raw = response.content.strip().strip("`").removeprefix("json").strip()
    try:
        new_queries = json.loads(raw)
        if not isinstance(new_queries, list) or not new_queries:
            raise ValueError
    except (json.JSONDecodeError, ValueError):
        new_queries = [state["question"] + " детали"]

    state["search_queries"] = new_queries
    state["attempts"] += 1
    _log_step(state, "rewrite_query", f"Переформулированы запросы: {new_queries}", {"queries": new_queries})
    return state


def generate_answer(state: ResearchState) -> ResearchState:
    llm = get_llm()
    context = "\n\n".join(
        f"[Источник: {r['project']} / {r['doc_type']} / {r['heading']}]\n{r['text']}"
        for r in state["search_results"][:6]
    )
    prompt = f"""Ответь на вопрос пользователя, используя ТОЛЬКО контекст ниже.
Если в контексте нет ответа — честно скажи, что информации недостаточно.
Отвечай кратко и по делу, на русском.

КОНТЕКСТ:
{context}

ВОПРОС: {state['question']}"""

    response = llm.invoke(prompt)
    state["answer"] = response.content.strip()
    _log_step(state, "generate_answer", "Ответ сгенерирован на основе найденного контекста", {"answer_preview": state["answer"][:200]})
    return state


def verify_answer(state: ResearchState) -> ResearchState:
    llm = get_llm()
    prompt = f"""Вопрос: {state['question']}
Ответ: {state['answer']}

Проверь: отвечает ли этот ответ на вопрос по существу, без явных противоречий с общим смыслом вопроса?
Ответь СТРОГО одним словом: PASS или FAIL."""

    response = llm.invoke(prompt)
    verdict = response.content.strip().upper()
    state["verification_passed"] = "PASS" in verdict

    _log_step(
        state, "verify_answer",
        f"Верификация: {'PASS' if state['verification_passed'] else 'FAIL'}",
        {"raw_verdict": verdict},
    )
    return state


# ---------- Conditional edges ----------

def route_after_evaluate(state: ResearchState) -> str:
    return "rewrite_query" if state["needs_more_search"] else "generate_answer"


def route_after_verify(state: ResearchState) -> str:
    # если верификация не прошла и есть ещё попытки — перегенерировать ответ
    if not state["verification_passed"] and state["attempts"] < MAX_ATTEMPTS:
        return "generate_answer"
    return END


# ---------- Сборка графа ----------

def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("analyze_query", analyze_query)
    graph.add_node("search", search)
    graph.add_node("evaluate_results", evaluate_results)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("verify_answer", verify_answer)

    graph.set_entry_point("analyze_query")
    graph.add_edge("analyze_query", "search")
    graph.add_edge("search", "evaluate_results")
    graph.add_conditional_edges("evaluate_results", route_after_evaluate, {
        "rewrite_query": "rewrite_query",
        "generate_answer": "generate_answer",
    })
    graph.add_edge("rewrite_query", "search")
    graph.add_edge("generate_answer", "verify_answer")
    graph.add_conditional_edges("verify_answer", route_after_verify, {
        "generate_answer": "generate_answer",
        END: END,
    })

    return graph.compile()


def run(question: str) -> ResearchState:
    app = build_graph()
    initial_state: ResearchState = {
        "question": question,
        "search_queries": [],
        "search_results": [],
        "answer": "",
        "needs_more_search": False,
        "verification_passed": False,
        "attempts": 1,
        "trace": [],
    }
    start = time.time()
    final_state = app.invoke(initial_state, config={"recursion_limit": 25})
    elapsed = round(time.time() - start, 2)
    final_state["_elapsed_sec"] = elapsed
    return final_state


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Как работает Near-SL Guard в TradeBot?"
    print(f"Вопрос: {question}\n")
    result = run(question)

    print("=== TRACE ===")
    for step in result["trace"]:
        print(f"  [{step['node']}] {step['summary']}")

    print(f"\n=== ОТВЕТ ===\n{result['answer']}")
    print(f"\nПопыток: {result['attempts']}, время: {result.get('_elapsed_sec', '?')}с")

    out_path = "last_run.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nТрейс сохранён в {out_path}")
