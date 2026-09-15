# LangGraph Research Agent

Stateful research-агент на LangGraph: отвечает на вопросы о трёх проектах портфолио (Elina, TradeBot Backtest, TaskFlow Scheduler Bot), используя TF-IDF retrieval по их документации как search tool. Показывает conditional routing и цикл — граф умеет переформулировать запрос и искать заново, если найденного контекста недостаточно.

## 🎯 Обзор

Агент проходит граф из 6 узлов: анализирует вопрос → формирует поисковые запросы → ищет по документации → оценивает, достаточно ли найдено → либо генерирует ответ, либо переформулирует запрос и ищет снова → верифицирует финальный ответ. При недостаточном контексте или провале верификации граф возвращается на предыдущие шаги — это и есть демонстрация stateful workflow, а не линейного пайплайна.

```
analyze_query → search → evaluate_results ─┬─ (нет) → rewrite_query → search (снова)
                                             └─ (да) → generate_answer → verify_answer ─┬─ PASS → END
                                                                                          └─ FAIL → generate_answer (снова)
```

## 📁 Структура

| Файл | Назначение |
|---|---|
| `agent.py` | Граф: состояние (`ResearchState`), 6 узлов, conditional edges, точка входа CLI |
| `retriever.py` | TF-IDF + cosine similarity поиск по документации — тот же алгоритм, что в [rag-docs-search](https://github.com/Talooren/rag-docs-search), портированный на Python |
| `render_trace.py` | Рендерит записанный трейс выполнения (`last_run.json`) в статичную HTML-страницу |
| `chunks.json` | Проиндексированные чанки документации трёх проектов |

## 🚀 Запуск

```bash
pip install -r requirements.txt
cp .env.example .env   # впишите свой OPENROUTER_API_KEY
export OPENROUTER_API_KEY=...

python agent.py "Как работает Near-SL Guard в TradeBot?"
python render_trace.py   # last_run.json -> trace.html
```

## 🔐 Безопасность

API-ключ читается только из переменной окружения / `.env` (в `.gitignore`, никогда не коммитится). Генерируемый `trace.html` — статичная страница без LLM-вызовов: безопасно публиковать на GitHub Pages, ключ туда не попадает.

## 🛠 Технологический стек

- Python 3.11+
- LangGraph (StateGraph, conditional edges, TypedDict state)
- LangChain + OpenRouter API (`qwen/qwen3-coder-next`)
- TF-IDF retrieval собственной реализации (без внешних ML-библиотек)

## 📦 Статус

🟢 Рабочий CLI-агент. `trace.html` в этом репозитории — пример записанного выполнения на реальном вопросе (см. `example-run/`).

---

*Дата создания: 2026-09-14*
