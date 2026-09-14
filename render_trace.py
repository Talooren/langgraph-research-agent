"""Рендерит last_run.json (записанный трейс выполнения агента) в статичную HTML-страницу.

Запуск: python render_trace.py [путь_к_run.json] [путь_к_output.html]
По умолчанию: last_run.json -> trace.html
"""
import json
import sys
from pathlib import Path

NODE_LABELS = {
    "analyze_query": "Analyze Query",
    "search": "Search",
    "evaluate_results": "Evaluate Results",
    "rewrite_query": "Rewrite Query",
    "generate_answer": "Generate Answer",
    "verify_answer": "Verify Answer",
}


def escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render(run_data: dict) -> str:
    trace = run_data["trace"]
    question = run_data["question"]
    answer = run_data["answer"]
    attempts = run_data["attempts"]
    elapsed = run_data.get("_elapsed_sec", "?")
    search_results = run_data.get("search_results", [])
    searches_count = sum(1 for s in trace if s["node"] == "search")

    steps_html = ""
    for i, step in enumerate(trace):
        label = NODE_LABELS.get(step["node"], step["node"])
        detail = step.get("detail", {})
        detail_html = ""
        if detail:
            rows = []
            for k, v in detail.items():
                v_str = json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else str(v)
                if len(v_str) > 200:
                    v_str = v_str[:200] + "…"
                rows.append(f'<div class="detail-row"><span class="detail-key">{escape(k)}</span><span class="detail-val">{escape(v_str)}</span></div>')
            detail_html = f'<div class="step-detail">{"".join(rows)}</div>'

        steps_html += f"""
        <div class="trace-step">
          <div class="step-marker">
            <div class="step-dot">✓</div>
            {'<div class="step-line"></div>' if i < len(trace) - 1 else ''}
          </div>
          <div class="step-body">
            <div class="step-title">{escape(label)}</div>
            <div class="step-summary">{escape(step['summary'])}</div>
            {detail_html}
          </div>
        </div>"""

    sources_html = ""
    for r in search_results[:8]:
        sources_html += f"""
        <div class="source-card">
          <div class="source-head">
            <span class="project-tag">{escape(r['project'])}</span>
            <span class="doc-tag">{escape(r['doc_type'])}</span>
            <span class="heading">{escape(r['heading'])}</span>
          </div>
          <div class="source-score">cosine {r['score']:.3f}</div>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LangGraph Research Agent — запись выполнения</title>
<style>
:root {{
  --bg: #10131a; --panel: #171b24; --panel-raised: #1e2330; --border: #2a3040;
  --text: #dbe1ec; --text-dim: #838fa3; --text-faint: #566078;
  --accent: #6fa8c9; --accent-dim: #4a7691; --accent-glow: rgba(111,168,201,0.14);
  --ok: #6bbf8a;
  --mono: 'JetBrains Mono', 'SF Mono', Consolas, monospace;
  --sans: 'Inter', -apple-system, sans-serif;
}}
* {{ box-sizing: border-box; }}
html, body {{ margin:0; padding:0; background:var(--bg); color:var(--text); font-family:var(--sans); }}
body {{ background-image: radial-gradient(circle at 15% 0%, rgba(111,168,201,0.06), transparent 45%); }}
.wrap {{ max-width: 900px; margin:0 auto; padding: 48px 24px 80px; }}
.kicker {{ font-family: var(--mono); font-size:12.5px; color:var(--text-faint); margin-bottom:14px; display:flex; align-items:center; gap:8px; }}
.kicker .dot {{ width:6px; height:6px; border-radius:50%; background:var(--accent); box-shadow:0 0 8px var(--accent); }}
h1 {{ font-size:26px; font-weight:600; margin:0 0 16px; letter-spacing:-0.01em; line-height:1.3; }}
h1 em {{ font-style:normal; color:var(--accent); }}
.question-box {{ background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:18px 20px; margin-bottom:24px; font-size:15px; }}
.question-box .q-label {{ font-family:var(--mono); font-size:11px; color:var(--text-faint); margin-bottom:8px; }}
.stats-row {{ display:flex; gap:18px; font-family:var(--mono); font-size:12px; color:var(--text-faint); margin-bottom:32px; flex-wrap:wrap; }}
.stats-row strong {{ color:var(--accent); font-weight:600; }}
h2 {{ font-size:13px; font-family:var(--mono); color:var(--text-faint); font-weight:500; margin: 36px 0 18px; text-transform:uppercase; letter-spacing:0.03em; }}
.trace-step {{ display:flex; gap:16px; }}
.step-marker {{ display:flex; flex-direction:column; align-items:center; }}
.step-dot {{ width:26px; height:26px; border-radius:50%; background:var(--accent-glow); border:1.5px solid var(--accent-dim); color:var(--accent); display:flex; align-items:center; justify-content:center; font-size:13px; flex-shrink:0; }}
.step-line {{ width:2px; flex:1; background:var(--border); margin:4px 0; }}
.step-body {{ padding-bottom:24px; flex:1; }}
.step-title {{ font-family:var(--mono); font-size:13.5px; font-weight:600; color:var(--text); margin-bottom:4px; }}
.step-summary {{ font-size:13.5px; color:var(--text-dim); line-height:1.5; }}
.step-detail {{ margin-top:10px; padding:10px 14px; background:var(--panel); border:1px solid var(--border); border-radius:8px; }}
.detail-row {{ display:flex; gap:8px; font-family:var(--mono); font-size:11px; padding:3px 0; }}
.detail-key {{ color:var(--text-faint); min-width:90px; flex-shrink:0; }}
.detail-val {{ color:var(--text-dim); word-break:break-word; }}
.answer-box {{ background:var(--panel); border:1px solid var(--accent-dim); border-radius:10px; padding:20px 22px; font-size:14.5px; line-height:1.65; white-space:pre-wrap; }}
.sources-grid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(220px,1fr)); gap:10px; }}
.source-card {{ background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:12px 14px; }}
.source-head {{ display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin-bottom:6px; }}
.project-tag {{ font-family:var(--mono); font-size:10.5px; color:var(--accent); background:var(--accent-glow); padding:2px 7px; border-radius:5px; }}
.doc-tag {{ font-family:var(--mono); font-size:10.5px; color:var(--text-faint); }}
.heading {{ font-size:12.5px; font-weight:500; color:var(--text); }}
.source-score {{ font-family:var(--mono); font-size:11px; color:var(--text-faint); }}
footer {{ margin-top:48px; padding-top:24px; border-top:1px solid var(--border); color:var(--text-faint); font-size:12.5px; font-family:var(--mono); }}
footer a {{ color:var(--text-dim); text-decoration:none; }}
footer a:hover {{ color:var(--accent); }}
</style>
</head>
<body>
<div class="wrap">

  <div class="kicker"><span class="dot"></span>LANGGRAPH RESEARCH AGENT — ЗАПИСЬ РЕАЛЬНОГО ВЫПОЛНЕНИЯ</div>
  <h1>Stateful граф с <em>conditional routing</em> и циклом</h1>

  <div class="question-box">
    <div class="q-label">ВОПРОС</div>
    {escape(question)}
  </div>

  <div class="stats-row">
    <span>Шагов графа: <strong>{len(trace)}</strong></span>
    <span>Итераций поиска: <strong>{searches_count}</strong></span>
    <span>Попыток: <strong>{attempts}</strong></span>
    <span>Время выполнения: <strong>{elapsed}с</strong></span>
    <span>Модель: <strong>claude-haiku-4-5</strong></span>
  </div>

  <h2>ВЫПОЛНЕНИЕ ГРАФА</h2>
  <div class="trace">
    {steps_html}
  </div>

  <h2>ОТВЕТ</h2>
  <div class="answer-box">{escape(answer)}</div>

  <h2>ИСПОЛЬЗОВАННЫЕ ИСТОЧНИКИ ({len(search_results)})</h2>
  <div class="sources-grid">
    {sources_html}
  </div>

  <footer>
    Это запись реального выполнения LangGraph-агента (не имитация) — LLM-вызовы делались
    напрямую через Anthropic API с локальным ключом. Страница статична и не выполняет
    LLM-вызовов сама — безопасно публиковать на GitHub Pages без риска утечки ключа.
    <br><br>
    <a href="https://github.com/Talooren" target="_blank">github.com/Talooren →</a>
  </footer>

</div>
</body>
</html>"""


if __name__ == "__main__":
    run_path = sys.argv[1] if len(sys.argv) > 1 else "last_run.json"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "trace.html"

    with open(run_path, encoding="utf-8") as f:
        run_data = json.load(f)

    html = render(run_data)
    Path(out_path).write_text(html, encoding="utf-8")
    print(f"Записано: {out_path}")
