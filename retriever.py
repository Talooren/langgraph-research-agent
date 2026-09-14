"""TF-IDF retrieval поверх той же документации, что и в RAG-демо (rag-docs-search).
Используется как search tool внутри LangGraph Research Agent.
"""
import json
import math
import re
from pathlib import Path

STOPWORDS = set("""
и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только
ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни
быть был него до вас нибудь опять уж вам сказал ведь там потом себя ничего ей может они
тут где есть надо ней для мы тебя их чем была сам чтоб без будто человек чего раз тоже
себе под будет ж тогда кто этот того потому этого какой совсем ним здесь этом один почти
мой тем чтобы нее сейчас были куда зачем всех можно при наконец два об другой хоть после
над больше тот через эти нас про всего них какая много разве три эту моя впрочем хорошо
свою этой перед иногда лучше чуть том нельзя такой им более всегда конечно всю между
the a an and or of to in on for with is are was were be this that it as at by from into
""".split())

_TOKEN_RE = re.compile(r"[^\wа-яё0-9./_-]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    cleaned = text.lower()
    cleaned = cleaned.replace("«", " ").replace("»", " ").replace('"', " ").replace("'", " ").replace("`", " ")
    cleaned = _TOKEN_RE.sub(" ", cleaned)
    raw_tokens = [t for t in cleaned.split() if len(t) > 1 and t not in STOPWORDS]

    expanded = []
    for t in raw_tokens:
        expanded.append(t)
        if "-" in t and "/" not in t:
            for part in t.split("-"):
                if len(part) > 1 and part not in STOPWORDS:
                    expanded.append(part)
    return expanded


class TfidfIndex:
    """Тот же алгоритм, что и в браузерной версии: TF-IDF + cosine similarity."""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.doc_freq: dict[str, int] = {}
        self.vocab_idf: dict[str, float] = {}
        self.doc_vectors: list[dict[str, float]] = []
        self._build()

    def _build(self):
        n = len(self.chunks)
        tf_per_doc = []

        for chunk in self.chunks:
            tokens = tokenize(chunk["text"] + " " + chunk["heading"])
            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            tf_per_doc.append(tf)
            for term in tf:
                self.doc_freq[term] = self.doc_freq.get(term, 0) + 1

        for term, df in self.doc_freq.items():
            self.vocab_idf[term] = math.log((n + 1) / (df + 1)) + 1

        for tf in tf_per_doc:
            vec = {}
            norm_sq = 0.0
            for term, count in tf.items():
                w = count * self.vocab_idf[term]
                vec[term] = w
                norm_sq += w * w
            norm = math.sqrt(norm_sq) or 1.0
            for term in vec:
                vec[term] /= norm
            self.doc_vectors.append(vec)

    def search(self, query: str, top_k: int = 5, min_score: float = 0.001) -> list[dict]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        qtf: dict[str, int] = {}
        for t in q_tokens:
            qtf[t] = qtf.get(t, 0) + 1

        qvec = {}
        q_norm_sq = 0.0
        for term, count in qtf.items():
            idf = self.vocab_idf.get(term, math.log((len(self.chunks) + 1) / 1) + 1)
            w = count * idf
            qvec[term] = w
            q_norm_sq += w * w
        q_norm = math.sqrt(q_norm_sq) or 1.0
        for term in qvec:
            qvec[term] /= q_norm

        results = []
        for chunk, dvec in zip(self.chunks, self.doc_vectors):
            dot = 0.0
            matched = []
            for term, qw in qvec.items():
                if term in dvec:
                    dot += qw * dvec[term]
                    matched.append(term)
            if dot > min_score:
                results.append({"chunk": chunk, "score": dot, "matched_terms": matched})

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]


def load_index(chunks_path: str | Path = "chunks.json") -> TfidfIndex:
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)
    return TfidfIndex(chunks)


if __name__ == "__main__":
    idx = load_index()
    for q in ["near-sl guard", "trust уровни", "airtable интеграция"]:
        print(f"\n=== Запрос: {q} ===")
        for r in idx.search(q, top_k=3):
            c = r["chunk"]
            print(f"  [{r['score']:.3f}] {c['project']} / {c['doc_type']} / {c['heading']}")
