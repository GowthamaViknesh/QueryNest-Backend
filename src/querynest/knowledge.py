"""Knowledge layer (M9): business terminology + example SQL + agent memory, found by retrieval.

RAG = Retrieval-Augmented Generation: before asking the LLM, look up the knowledge relevant to
THIS question and put it in the prompt. The LLM then uses our definitions ("revenue" means
SUM(salesamt)) and proven query patterns instead of guessing.

Retrieval here is keyword-based (TF-IDF cosine similarity), computed in Python: no extra
service, no API calls, works offline. With thousands of examples you would switch to
embeddings (vector search); the interface below stays the same.

Memory: when an answer used SQL successfully, the question + SQL are saved as a "learned"
example. A thumbs-up verifies it (ranked higher); a thumbs-down deletes it.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass

from sqlalchemy import select

from querynest.appdb import GlossaryTerm, SqlExample, session

STOPWORDS = set("""a an the of for in on at to by with and or is are was were be been what which who
whom how many much show me give list all our we us i my from per each did do does this that
these those it its as than then there their please can could would should""".split())
MAX_EXAMPLES = 3
MIN_SIMILARITY = 0.25


def tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    # crude stemming: customers -> customer, sales stays sales (too short to strip meaningfully)
    return [w[:-1] if len(w) > 4 and w.endswith("s") else w for w in words if w not in STOPWORDS]


def cosine(a: Counter, b: Counter, idf: dict[str, float]) -> float:
    dot = sum(a[t] * b[t] * idf.get(t, 1.0) ** 2 for t in a if t in b)
    na = math.sqrt(sum((c * idf.get(t, 1.0)) ** 2 for t, c in a.items()))
    nb = math.sqrt(sum((c * idf.get(t, 1.0)) ** 2 for t, c in b.items()))
    return dot / (na * nb) if na and nb else 0.0


@dataclass
class Knowledge:
    terms: list[GlossaryTerm]
    examples: list[tuple[SqlExample, float]]

    def as_prompt(self) -> str:
        """Text added before the user's question. Empty when nothing relevant was found."""
        parts = []
        if self.terms:
            parts.append("Business terms used in this question:")
            for t in self.terms:
                hint = f" SQL: {t.sql_hint}" if t.sql_hint else ""
                parts.append(f"- {t.term}: {t.meaning}{hint}")
        if self.examples:
            parts.append("Similar questions answered before (reuse the approach if it fits):")
            for ex, _ in self.examples:
                label = "verified" if ex.verified else "unverified"
                parts.append(f"- Q ({label}): {ex.question}\n  SQL: {ex.sql}")
        return "\n".join(parts)


def retrieve(question: str) -> Knowledge:
    q_tokens = tokens(question)
    q_set = set(q_tokens)
    with session() as s:
        terms = s.scalars(select(GlossaryTerm)).all()
        examples = s.scalars(select(SqlExample)).all()

    # Glossary: a term matches if all words of the term (or of a synonym) are in the question
    matched = []
    for term in terms:
        for phrase in [term.term, *term.synonyms]:
            words = set(tokens(phrase))
            if words and words <= q_set:
                matched.append(term)
                break

    # Examples: TF-IDF cosine similarity; verified examples get a boost
    docs = [Counter(tokens(e.question)) for e in examples]
    df = Counter(t for d in docs for t in d)
    idf = {t: math.log((1 + len(docs)) / (1 + n)) + 1 for t, n in df.items()}
    q_vec = Counter(q_tokens)
    scored = []
    for ex, doc in zip(examples, docs):
        score = cosine(q_vec, doc, idf) * (1.3 if ex.verified else 1.0)
        if score >= MIN_SIMILARITY:
            scored.append((ex, score))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return Knowledge(terms=matched, examples=scored[:MAX_EXAMPLES])


def mark_used(examples: list[tuple[SqlExample, float]]) -> None:
    if not examples:
        return
    with session() as s:
        for ex, _ in examples:
            row = s.get(SqlExample, ex.id)
            if row:
                row.uses += 1
        s.commit()


def remember(question: str, sql: str, user_id: int | None) -> int | None:
    """Agent memory: store a successful question -> SQL pair (unverified). Returns its id."""
    question = question.strip()
    with session() as s:
        existing = s.scalars(select(SqlExample).where(SqlExample.question == question)).first()
        if existing:
            if not existing.verified:
                existing.sql = sql  # keep the latest working SQL for an unverified example
                s.commit()
            return existing.id
        example = SqlExample(question=question, sql=sql, source="learned", verified=False, created_by=user_id)
        s.add(example)
        s.commit()
        return example.id


def feedback(example_id: int, helpful: bool) -> None:
    """Thumbs up verifies a learned example; thumbs down removes it (curated ones are kept)."""
    with session() as s:
        example = s.get(SqlExample, example_id)
        if example is None:
            return
        if helpful:
            example.verified = True
        elif example.source == "learned":
            s.delete(example)
        s.commit()


# ---------------------------------------------------------------------------
# Starter knowledge for the sales data (loaded by `uv run seed`)
# ---------------------------------------------------------------------------
SEED_GLOSSARY = [
    ("revenue", ["sales", "turnover", "income"], "Net sales amount of invoice lines, in OMR.",
     "SUM(salesamt) FROM sales.invoices"),
    ("gross sales", ["gross amount"], "Sales before discount.", "SUM(grossamt) FROM sales.invoices"),
    ("discount", ["discounts"], "Discount given on invoice lines.", "SUM(totdisamt) FROM sales.invoices"),
    ("cost", ["cost of sales", "cogs"], "Cost value of goods sold.", "SUM(costvalue) FROM sales.invoices"),
    ("margin", ["profit", "gross profit"], "Sales minus cost.",
     "SUM(salesamt) - SUM(costvalue); margin % = (SUM(salesamt)-SUM(costvalue)) / NULLIF(SUM(salesamt),0) * 100"),
    ("vat", ["tax", "output vat"], "VAT collected is posted in the ledger, not in invoices.",
     "SUM(cramount) - SUM(dramount) FROM sales.account_entries WHERE accountname ILIKE '%VAT%'"),
    ("salesman", ["sales rep", "salesperson", "sales person", "seller"], "The employee who made the sale.",
     "salesmna column of sales.invoices"),
    ("customer", ["client", "buyer"], "Who bought. Name in customername, code in customercode.",
     "customername column of sales.invoices"),
    ("category", ["product line", "sales category", "segment"], "Sales category, e.g. 'BATTERY SALES'.",
     "mname column of sales.invoices"),
    ("brand", ["make"], "Product brand.", "brandname column of sales.invoices"),
    ("branch", ["location", "store"], "Selling branch code.", "branchcode column of sales.invoices"),
    ("month", ["monthly"], "Calendar month of the invoice date.",
     "date_trunc('month', docdt) or cmn (1-12) with cyyyy for the year"),
    ("year", ["yearly", "annual"], "Calendar year of the invoice.", "cyyyy column (or EXTRACT(YEAR FROM docdt))"),
    ("quantity", ["qty", "units"], "Units sold.", "SUM(totqty) FROM sales.invoices"),
    ("invoice", ["bill", "invoices"], "One invoice can have several lines (rows).",
     "COUNT(DISTINCT docid) counts invoices; COUNT(*) counts lines"),
]

SEED_EXAMPLES = [
    ("Top 10 customers by revenue in 2025",
     "SELECT customername, SUM(salesamt) AS revenue FROM sales.invoices WHERE cyyyy = 2025 "
     "GROUP BY customername ORDER BY revenue DESC LIMIT 10"),
    ("Monthly revenue trend for 2025",
     "SELECT date_trunc('month', docdt)::date AS month, SUM(salesamt) AS revenue FROM sales.invoices "
     "WHERE cyyyy = 2025 GROUP BY 1 ORDER BY 1"),
    ("Revenue share by sales category this year",
     "SELECT mname AS category, SUM(salesamt) AS revenue FROM sales.invoices "
     "WHERE cyyyy = EXTRACT(YEAR FROM CURRENT_DATE) GROUP BY mname ORDER BY revenue DESC"),
    ("Total VAT collected per year",
     "SELECT EXTRACT(YEAR FROM vchdt)::int AS year, SUM(cramount) - SUM(dramount) AS vat "
     "FROM sales.account_entries WHERE accountname ILIKE '%VAT%' GROUP BY 1 ORDER BY 1"),
    ("Salesman performance: revenue and number of invoices in 2025",
     "SELECT salesmna AS salesman, SUM(salesamt) AS revenue, COUNT(DISTINCT docid) AS invoices "
     "FROM sales.invoices WHERE cyyyy = 2025 GROUP BY salesmna ORDER BY revenue DESC"),
]


def seed() -> tuple[int, int]:
    added_terms = added_examples = 0
    with session() as s:
        have_terms = set(s.scalars(select(GlossaryTerm.term)).all())
        for term, synonyms, meaning, hint in SEED_GLOSSARY:
            if term not in have_terms:
                s.add(GlossaryTerm(term=term, synonyms=synonyms, meaning=meaning, sql_hint=hint))
                added_terms += 1
        have_q = set(s.scalars(select(SqlExample.question)).all())
        for question, sql in SEED_EXAMPLES:
            if question not in have_q:
                s.add(SqlExample(question=question, sql=sql, source="curated", verified=True))
                added_examples += 1
        s.commit()
    return added_terms, added_examples
