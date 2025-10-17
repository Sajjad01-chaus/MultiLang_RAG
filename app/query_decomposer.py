"""
app/query_decomposer.py

Hybrid query decomposition:
- Rule-based splitter for enumerations and connectors ("and", "or", ";", newline-separated lists).
- LLM fallback that rewrites/splits a complex query into smaller sub-queries.
- Recomposing step to merge sub-answer snippets with provenance.

Usage:
    decomposer = QueryDecomposer(llm_fn=call_llm)
    subqs = decomposer.decompose(query)
    answers = [retrieve_and_answer(sq) for sq in subqs]
    final = decomposer.recompose(query, subqs, answers)
"""

import re
from typing import List, Callable, Tuple, Optional

# simple rule-based heuristics
SPLIT_PATTERNS = [
    r"\band\b",
    r";",
    r"\n",
    r", then",  # conversational separators
    r"\bfirst\b|\bsecond\b|\bthird\b",  # enumerations
]

class QueryDecomposer:
    def __init__(self, llm_fn: Optional[Callable[[str], str]] = None, max_subqueries: int = 6):
        """
        llm_fn(prompt) -> str : optional LLM-based splitter / rewriter
        """
        self.llm_fn = llm_fn
        self.max_subqueries = max_subqueries

    def _rule_split(self, query: str) -> List[str]:
        # naive split by separators, but preserve short queries
        # first split by newline and semicolon, then by ' and ' if query length > threshold
        parts = re.split(r"\n|;", query)
        result = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if len(p.split()) > 6 and " and " in p.lower():
                # split by ' and ' for longer parts
                for sp in re.split(r"\band\b", p, flags=re.I):
                    sp = sp.strip()
                    if sp:
                        result.append(sp)
            else:
                result.append(p)
        # further split comma-separated lists that are actually distinct asks (heuristic)
        final = []
        for r in result:
            if len(r.split(",")) > 2 and len(r) > 60:
                final.extend([x.strip() for x in r.split(",") if x.strip()])
            else:
                final.append(r)
        # limit number of subqueries
        return final[: self.max_subqueries]

    def decompose(self, query: str) -> List[str]:
        """
        Return a list of subqueries.
        Steps:
          1. Try rule-based splitting.
          2. If still long/complex and llm_fn provided, ask LLM to split into 3-6 sub-questions.
        """
        rule_parts = self._rule_split(query)
        # If rule split yields multiple good parts -> return
        if len(rule_parts) >= 2:
            return rule_parts
        # If single part but long/complex, use LLM to rewrite as bullets
        if self.llm_fn and len(query.split()) > 20:
            prompt = (
                "Split the following user request into a short list of focused sub-questions. "
                "Return as a JSON array of strings. Keep each sub-question concise.\n\n"
                f"Request:\n{query}\n\nSub-questions:"
            )
            resp = self.llm_fn(prompt)
            # Attempt to parse JSON array; if not JSON, fallback to newline split
            import json
            try:
                arr = json.loads(resp)
                if isinstance(arr, list) and arr:
                    return [a.strip() for a in arr if isinstance(a, str) and a.strip()][: self.max_subqueries]
            except Exception:
                # fallback newline split
                lines = [l.strip("-* \t") for l in resp.splitlines() if l.strip()]
                if lines:
                    return lines[: self.max_subqueries]
        # default return single query
        return [query]

    def recompose(self, original_query: str, subqueries: List[str], answers: List[dict]) -> dict:
        """
        Recompose sub-answers into a final answer.
        `answers` expected to be a list of dicts with keys: {'answer': str, 'provenance': [...], 'score': float}
        Returns a dict: {'answer': ..., 'provenance': [...], 'notes': ...}
        If llm_fn provided, use it to synthesize.
        """
        if self.llm_fn:
            # build synthesis prompt
            parts = []
            for i, (sq, ans) in enumerate(zip(subqueries, answers)):
                parts.append(f"Subquery {i+1}: {sq}\nAnswer: {ans.get('answer','')}\nProvenance: {ans.get('provenance',[])}\n")
            prompt = (
                "You are a concise assistant. Combine the following sub-answers into a single, coherent answer to the user's original request. "
                "Keep provenance brief using the format [doc_id:page]. If information conflicts, state the conflict and cite sources.\n\n"
                f"Original request: {original_query}\n\n"
                "Sub-answers:\n" + "\n".join(parts)
            )
            synthesized = self.llm_fn(prompt)
            # return with aggregated provenance
            prov = []
            for a in answers:
                prov.extend(a.get("provenance", []))
            # dedupe provenance by id
            seen = set()
            unique_prov = []
            for p in prov:
                pid = str(p.get("doc_id")) + ":" + str(p.get("page", ""))
                if pid not in seen:
                    seen.add(pid)
                    unique_prov.append(p)
            return {"answer": synthesized, "provenance": unique_prov}
        else:
            # simple concatenation
            combined = "\n\n".join([f"Q: {sq}\nA: {ans.get('answer','')}" for sq, ans in zip(subqueries, answers)])
            prov = []
            for a in answers:
                prov.extend(a.get("provenance", []))
            return {"answer": combined, "provenance": prov}
