"""Structured generator synthesis module for multi-hop reasoning."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple
import urllib.error
import urllib.request

from src.models import RetrievalResult, SynthesisOutput, TraversalPath

logger = logging.getLogger(__name__)


GENERATOR_PROMPT_TEMPLATE = """You are a grounded multi-hop question answering system.
Answer the following question using ONLY the provided structured relational paths and retrieved passages.

Instructions:
1. First, trace the relational bridge and intermediate hops step by step in the Reasoning field.
2. In the Answer field, provide ONLY the concise target entity name or short phrase (1 to 5 words, e.g. "Chicago" or "Miquette Giraudy").
   Do NOT output full sentences, preambles, introductory clauses, or trailing punctuation in the Answer field.
3. In the Citations field, cite the exact supporting [Chunk ID: ...] values used for each step of your reasoning.

{context}

Question: {question}

Structured Response Format:
Reasoning: <Step-by-step trace of intermediate relational connections>
Answer: <Concise entity name or short noun phrase only>
Citations: <Comma-separated list of chunk IDs, e.g. chunk1, chunk2>
"""


def clean_answer_span(raw_answer: str) -> str:
    """Normalize and clean raw answer string to extract concise entity span."""
    if not raw_answer:
        return ""

    s = raw_answer.strip()

    # If multiline, take first non-empty line
    lines = [line.strip() for line in s.splitlines() if line.strip()]
    if lines:
        s = lines[0]

    prefixes = [
        r"^the\s+answer\s+is\s*:?\s*",
        r"^it\s+is\s*:?\s*",
        r"^it\s+was\s*:?\s*",
        r"^he\s+was\s+born\s+in\s*:?\s*",
        r"^she\s+was\s+born\s+in\s*:?\s*",
        r"^located\s+in\s*:?\s*",
        r"^answer\s*:?\s*",
    ]

    strip_chars = "\"' *`_.,;:!?~#\t\n\r"
    for _ in range(5):
        orig = s
        s = s.strip(strip_chars)
        for pat in prefixes:
            s = re.sub(pat, "", s, flags=re.IGNORECASE).strip()
        if s == orig:
            break

    return s


class HeuristicSpanExtractor:
    """Deterministic entity span extractor for offline benchmarking and fallback synthesis."""

    def extract_span(
        self,
        query: str,
        context: str,
        retrieval_result: Optional[RetrievalResult] = None,
    ) -> SynthesisOutput:
        """Extract candidate answer span by inspecting graph paths and passage entity mentions."""
        citations = re.findall(r"\[Chunk ID:\s*([a-f0-9]{16}|[a-zA-Z0-9_-]+)\]", context)
        q_lower = query.lower()

        candidate_answer = ""
        reasoning_trace = "Extracted via multi-hop path and passage alignment."

        # 1. Check structured traversal paths in retrieval result
        if retrieval_result and retrieval_result.subgraph_paths:
            scored_candidates: List[Tuple[float, float, str, str]] = []

            for path in retrieval_result.subgraph_paths:
                target = path.target.strip()
                bridge = path.bridge.strip()
                seed = path.seed.strip()
                p1 = path.predicate_1.upper()
                p2 = path.predicate_2.upper()
                terminal_predicate = p1 if path.hop_count == 1 else p2

                # Seed or bridge should match the query to anchor the multi-hop chain
                is_anchored = seed.lower() in q_lower or (bridge.lower() in q_lower and len(bridge) > 3)
                if not is_anchored:
                    continue

                # Multi-hop candidate: target
                if target and target.lower() not in q_lower and target.lower() != seed.lower():
                    # Base score based on hop depth (multi-hop queries prefer deeper paths)
                    score = path.hop_count * 15.0

                    # Penalize 1-hop paths when target is just the bridge
                    if path.hop_count == 1 or target.lower() == bridge.lower():
                        score -= 20.0

                    # Match the answer relation before preferring a deeper path.
                    alignment_score = 0.0
                    if any(w in q_lower for w in ["city", "born", "birth"]) and any(p in terminal_predicate for p in ["BORN", "BIRTH", "CITY"]):
                        alignment_score += 25.0
                    if any(w in q_lower for w in ["award", "win", "won", "prize"]) and any(p in terminal_predicate for p in ["WON", "AWARD", "PRIZE"]):
                        alignment_score += 25.0
                    if any(w in q_lower for w in ["country", "nation", "located", "where", "state"]) and any(p in terminal_predicate for p in ["LOCATED", "LOCATION", "COUNTRY"]):
                        alignment_score += 25.0
                    if any(w in q_lower for w in ["university", "college", "school", "studied", "attend"]) and any(p in terminal_predicate for p in ["STUDIED", "AFFILIATION", "ATTENDED"]):
                        alignment_score += 20.0
                    if any(w in q_lower for w in ["inventor", "invented", "dynamo", "founder", "founded", "director"]) and any(p in terminal_predicate for p in ["INVENTED", "FOUNDED", "DIRECTED"]):
                        alignment_score += 15.0
                    if any(w in q_lower for w in ["spouse", "married", "wife", "husband"]) and any(p in terminal_predicate for p in ["SPOUSE", "MARRIED"]):
                        alignment_score += 25.0

                    # Context passage presence bonus
                    if target.lower() in context.lower():
                        score += 10.0

                    trace = f"Connected ({seed}) -> ({bridge}) -> ({target}) via [{path.predicate_2}]"
                    scored_candidates.append((alignment_score, score, target, trace))

                # If no target or 1-hop fallback
                elif bridge and bridge.lower() not in q_lower and bridge.lower() != seed.lower():
                    score = 5.0
                    if bridge.lower() in context.lower():
                        score += 5.0
                    trace = f"Connected ({seed}) -> ({bridge})"
                    scored_candidates.append((0.0, score, bridge, trace))

            if scored_candidates:
                scored_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
                candidate_answer = scored_candidates[0][2]
                reasoning_trace = scored_candidates[0][3]

        # 2. If no candidate from paths, parse structured triples in context text
        if not candidate_answer:
            triples = re.findall(
                r"\(([^)]+)\)\s*-+\[:?([^\]]+)\]-+>\s*\(([^)]+)\)",
                context,
            )
            for src, pred, tgt in triples:
                src_clean = src.strip()
                tgt_clean = tgt.strip()
                if src_clean.lower() in q_lower and tgt_clean.lower() not in q_lower:
                    candidate_answer = tgt_clean
                    reasoning_trace = f"Relational fact: ({src_clean}) -[:{pred}]-> ({tgt_clean})"
                    break
                elif tgt_clean.lower() in q_lower and src_clean.lower() not in q_lower:
                    candidate_answer = src_clean
                    reasoning_trace = f"Inverse relational fact: ({tgt_clean}) <-[:{pred}]- ({src_clean})"
                    break

        # 3. Fallback: Identify salient named entities in the supporting context
        if not candidate_answer:
            # Formatting headers, document titles, and IDs are not answer evidence.
            passage_text = (
                "\n".join(chunk.text for chunk in retrieval_result.top_chunks)
                if retrieval_result is not None else context
            )
            entities = re.findall(
                r"\b[A-Z\u00C0-\u024F][\w\u00C0-\u024F]*(?:\s+(?:du|de|la|le|of|the|and|in|at|van|von|da|di)\s+[A-Z\u00C0-\u024F][\w\u00C0-\u024F]*|\s+[A-Z\u00C0-\u024F][\w\u00C0-\u024F]*)*\b",
                passage_text,
            )
            stop_words = {
                "The", "A", "An", "In", "On", "At", "By", "For", "With", "Chunk",
                "ID", "Retrieved", "Passages", "Structured", "Relational", "Paths"
            }
            valid_candidates = [e for e in entities if e not in stop_words and e.lower() not in q_lower and len(e) > 2]
            if valid_candidates:
                # Preserve the supplied passage order when selecting a candidate.
                candidate_answer = valid_candidates[0]

        if not candidate_answer:
            candidate_answer = "Unknown"

        return SynthesisOutput(
            reasoning_trace=reasoning_trace,
            final_answer=clean_answer_span(candidate_answer),
            cited_chunk_ids=citations[:3],
        )


class OpenAIProvider:
    """OpenAI API provider for synthesis generation."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1/chat/completions",
        timeout: float = 15.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.timeout = timeout

    def __call__(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            res_obj = json.loads(resp.read().decode("utf-8"))
            return str(res_obj["choices"][0]["message"]["content"])


class OllamaProvider:
    """Local Ollama endpoint provider for synthesis generation."""

    def __init__(
        self,
        model: str = "llama3:8b",
        endpoint: str = "http://localhost:11434/api/generate",
        timeout: float = 30.0,
    ) -> None:
        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout

    def __call__(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            res_obj = json.loads(resp.read().decode("utf-8"))
            return str(res_obj.get("response", ""))


class MockLLMProvider:
    """Deterministic mock provider for unit and regression testing."""

    def __init__(self, responses: Optional[Dict[str, str]] = None) -> None:
        self.responses = responses or {}

    def __call__(self, prompt: str) -> str:
        for k, v in self.responses.items():
            if k in prompt:
                return v
        return "Reasoning: Direct bridge match.\nAnswer: Mock Answer\nCitations: mock1"


class SynthesisGenerator:
    """Generates structured, grounded multi-hop answers with bridge tracing and citations."""

    def __init__(
        self,
        llm_callable: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.llm_callable = llm_callable
        self.heuristic_extractor = HeuristicSpanExtractor()

    def build_prompt(self, query: str, context: str) -> str:
        """Assemble the synthesis prompt with linearized context and question."""
        return GENERATOR_PROMPT_TEMPLATE.format(context=context, question=query)

    def parse_response(self, raw_text: str) -> SynthesisOutput:
        """Parse structured model response into typed output with cleaned answer span."""
        reasoning = ""
        answer = ""
        citations: List[str] = []

        reasoning_match = re.search(r"Reasoning:\s*(.*?)(?=\nAnswer:|\Z)", raw_text, re.DOTALL | re.IGNORECASE)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        answer_match = re.search(r"Answer:\s*(.*?)(?=\nCitations:|\Z)", raw_text, re.DOTALL | re.IGNORECASE)
        if answer_match:
            answer = answer_match.group(1).strip()
        elif not reasoning:
            answer = raw_text.strip()

        citations_match = re.search(r"Citations:\s*(.*)", raw_text, re.DOTALL | re.IGNORECASE)
        if citations_match:
            raw_cites = citations_match.group(1).strip()
            cites = re.findall(r"[a-f0-9]{16}|[a-zA-Z0-9_-]+", raw_cites)
            citations = [c for c in cites if c.lower() not in {"chunk", "id", "none"}]

        return SynthesisOutput(
            reasoning_trace=reasoning,
            final_answer=clean_answer_span(answer),
            cited_chunk_ids=citations,
        )

    def generate(
        self,
        query: str,
        context: str,
        retrieval_result: Optional[RetrievalResult] = None,
    ) -> SynthesisOutput:
        """Run generation via model with automatic fallback to heuristic span extractor."""
        if self.llm_callable:
            prompt = self.build_prompt(query, context)
            try:
                raw_response = self.llm_callable(prompt)
                parsed = self.parse_response(raw_response)
                if parsed.final_answer:
                    return parsed
            except Exception as e:
                logger.warning(f"Synthesis model call failed: {e}, using heuristic span extractor.")

        return self.heuristic_extractor.extract_span(query, context, retrieval_result)
